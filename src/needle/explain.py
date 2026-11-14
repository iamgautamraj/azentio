from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import json
import math
import os
import re
import httpx
import numpy as np
import pandas as pd
import xgboost as xgb

from needle.features import FEATURE_COLUMNS, FEATURE_LABELS
from needle.model import prepare_temporal_splits
from needle.schemas import (
    AlertExplanationResponse,
    Claim,
    EvidenceItem,
    FaithfulnessResult,
    LLMExplanationOutput,
)

BANNED_PHRASES: tuple[str, ...] = (
    "confirmed fraud",
    "definitely fraud",
    "proven fraud",
    "stolen card",
    "scammer",
    "fraudster",
    "fraudulent customer",
    "customer committed",
    "you committed",
    "criminal",
)

NUMERIC_TOLERANCE_ABS = 0.02
NUMERIC_TOLERANCE_REL = 0.015
MAX_NARRATIVE_WORDS = 80


def extract_top_evidence(
    model: xgb.XGBClassifier,
    features: dict[str, float],
    feature_columns: list[str] = FEATURE_COLUMNS,
    top_k: int = 5,
) -> list[EvidenceItem]:
    """Extracts top-k feature contributions using XGBoost native TreeSHAP (pred_contribs=True)."""
    row_df = pd.DataFrame([features])[feature_columns]
    dmat = xgb.DMatrix(row_df, feature_names=feature_columns)
    contribs = model.get_booster().predict(dmat, pred_contribs=True)[0]
    # Last element of pred_contribs is the bias term
    feat_contribs = contribs[: len(feature_columns)]

    ranked_indices = np.argsort(np.abs(feat_contribs))[::-1][:top_k]
    evidence: list[EvidenceItem] = []
    for idx in ranked_indices:
        fname = feature_columns[int(idx)]
        c_val = float(feat_contribs[int(idx)])
        raw_val = float(features[fname])
        clean_val: float | int = int(raw_val) if raw_val.is_integer() else round(raw_val, 4)
        direction = "raises_risk" if c_val >= 0.0 else "lowers_risk"
        evidence.append(
            EvidenceItem(
                feature=fname,
                label=FEATURE_LABELS.get(fname, fname),
                value=clean_val,
                contribution=round(c_val, 4),
                direction=direction,
            )
        )
    return evidence


def _numbers_match(claimed: float, allowed_values: list[float]) -> bool:
    for ref in allowed_values:
        if math.isclose(claimed, ref, rel_tol=NUMERIC_TOLERANCE_REL, abs_tol=NUMERIC_TOLERANCE_ABS):
            return True
    return False


def check_faithfulness(
    explanation: LLMExplanationOutput,
    evidence: list[EvidenceItem],
    txn_fields: dict[str, object] | None = None,
) -> FaithfulnessResult:
    """Deterministic faithfulness verifier checking feature membership, sign, numeric grounding, and tone."""
    reasons: list[str] = []
    evidence_map = {e.feature: e for e in evidence}

    words = explanation.narrative.strip().split()
    if len(words) > MAX_NARRATIVE_WORDS:
        reasons.append(f"Narrative exceeds {MAX_NARRATIVE_WORDS} words ({len(words)} words).")

    lower_narr = explanation.narrative.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lower_narr:
            reasons.append(f"Narrative contains disallowed accusation/certainty phrase: '{phrase}'.")

    allowed_numbers: list[float] = []
    for e in evidence:
        if isinstance(e.value, (int, float)):
            allowed_numbers.append(float(e.value))
    if txn_fields:
        for v in txn_fields.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                allowed_numbers.append(float(v))

    for claim in explanation.claims:
        if claim.feature not in evidence_map:
            reasons.append(f"Claimed feature '{claim.feature}' is not in top-5 model evidence.")
            continue
        ev = evidence_map[claim.feature]
        if claim.direction != ev.direction:
            reasons.append(
                f"Direction mismatch for '{claim.feature}': claimed '{claim.direction}', model '{ev.direction}'."
            )
        if isinstance(claim.value, (int, float)) and isinstance(ev.value, (int, float)):
            if not math.isclose(
                float(claim.value),
                float(ev.value),
                rel_tol=NUMERIC_TOLERANCE_REL,
                abs_tol=NUMERIC_TOLERANCE_ABS,
            ):
                reasons.append(
                    f"Value mismatch for '{claim.feature}': claimed {claim.value}, evidence has {ev.value}."
                )
        elif str(claim.value) != str(ev.value):
            reasons.append(f"Value mismatch for '{claim.feature}': claimed {claim.value}, evidence has {ev.value}.")

    # Check every number appearing in narrative against evidence + raw transaction fields
    found_nums = re.findall(r"-?\d+(?:\.\d+)?", explanation.narrative)
    for num_str in found_nums:
        val = float(num_str)
        if not _numbers_match(val, allowed_numbers):
            reasons.append(f"Unverified number '{num_str}' in narrative not grounded in evidence or transaction fields.")

    return FaithfulnessResult(passed=len(reasons) == 0, reasons=reasons)


def build_template_explanation(evidence: list[EvidenceItem]) -> LLMExplanationOutput:
    """Deterministic fallback explanation guaranteed to pass check_faithfulness."""
    top_items = evidence[:3]
    parts: list[str] = []
    claims: list[Claim] = []
    for item in top_items:
        verb = "increases" if item.direction == "raises_risk" else "reduces"
        parts.append(f"{item.label} ({item.value}) {verb} risk")
        claims.append(Claim(feature=item.feature, value=item.value, direction=item.direction))

    narrative = "Alert flagged for analyst review based on top model signals: " + "; ".join(parts) + "."
    return LLMExplanationOutput(narrative=narrative, claims=claims)


class LLMProvider(ABC):
    @abstractmethod
    def generate_explanation(
        self,
        txn_fields: dict[str, object],
        evidence: list[EvidenceItem],
        retry_reasons: list[str] | None = None,
    ) -> LLMExplanationOutput:
        raise NotImplementedError


class FakeProvider(LLMProvider):
    """Deterministic offline provider grounded strictly in provided top-5 evidence."""

    def generate_explanation(
        self,
        txn_fields: dict[str, object],
        evidence: list[EvidenceItem],
        retry_reasons: list[str] | None = None,
    ) -> LLMExplanationOutput:
        raising = [e for e in evidence if e.direction == "raises_risk"][:3]
        selected = raising if raising else evidence[:2]
        clauses = [f"{e.label} is {e.value} ({e.direction})" for e in selected]
        narrative = (
            f"Transaction on channel {txn_fields.get('channel', 'UNKNOWN')} for amount {txn_fields.get('amount')} "
            f"requires manual review because " + ", ".join(clauses) + "."
        )
        claims = [Claim(feature=e.feature, value=e.value, direction=e.direction) for e in selected]
        return LLMExplanationOutput(narrative=narrative, claims=claims)


class OllamaProvider(LLMProvider):
    """Optional local Ollama provider selected via LLM_PROVIDER=ollama."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate_explanation(
        self,
        txn_fields: dict[str, object],
        evidence: list[EvidenceItem],
        retry_reasons: list[str] | None = None,
    ) -> LLMExplanationOutput:
        prompt = {
            "instruction": (
                "Return strict JSON with keys 'narrative' (<=80 words, neutral tone, never say confirmed fraud or blame customer, "
                "use ONLY exact numbers from transaction or evidence) and 'claims' (list of {feature, value, direction} from evidence)."
            ),
            "transaction": txn_fields,
            "evidence": [e.model_dump() for e in evidence],
            "previous_rejection_reasons": retry_reasons or [],
        }
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": json.dumps(prompt), "format": "json", "stream": False},
            timeout=20.0,
        )
        resp.raise_for_status()
        raw_text = resp.json().get("response", "{}")
        return LLMExplanationOutput.model_validate_json(raw_text)


def get_llm_provider() -> LLMProvider:
    provider_name = os.getenv("LLM_PROVIDER", "fake").strip().lower()
    if provider_name == "ollama":
        return OllamaProvider(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        )
    return FakeProvider()


def generate_faithful_explanation(
    txn_id: str,
    txn_fields: dict[str, object],
    evidence: list[EvidenceItem],
    provider: LLMProvider | None = None,
) -> tuple[AlertExplanationResponse, int]:
    """
    Generates an explanation via LLMProvider, checks faithfulness, retries once on failure,
    and falls back to a deterministic template if still unfaithful.
    Returns (AlertExplanationResponse, attempts_used).
    """
    llm = provider or get_llm_provider()

    # Attempt 1
    try:
        candidate_1 = llm.generate_explanation(txn_fields=txn_fields, evidence=evidence, retry_reasons=None)
        check_1 = check_faithfulness(candidate_1, evidence, txn_fields=txn_fields)
    except Exception as exc:
        candidate_1 = None
        check_1 = FaithfulnessResult(passed=False, reasons=[f"Provider error: {exc}"])

    if candidate_1 is not None and check_1.passed:
        return (
            AlertExplanationResponse(
                txn_id=txn_id,
                narrative=candidate_1.narrative,
                claims=candidate_1.claims,
                evidence=evidence,
                explanation_source="llm",
                faithfulness=check_1,
            ),
            1,
        )

    # Attempt 2 (retry once with rejection reasons)
    try:
        candidate_2 = llm.generate_explanation(
            txn_fields=txn_fields, evidence=evidence, retry_reasons=check_1.reasons
        )
        check_2 = check_faithfulness(candidate_2, evidence, txn_fields=txn_fields)
    except Exception as exc:
        candidate_2 = None
        check_2 = FaithfulnessResult(passed=False, reasons=[f"Provider retry error: {exc}"])

    if candidate_2 is not None and check_2.passed:
        return (
            AlertExplanationResponse(
                txn_id=txn_id,
                narrative=candidate_2.narrative,
                claims=candidate_2.claims,
                evidence=evidence,
                explanation_source="llm",
                faithfulness=check_2,
            ),
            2,
        )

    # Mandatory deterministic fallback
    fallback = build_template_explanation(evidence)
    fallback_check = check_faithfulness(fallback, evidence, txn_fields=txn_fields)
    return (
        AlertExplanationResponse(
            txn_id=txn_id,
            narrative=fallback.narrative,
            claims=fallback.claims,
            evidence=evidence,
            explanation_source="template",
            faithfulness=fallback_check,
        ),
        2,
    )


def run_explain_eval(
    data_path: str | Path = "data/transactions.parquet",
    artifacts_dir: str | Path = "artifacts",
    unfaithful_path: str | Path = "data/eval/unfaithful_explanations.jsonl",
    reports_dir: str | Path = "reports",
    min_alerts: int = 35,
) -> dict:
    df = pd.read_parquet(data_path)
    splits = prepare_temporal_splits(df)

    art_path = Path(artifacts_dir)
    meta = json.loads((art_path / "model_meta.json").read_text())
    feat_cols: list[str] = meta.get("feature_columns", FEATURE_COLUMNS)
    threshold = float(meta.get("alert_threshold", 0.5))

    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(art_path / "xgb_model.json")

    test_probs = xgb_clf.predict_proba(splits.X_test[feat_cols])[:, 1]
    alert_indices = np.where(test_probs >= threshold)[0]
    if len(alert_indices) < min_alerts:
        alert_indices = np.argsort(test_probs)[::-1][:min_alerts]
    else:
        alert_indices = alert_indices[:max(min_alerts, 40)]

    provider = get_llm_provider()
    first_pass = 0
    after_retry_pass = 0
    template_fallbacks = 0

    for idx in alert_indices:
        row = splits.test_df.iloc[int(idx)]
        feats = {col: float(splits.X_test.iloc[int(idx)][col]) for col in feat_cols}
        evidence = extract_top_evidence(xgb_clf, feats, feature_columns=feat_cols, top_k=5)
        txn_fields = {
            "txn_id": str(row["txn_id"]),
            "card_id": str(row["card_id"]),
            "amount": float(row["amount"]),
            "mcc": int(row["mcc"]),
            "channel": str(row["channel"]),
            "country": str(row["country"]),
        }
        resp, attempts = generate_faithful_explanation(
            txn_id=str(row["txn_id"]),
            txn_fields=txn_fields,
            evidence=evidence,
            provider=provider,
        )
        if resp.explanation_source == "llm" and attempts == 1:
            first_pass += 1
            after_retry_pass += 1
        elif resp.explanation_source == "llm" and attempts == 2:
            after_retry_pass += 1
        else:
            template_fallbacks += 1

    n_eval = len(alert_indices)

    # Evaluate detector on hand-crafted unfaithful cases
    unfaithful_lines = [
        json.loads(line)
        for line in Path(unfaithful_path).read_text().splitlines()
        if line.strip()
    ]
    detected = 0
    unfaithful_details: list[dict] = []
    for case in unfaithful_lines:
        ev_items = [EvidenceItem.model_validate(e) for e in case["evidence"]]
        cand = LLMExplanationOutput.model_validate(case["explanation"])
        res = check_faithfulness(cand, ev_items, txn_fields=case.get("txn"))
        if not res.passed:
            detected += 1
        unfaithful_details.append(
            {
                "id": case["id"],
                "rule": case["rule"],
                "rejected": not res.passed,
                "reasons": res.reasons,
            }
        )

    n_unfaithful = len(unfaithful_lines)
    report = {
        "provider": os.getenv("LLM_PROVIDER", "fake"),
        "alerts_evaluated": n_eval,
        "first_attempt_pass_rate": round(first_pass / max(n_eval, 1), 4),
        "pass_rate_after_retry": round(after_retry_pass / max(n_eval, 1), 4),
        "template_fallback_rate": round(template_fallbacks / max(n_eval, 1), 4),
        "unfaithful_cases_total": n_unfaithful,
        "unfaithful_cases_detected": detected,
        "unfaithful_detection_rate": round(detected / max(n_unfaithful, 1), 4),
        "unfaithful_audit": unfaithful_details,
    }

    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)
    (rep_path / "explain_eval.json").write_text(json.dumps(report, indent=2))
    return report

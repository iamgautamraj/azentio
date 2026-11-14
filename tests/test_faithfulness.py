from __future__ import annotations

from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient

from needle.api import create_app
from needle.explain import (
    FakeProvider,
    LLMProvider,
    check_faithfulness,
    generate_faithful_explanation,
)
from needle.generator import generate_transactions, save_transactions
from needle.model import train_models
from needle.schemas import Claim, EvidenceItem, LLMExplanationOutput


class AlwaysUnfaithfulProvider(LLMProvider):
    def generate_explanation(self, txn_fields, evidence, retry_reasons=None) -> LLMExplanationOutput:
        return LLMExplanationOutput(
            narrative="Confirmed fraud with fabricated metric 99999.",
            claims=[Claim(feature="non_existent_feature", value=99999, direction="raises_risk")],
        )


def test_rejection_rules_individually() -> None:
    ev = [
        EvidenceItem(
            feature="is_new_device",
            label="First-time device for this card",
            value=1,
            contribution=2.1,
            direction="raises_risk",
        ),
        EvidenceItem(
            feature="amount",
            label="Transaction amount (INR)",
            value=50000.0,
            contribution=1.8,
            direction="raises_risk",
        ),
    ]
    txn = {"amount": 50000.0, "mcc": 5732, "channel": "ECOM"}

    # Rule 1: Feature not in evidence
    r1 = check_faithfulness(
        LLMExplanationOutput(
            narrative="Flagged due to amount 50000.0.",
            claims=[Claim(feature="unknown_col", value=50000.0, direction="raises_risk")],
        ),
        ev,
        txn,
    )
    assert not r1.passed and any("not in top-5" in m for m in r1.reasons)

    # Rule 2: Direction disagrees with contribution sign
    r2 = check_faithfulness(
        LLMExplanationOutput(
            narrative="New device 1 lowered risk.",
            claims=[Claim(feature="is_new_device", value=1, direction="lowers_risk")],
        ),
        ev,
        txn,
    )
    assert not r2.passed and any("Direction mismatch" in m for m in r2.reasons)

    # Rule 3: Number in narrative or claim not grounded
    r3 = check_faithfulness(
        LLMExplanationOutput(
            narrative="Amount of 77777.0 triggered review.",
            claims=[Claim(feature="amount", value=50000.0, direction="raises_risk")],
        ),
        ev,
        txn,
    )
    assert not r3.passed and any("Unverified number" in m for m in r3.reasons)

    # Rule 4: Confirmed fraud or customer blame
    r4 = check_faithfulness(
        LLMExplanationOutput(
            narrative="This is confirmed fraud on a new device 1.",
            claims=[Claim(feature="is_new_device", value=1, direction="raises_risk")],
        ),
        ev,
        txn,
    )
    assert not r4.passed and any("disallowed" in m for m in r4.reasons)


def test_faithful_fake_provider_and_template_fallback() -> None:
    ev = [
        EvidenceItem(
            feature="is_new_device",
            label="First-time device for this card",
            value=1,
            contribution=2.1,
            direction="raises_risk",
        ),
        EvidenceItem(
            feature="amount",
            label="Transaction amount (INR)",
            value=50000.0,
            contribution=1.8,
            direction="raises_risk",
        ),
    ]
    txn = {"amount": 50000.0, "mcc": 5732, "channel": "ECOM"}

    resp_llm, attempts_llm = generate_faithful_explanation("TXN_OK", txn, ev, provider=FakeProvider())
    assert resp_llm.faithfulness.passed is True
    assert resp_llm.explanation_source == "llm"
    assert attempts_llm == 1

    resp_fb, attempts_fb = generate_faithful_explanation(
        "TXN_FB", txn, ev, provider=AlwaysUnfaithfulProvider()
    )
    assert resp_fb.faithfulness.passed is True
    assert resp_fb.explanation_source == "template"
    assert attempts_fb == 2


def test_end_to_end_alert_explanation_endpoint(tmp_path) -> None:
    small_df = generate_transactions(seed=13, n_cards=150, n_days=30, target_txns=3000)
    parquet_path = save_transactions(small_df, tmp_path / "txns.parquet")
    art_dir = tmp_path / "artifacts"
    train_models(data_path=parquet_path, artifacts_dir=art_dir, seed=13)

    t0 = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)
    burst = [
        {
            "txn_id": f"EX_PROBE_{i}",
            "card_id": "CARD_EXPLAIN",
            "ts": (t0 + timedelta(seconds=i * 60)).isoformat(),
            "amount": 50.0,
            "merchant_id": "M_5969_202",
            "mcc": 5969,
            "channel": "ECOM",
            "country": "IN",
            "device_id": "DEV_FRD_888",
        }
        for i in range(3)
    ] + [
        {
            "txn_id": "EX_CASHOUT",
            "card_id": "CARD_EXPLAIN",
            "ts": (t0 + timedelta(minutes=7)).isoformat(),
            "amount": 70000.0,
            "merchant_id": "M_5732_202",
            "mcc": 5732,
            "channel": "ECOM",
            "country": "IN",
            "device_id": "DEV_FRD_888",
        }
    ]

    app = create_app(data_path=parquet_path, artifacts_dir=art_dir, seed_history=False)
    with TestClient(app) as client:
        for payload in burst:
            r = client.post("/score", json=payload)
            assert r.status_code == 200

        exp_resp = client.get("/alerts/EX_CASHOUT/explanation")
        assert exp_resp.status_code == 200
        body = exp_resp.json()
        assert body["txn_id"] == "EX_CASHOUT"
        assert body["explanation_source"] == "llm"
        assert body["faithfulness"]["passed"] is True
        assert len(body["evidence"]) == 5
        assert len(body["narrative"].split()) <= 80

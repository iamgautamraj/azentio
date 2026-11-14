from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score

from needle.features import FEATURE_COLUMNS
from needle.model import prepare_temporal_splits


@dataclass
class CostConfig:
    chargeback_loss_rate: float
    analyst_review_cost_inr: float
    customer_friction_cost_inr: float
    max_alerts_per_day: float

    @classmethod
    def from_yaml(cls, path: str | Path = "config/costs.yaml") -> CostConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            chargeback_loss_rate=float(raw["chargeback_loss_rate"]),
            analyst_review_cost_inr=float(raw["analyst_review_cost_inr"]),
            customer_friction_cost_inr=float(raw["customer_friction_cost_inr"]),
            max_alerts_per_day=float(raw["max_alerts_per_day"]),
        )


def compute_expected_cost(
    y_true: np.ndarray,
    y_pred_alert: np.ndarray,
    amounts: np.ndarray,
    costs: CostConfig,
) -> float:
    """Exact rupee cost: missed fraud chargeback + analyst review on alerts + customer friction on false alerts."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred_alert, dtype=int)
    amt = np.asarray(amounts, dtype=float)

    fn_mask = (y_t == 1) & (y_p == 0)
    tp_count = int(np.sum((y_t == 1) & (y_p == 1)))
    fp_count = int(np.sum((y_t == 0) & (y_p == 1)))

    missed_fraud_cost = float(np.sum(amt[fn_mask] * costs.chargeback_loss_rate))
    review_cost = float((tp_count + fp_count) * costs.analyst_review_cost_inr)
    friction_cost = float(fp_count * costs.customer_friction_cost_inr)
    return round(missed_fraud_cost + review_cost + friction_cost, 2)


def select_validation_threshold(
    val_y: np.ndarray,
    val_probs: np.ndarray,
    val_amounts: np.ndarray,
    val_days: float,
    costs: CostConfig,
) -> tuple[float, float]:
    """Sweeps thresholds strictly on validation data subject to daily alert capacity."""
    days = max(float(val_days), 1.0)
    best_thresh = 0.5
    best_cost = float("inf")
    found_feasible = False

    # Sweep candidate thresholds (including 1.01 for flag-nothing if needed)
    candidates = np.round(np.arange(0.01, 1.005, 0.005), 4)
    for th in candidates:
        preds = (val_probs >= th).astype(int)
        alerts_per_day = float(preds.sum()) / days
        if alerts_per_day > costs.max_alerts_per_day:
            continue
        cost = compute_expected_cost(val_y, preds, val_amounts, costs)
        if not found_feasible or cost < best_cost:
            found_feasible = True
            best_cost = cost
            best_thresh = float(th)

    if not found_feasible:
        # Fallback to highest threshold (fewest alerts) if even 0.995 exceeds capacity
        best_thresh = 0.995
        preds = (val_probs >= best_thresh).astype(int)
        best_cost = compute_expected_cost(val_y, preds, val_amounts, costs)

    return round(best_thresh, 4), round(best_cost, 2)


def _window_days(df: pd.DataFrame) -> float:
    ts = pd.to_datetime(df["ts"], utc=True)
    span_sec = (ts.max() - ts.min()).total_seconds()
    return max(span_sec / 86400.0, 1.0)


def run_threshold_selection(
    data_path: str | Path = "data/transactions.parquet",
    artifacts_dir: str | Path = "artifacts",
    costs_path: str | Path = "config/costs.yaml",
) -> dict:
    df = pd.read_parquet(data_path)
    splits = prepare_temporal_splits(df)
    costs = CostConfig.from_yaml(costs_path)

    art_path = Path(artifacts_dir)
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(art_path / "xgb_model.json")

    val_probs = xgb_clf.predict_proba(splits.X_val[FEATURE_COLUMNS])[:, 1]
    val_amounts = splits.val_df["amount"].to_numpy(dtype=float)
    val_days = _window_days(splits.val_df)

    chosen_th, val_cost = select_validation_threshold(
        val_y=splits.y_val,
        val_probs=val_probs,
        val_amounts=val_amounts,
        val_days=val_days,
        costs=costs,
    )

    meta_path = art_path / "model_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["alert_threshold"] = chosen_th
    meta["validation_expected_cost_inr"] = val_cost
    meta["cost_config"] = {
        "chargeback_loss_rate": costs.chargeback_loss_rate,
        "analyst_review_cost_inr": costs.analyst_review_cost_inr,
        "customer_friction_cost_inr": costs.customer_friction_cost_inr,
        "max_alerts_per_day": costs.max_alerts_per_day,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def _eval_policy(
    name: str,
    y_true: np.ndarray,
    preds: np.ndarray,
    amounts: np.ndarray,
    test_days: float,
    costs: CostConfig,
) -> dict:
    tp = int(np.sum((y_true == 1) & (preds == 1)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))
    total_alerts = tp + fp
    total_fraud_amt = float(np.sum(amounts[y_true == 1]))
    caught_fraud_amt = float(np.sum(amounts[(y_true == 1) & (preds == 1)]))

    precision = tp / total_alerts if total_alerts > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    amt_recall = caught_fraud_amt / total_fraud_amt if total_fraud_amt > 0 else 0.0
    exp_cost = compute_expected_cost(y_true, preds, amounts, costs)

    return {
        "policy": name,
        "expected_cost_inr": round(exp_cost, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "amount_weighted_recall": round(amt_recall, 4),
        "avg_alerts_per_day": round(total_alerts / test_days, 2),
        "total_alerts": total_alerts,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def generate_threshold_report(
    data_path: str | Path = "data/transactions.parquet",
    artifacts_dir: str | Path = "artifacts",
    costs_path: str | Path = "config/costs.yaml",
    reports_dir: str | Path = "reports",
) -> dict:
    df = pd.read_parquet(data_path)
    splits = prepare_temporal_splits(df)
    costs = CostConfig.from_yaml(costs_path)

    art_path = Path(artifacts_dir)
    meta = json.loads((art_path / "model_meta.json").read_text())
    chosen_th = float(meta["alert_threshold"])

    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(art_path / "xgb_model.json")

    test_probs = xgb_clf.predict_proba(splits.X_test[FEATURE_COLUMNS])[:, 1]
    test_amounts = splits.test_df["amount"].to_numpy(dtype=float)
    test_days = _window_days(splits.test_df)

    policies = [
        _eval_policy(f"chosen ({chosen_th})", splits.y_test, (test_probs >= chosen_th).astype(int), test_amounts, test_days, costs),
        _eval_policy("default (0.5)", splits.y_test, (test_probs >= 0.5).astype(int), test_amounts, test_days, costs),
        _eval_policy("flag_nothing (1.01)", splits.y_test, np.zeros_like(splits.y_test), test_amounts, test_days, costs),
        _eval_policy("flag_everything (0.0)", splits.y_test, np.ones_like(splits.y_test), test_amounts, test_days, costs),
    ]

    # Per-pattern recall at chosen threshold
    chosen_preds = (test_probs >= chosen_th).astype(int)
    pattern_breakdown: list[dict] = []
    fraud_mask = splits.test_df["is_fraud"] == 1
    for pat, grp in splits.test_df[fraud_mask].groupby("fraud_pattern"):
        idx = grp.index.to_numpy()
        caught = int(chosen_preds[idx].sum())
        total = int(len(idx))
        pattern_breakdown.append(
            {
                "fraud_pattern": str(pat),
                "total_cases": total,
                "caught": caught,
                "recall": round(caught / total, 4) if total > 0 else 0.0,
            }
        )
    pattern_breakdown.sort(key=lambda x: x["recall"])

    # Per-channel PR-AUC
    channel_breakdown: list[dict] = []
    for ch, grp in splits.test_df.groupby("channel"):
        idx = grp.index.to_numpy()
        y_ch = splits.y_test[idx]
        p_ch = test_probs[idx]
        pr_auc = float(average_precision_score(y_ch, p_ch)) if y_ch.sum() > 0 else 0.0
        channel_breakdown.append(
            {
                "channel": str(ch),
                "transactions": int(len(idx)),
                "fraud_cases": int(y_ch.sum()),
                "pr_auc": round(pr_auc, 4),
            }
        )
    channel_breakdown.sort(key=lambda x: x["pr_auc"])

    weakest_areas = [
        {
            "area": f"Pattern: {pattern_breakdown[0]['fraud_pattern']} (Recall={pattern_breakdown[0]['recall']})",
            "hypothesis": "Initial small probe transactions within a burst have low amounts before velocity counters build up.",
        },
        {
            "area": f"Pattern: {pattern_breakdown[1]['fraud_pattern']} (Recall={pattern_breakdown[1]['recall']})",
            "hypothesis": "Overlaps with genuine high-value shoppers or travelers, requiring a higher threshold to respect daily review capacity.",
        },
        {
            "area": f"Channel: {channel_breakdown[0]['channel']} (PR-AUC={channel_breakdown[0]['pr_auc']})",
            "hypothesis": "Fewer behavioral signals (e.g. null device_id on POS/ATM) reduce separability compared to ECOM.",
        },
    ]

    report_data = {
        "chosen_threshold": chosen_th,
        "validation_expected_cost_inr": meta.get("validation_expected_cost_inr"),
        "max_alerts_per_day": costs.max_alerts_per_day,
        "test_window_days": round(test_days, 2),
        "policy_comparison": policies,
        "per_pattern_recall": pattern_breakdown,
        "per_channel_pr_auc": channel_breakdown,
        "weakest_areas": weakest_areas,
    }

    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)
    (rep_path / "threshold.json").write_text(json.dumps(report_data, indent=2))

    # Build Markdown report
    lines = [
        "# Cost- & Capacity-Aware Threshold Report (Test Window)",
        "",
        f"- **Selected Threshold (Validation):** `{chosen_th}`",
        f"- **Daily Analyst Capacity Limit:** `{costs.max_alerts_per_day}` alerts/day",
        f"- **Cost Assumptions:** Chargeback = `{costs.chargeback_loss_rate * 100:.0f}%` of txn amount | Review = `INR {costs.analyst_review_cost_inr}` | False-Alert Friction = `INR {costs.customer_friction_cost_inr}`",
        "",
        "## Policy Comparison (Test Window)",
        "",
        "| Policy | Expected Cost (INR) | Precision | Recall | Amount-Weighted Recall | Avg Alerts / Day |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in policies:
        lines.append(
            f"| {p['policy']} | {p['expected_cost_inr']:,.2f} | {p['precision']:.4f} | {p['recall']:.4f} | {p['amount_weighted_recall']:.4f} | {p['avg_alerts_per_day']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Recall by Fraud Pattern (Chosen Threshold)",
            "",
            "| Fraud Pattern | Test Cases | Caught | Recall |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for row in pattern_breakdown:
        lines.append(f"| {row['fraud_pattern']} | {row['total_cases']} | {row['caught']} | {row['recall']:.4f} |")

    lines.extend(
        [
            "",
            "## PR-AUC by Channel (Test Window)",
            "",
            "| Channel | Transactions | Fraud Cases | PR-AUC |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for row in channel_breakdown:
        lines.append(f"| {row['channel']} | {row['transactions']} | {row['fraud_cases']} | {row['pr_auc']:.4f} |")

    lines.extend(["", "## Top 3 Weakest Areas & Hypotheses", ""])
    for idx, w in enumerate(weakest_areas, 1):
        lines.append(f"{idx}. **{w['area']}** — {w['hypothesis']}")

    (rep_path / "threshold.md").write_text("\n".join(lines) + "\n")
    return report_data

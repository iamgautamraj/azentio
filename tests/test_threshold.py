from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from needle.api import create_app
from needle.generator import generate_transactions, save_transactions
from needle.model import train_models
from needle.threshold import (
    CostConfig,
    compute_expected_cost,
    run_threshold_selection,
    select_validation_threshold,
)

COSTS_PATH = Path(__file__).resolve().parents[1] / "config" / "costs.yaml"


def test_exact_rupee_cost_math() -> None:
    cfg = CostConfig(
        chargeback_loss_rate=1.0,
        analyst_review_cost_inr=150.0,
        customer_friction_cost_inr=300.0,
        max_alerts_per_day=10.0,
    )
    # 4 txns:
    # idx 0: TP (fraud=1, alert=1, amt=10000) -> review cost = 150
    # idx 1: FN (fraud=1, alert=0, amt=20000) -> chargeback = 20000
    # idx 2: FP (fraud=0, alert=1, amt=500)   -> review + friction = 150 + 300 = 450
    # idx 3: TN (fraud=0, alert=0, amt=800)   -> 0
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    amounts = np.array([10000.0, 20000.0, 500.0, 800.0])

    total_cost = compute_expected_cost(y_true, y_pred, amounts, cfg)
    assert total_cost == 20600.0


def test_capacity_respected_and_test_label_invariance() -> None:
    cfg = CostConfig(
        chargeback_loss_rate=1.0,
        analyst_review_cost_inr=100.0,
        customer_friction_cost_inr=200.0,
        max_alerts_per_day=2.0,  # Over 2 days -> max 4 alerts total
    )
    val_probs = np.array([0.95, 0.90, 0.85, 0.80, 0.60, 0.55, 0.10, 0.05])
    val_y = np.array([1, 1, 1, 0, 1, 1, 0, 0])
    val_amt = np.array([50000.0, 40000.0, 30000.0, 1000.0, 25000.0, 20000.0, 500.0, 500.0])

    th1, cost1 = select_validation_threshold(val_y, val_probs, val_amt, val_days=2.0, costs=cfg)
    alerts_count = int(np.sum(val_probs >= th1))
    assert alerts_count / 2.0 <= cfg.max_alerts_per_day

    # Test-label invariance: select_validation_threshold takes only validation arrays
    # Re-running with identical validation inputs yields exact same threshold
    th2, cost2 = select_validation_threshold(val_y, val_probs, val_amt, val_days=2.0, costs=cfg)
    assert th1 == th2 and cost1 == cost2


def test_high_risk_burst_triggers_review_and_stores_alert(tmp_path) -> None:
    small_df = generate_transactions(seed=11, n_cards=150, n_days=30, target_txns=3000)
    parquet_path = save_transactions(small_df, tmp_path / "txns.parquet")
    art_dir = tmp_path / "artifacts"
    train_models(data_path=parquet_path, artifacts_dir=art_dir, seed=11)

    t0 = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
    burst = [
        {
            "txn_id": f"PROBE_{i}",
            "card_id": "CARD_BURST_TEST",
            "ts": (t0 + timedelta(seconds=i * 60)).isoformat(),
            "amount": 50.0,
            "merchant_id": "M_5969_101",
            "mcc": 5969,
            "channel": "ECOM",
            "country": "IN",
            "device_id": "DEV_FRD_999",
        }
        for i in range(3)
    ] + [
        {
            "txn_id": "CASHOUT_1",
            "card_id": "CARD_BURST_TEST",
            "ts": (t0 + timedelta(minutes=8)).isoformat(),
            "amount": 65000.0,
            "merchant_id": "M_5732_101",
            "mcc": 5732,
            "channel": "ECOM",
            "country": "IN",
            "device_id": "DEV_FRD_999",
        }
    ]

    app = create_app(data_path=parquet_path, artifacts_dir=art_dir, seed_history=False)
    with TestClient(app) as client:
        last_resp = None
        for payload in burst:
            r = client.post("/score", json=payload)
            assert r.status_code == 200
            last_resp = r.json()

        assert last_resp is not None
        assert last_resp["decision"] == "REVIEW"
        assert app.state.store.get_alert("CASHOUT_1") is not None

def test_chosen_threshold_unchanged_when_test_window_labels_flipped(tmp_path) -> None:
    df = generate_transactions(seed=21, n_cards=150, n_days=30, target_txns=3000)
    path_a = save_transactions(df, tmp_path / "a.parquet")
    art_dir = tmp_path / "artifacts"
    meta = train_models(data_path=path_a, artifacts_dir=art_dir, seed=21)

    th_before = run_threshold_selection(path_a, art_dir, COSTS_PATH)["alert_threshold"]

    test_min = pd.Timestamp(meta["splits"]["test_min_ts"])
    flipped = df.copy()
    in_test = pd.to_datetime(flipped["ts"], utc=True) >= test_min
    assert in_test.any()
    flipped.loc[in_test, "is_fraud"] = 1 - flipped.loc[in_test, "is_fraud"]
    path_b = save_transactions(flipped, tmp_path / "b.parquet")

    th_after = run_threshold_selection(path_b, art_dir, COSTS_PATH)["alert_threshold"]
    assert th_before == th_after

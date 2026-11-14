from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pandas as pd
from fastapi.testclient import TestClient

from needle.api import create_app
from needle.features import FEATURE_COLUMNS, build_features_batch
from needle.generator import generate_transactions, save_transactions
from needle.model import split_by_time, train_models


def test_no_future_leakage() -> None:
    t0 = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    base_rows = [
        {
            "txn_id": "T1",
            "card_id": "CARD_A",
            "ts": t0,
            "amount": 1000.0,
            "merchant_id": "M1",
            "mcc": 5411,
            "channel": "POS",
            "country": "IN",
            "device_id": None,
        },
        {
            "txn_id": "T2",
            "card_id": "CARD_A",
            "ts": t0 + timedelta(minutes=15),
            "amount": 2500.0,
            "merchant_id": "M2",
            "mcc": 5732,
            "channel": "ECOM",
            "country": "IN",
            "device_id": "DEV_1",
        },
    ]
    future_row = {
        "txn_id": "T3",
        "card_id": "CARD_A",
        "ts": t0 + timedelta(minutes=30),
        "amount": 90000.0,
        "merchant_id": "M3",
        "mcc": 5732,
        "channel": "ECOM",
        "country": "US",
        "device_id": "DEV_999",
    }

    X_before, _ = build_features_batch(pd.DataFrame(base_rows))
    X_after, _ = build_features_batch(pd.DataFrame(base_rows + [future_row]))

    pd.testing.assert_frame_equal(X_before, X_after.iloc[:2])


def test_temporal_split_strict_ordering() -> None:
    df = generate_transactions(seed=42, n_cards=150, n_days=30, target_txns=3000)
    train_df, val_df, test_df = split_by_time(df)

    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
    max_train = pd.to_datetime(train_df["ts"], utc=True).max()
    min_val = pd.to_datetime(val_df["ts"], utc=True).min()
    max_val = pd.to_datetime(val_df["ts"], utc=True).max()
    min_test = pd.to_datetime(test_df["ts"], utc=True).min()

    assert max_train < min_val <= max_val < min_test


def test_online_offline_parity_and_score_api(tmp_path) -> None:
    small_df = generate_transactions(seed=7, n_cards=120, n_days=30, target_txns=2500)
    parquet_path = save_transactions(small_df, tmp_path / "txns.parquet")
    art_dir = tmp_path / "artifacts"
    train_models(data_path=parquet_path, artifacts_dir=art_dir, seed=7)

    t0 = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    seq = [
        {
            "txn_id": "SEQ_1",
            "card_id": "CARD_PARITY",
            "ts": t0.isoformat(),
            "amount": 500.0,
            "merchant_id": "M10",
            "mcc": 5411,
            "channel": "POS",
            "country": "IN",
            "device_id": None,
        },
        {
            "txn_id": "SEQ_2",
            "card_id": "CARD_PARITY",
            "ts": (t0 + timedelta(minutes=5)).isoformat(),
            "amount": 45000.0,
            "merchant_id": "M20",
            "mcc": 5732,
            "channel": "ECOM",
            "country": "US",
            "device_id": "DEV_NEW_77",
        },
    ]

    offline_df = pd.DataFrame(seq)
    offline_df["ts"] = pd.to_datetime(offline_df["ts"], utc=True)
    X_offline, _ = build_features_batch(offline_df)

    app = create_app(data_path=parquet_path, artifacts_dir=art_dir, seed_history=False)
    with TestClient(app) as client:
        for idx, payload in enumerate(seq):
            resp = client.post("/score", json=payload)
            assert resp.status_code == 200
            body = resp.json()
            assert body["txn_id"] == payload["txn_id"]
            assert 0.0 <= body["fraud_score"] <= 1.0
            assert body["decision"] in ("ALLOW", "REVIEW")
            for col in FEATURE_COLUMNS:
                assert abs(body["features"][col] - float(X_offline.iloc[idx][col])) < 1e-6

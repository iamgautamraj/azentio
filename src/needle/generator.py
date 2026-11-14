from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd


MCCS_POS = [5411, 5812, 5541, 5912, 5311, 5999]
MCCS_ECOM = [5732, 5942, 5691, 4816, 7372, 5969]
MCCS_ATM = [6011]
FOREIGN_COUNTRIES = ["US", "GB", "AE", "SG", "TH", "DE"]


def generate_transactions(
    seed: int = 42,
    n_cards: int = 3500,
    n_days: int = 100,
    target_txns: int = 160_000,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    total_seconds = n_days * 86400

    card_ids = [f"CARD_{i:05d}" for i in range(n_cards)]
    card_home_device = {c: f"DEV_{i:05d}" for i, c in enumerate(card_ids)}
    card_typical_amt = rng.lognormal(mean=7.2, sigma=0.75, size=n_cards)  # ~INR 1,300 median
    traveler_cards = set(rng.choice(card_ids, size=int(n_cards * 0.03), replace=False))

    rows: list[dict] = []

    # 1. Genuine baseline transactions (~99.35% of volume)
    n_genuine = int(target_txns * 0.994)
    gen_card_idx = rng.integers(0, n_cards, size=n_genuine)
    gen_offsets = rng.integers(0, total_seconds, size=n_genuine)
    gen_channels = rng.choice(["POS", "ECOM", "ATM"], size=n_genuine, p=[0.48, 0.40, 0.12])

    for i in range(n_genuine):
        c_idx = int(gen_card_idx[i])
        cid = card_ids[c_idx]
        ts = start_dt + timedelta(seconds=int(gen_offsets[i]))
        ch = str(gen_channels[i])

        # Genuine look-alikes: occasional high-value purchase (2.5%)
        is_genuine_spike = rng.random() < 0.025
        base_amt = float(card_typical_amt[c_idx])
        if ch == "ATM":
            amt = float(rng.choice([500, 1000, 2000, 5000, 10000]))
            mcc = 6011
            dev = None
        elif ch == "POS":
            amt = round(base_amt * rng.lognormal(0.0, 0.55) * (6.0 if is_genuine_spike else 1.0), 2)
            mcc = int(rng.choice(MCCS_POS))
            dev = None
        else:
            amt = round(base_amt * rng.lognormal(0.1, 0.65) * (7.0 if is_genuine_spike else 1.0), 2)
            mcc = int(rng.choice(MCCS_ECOM))
            # Genuine new device (3% of ECOM txns)
            dev = f"DEV_NEW_{rng.integers(10000, 99999)}" if rng.random() < 0.03 else card_home_device[cid]

        # Genuine traveler look-alike
        if cid in traveler_cards and rng.random() < 0.25:
            country = str(rng.choice(FOREIGN_COUNTRIES))
        else:
            country = "IN"

        rows.append(
            {
                "card_id": cid,
                "ts": ts,
                "amount": max(50.0, amt),
                "merchant_id": f"M_{mcc}_{rng.integers(100, 999)}",
                "mcc": mcc,
                "channel": ch,
                "country": country,
                "device_id": dev,
                "is_fraud": 0,
                "fraud_pattern": None,
            }
        )

    # 2. Inject 4 distinct fraud patterns (~0.62% overall fraud rate across train/val/test)
    # Pattern A: card_testing_burst (3 small ECOM probes + 1 large ECOM cashout)
    for _ in range(95):
        c_idx = int(rng.integers(0, n_cards))
        cid = card_ids[c_idx]
        t0 = start_dt + timedelta(seconds=int(rng.integers(3600, total_seconds - 3600)))
        dev = f"DEV_FRD_{rng.integers(1000, 9999)}"
        for step in range(3):
            rows.append(
                {
                    "card_id": cid,
                    "ts": t0 + timedelta(seconds=step * int(rng.integers(45, 180))),
                    "amount": round(float(rng.uniform(25.0, 140.0)), 2),
                    "merchant_id": f"M_5969_{rng.integers(100, 999)}",
                    "mcc": 5969,
                    "channel": "ECOM",
                    "country": "IN",
                    "device_id": dev,
                    "is_fraud": 1,
                    "fraud_pattern": "card_testing_burst",
                }
            )
        rows.append(
            {
                "card_id": cid,
                "ts": t0 + timedelta(seconds=int(rng.integers(600, 1200))),
                "amount": round(float(rng.uniform(28000.0, 75000.0)), 2),
                "merchant_id": f"M_5732_{rng.integers(100, 999)}",
                "mcc": 5732,
                "channel": "ECOM",
                "country": "IN",
                "device_id": dev,
                "is_fraud": 1,
                "fraud_pattern": "card_testing_burst",
            }
        )

    # Pattern B: foreign_usage (sudden international high-risk spend on non-traveler card)
    non_travelers = [c for c in card_ids if c not in traveler_cards]
    for _ in range(210):
        cid = str(rng.choice(non_travelers))
        t0 = start_dt + timedelta(seconds=int(rng.integers(3600, total_seconds - 3600)))
        ch = str(rng.choice(["POS", "ECOM"]))
        rows.append(
            {
                "card_id": cid,
                "ts": t0,
                "amount": round(float(rng.uniform(12000.0, 65000.0)), 2),
                "merchant_id": f"M_5311_{rng.integers(100, 999)}",
                "mcc": 5311,
                "channel": ch,
                "country": str(rng.choice(FOREIGN_COUNTRIES)),
                "device_id": f"DEV_FRD_{rng.integers(1000, 9999)}" if ch == "ECOM" else None,
                "is_fraud": 1,
                "fraud_pattern": "foreign_usage",
            }
        )

    # Pattern C: new_device_high_ecom (account takeover: new device + large electronics/digital spend)
    for _ in range(220):
        c_idx = int(rng.integers(0, n_cards))
        cid = card_ids[c_idx]
        t0 = start_dt + timedelta(seconds=int(rng.integers(3600, total_seconds - 3600)))
        rows.append(
            {
                "card_id": cid,
                "ts": t0,
                "amount": round(float(rng.uniform(22000.0, 90000.0)), 2),
                "merchant_id": f"M_5732_{rng.integers(100, 999)}",
                "mcc": 5732,
                "channel": "ECOM",
                "country": "IN",
                "device_id": f"DEV_ATO_{rng.integers(10000, 99999)}",
                "is_fraud": 1,
                "fraud_pattern": "new_device_high_ecom",
            }
        )

    # Pattern D: atm_cashout_burst (3 rapid ATM withdrawals within 30 minutes)
    for _ in range(75):
        c_idx = int(rng.integers(0, n_cards))
        cid = card_ids[c_idx]
        t0 = start_dt + timedelta(seconds=int(rng.integers(3600, total_seconds - 3600)))
        for step in range(3):
            rows.append(
                {
                    "card_id": cid,
                    "ts": t0 + timedelta(seconds=step * int(rng.integers(180, 600))),
                    "amount": float(rng.choice([10000.0, 15000.0, 20000.0])),
                    "merchant_id": f"M_6011_{rng.integers(100, 999)}",
                    "mcc": 6011,
                    "channel": "ATM",
                    "country": "IN",
                    "device_id": None,
                    "is_fraud": 1,
                    "fraud_pattern": "atm_cashout_burst",
                }
            )

    df = pd.DataFrame(rows)
    # Deterministic chronological sort + tie-breaker before assigning txn_id
    df = df.sort_values(["ts", "card_id", "amount", "mcc"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "txn_id", [f"TXN_{i:07d}" for i in range(len(df))])
    return df


def save_transactions(df: pd.DataFrame, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path

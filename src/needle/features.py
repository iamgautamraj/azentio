from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import bisect
import numpy as np
import pandas as pd


FEATURE_COLUMNS: list[str] = [
    "amount",
    "mcc",
    "hour_of_day",
    "channel_POS",
    "channel_ECOM",
    "channel_ATM",
    "txn_count_1h",
    "txn_sum_1h",
    "txn_count_24h",
    "txn_sum_24h",
    "amt_to_median_ratio",
    "secs_since_last_txn",
    "is_new_country",
    "is_new_device",
]

FEATURE_LABELS: dict[str, str] = {
    "amount": "Transaction amount (INR)",
    "mcc": "Merchant category code (MCC)",
    "hour_of_day": "Hour of day (UTC)",
    "channel_POS": "POS channel indicator",
    "channel_ECOM": "E-commerce channel indicator",
    "channel_ATM": "ATM channel indicator",
    "txn_count_1h": "Card transaction count in past hour",
    "txn_sum_1h": "Card spend sum in past hour (INR)",
    "txn_count_24h": "Card transaction count in past day",
    "txn_sum_24h": "Card spend sum in past day (INR)",
    "amt_to_median_ratio": "Amount relative to card trailing median",
    "secs_since_last_txn": "Seconds since card's previous transaction",
    "is_new_country": "First-time country for this card",
    "is_new_device": "First-time device for this card",
}


def _to_epoch_seconds(ts: datetime | pd.Timestamp | str) -> float:
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return float(ts.timestamp())
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if ts.tzinfo is None:
        return float(ts.replace(tzinfo=None).timestamp())
    return float(ts.timestamp())


@dataclass
class CardHistory:
    win_1h: deque[tuple[float, float]] = field(default_factory=deque)
    sum_1h: float = 0.0
    win_24h: deque[tuple[float, float]] = field(default_factory=deque)
    sum_24h: float = 0.0
    sorted_amounts: list[float] = field(default_factory=list)
    last_ts: float | None = None
    seen_countries: set[str] = field(default_factory=set)
    seen_devices: set[str] = field(default_factory=set)

    def compute_and_update(
        self,
        ts_sec: float,
        hour_of_day: int,
        amount: float,
        mcc: int,
        channel: str,
        country: str,
        device_id: str | None,
    ) -> dict[str, float]:
        # Evict entries outside strict past windows (t_prev < ts_sec)
        cutoff_1h = ts_sec - 3600.0
        while self.win_1h and self.win_1h[0][0] < cutoff_1h:
            _, old_amt = self.win_1h.popleft()
            self.sum_1h -= old_amt

        cutoff_24h = ts_sec - 86400.0
        while self.win_24h and self.win_24h[0][0] < cutoff_24h:
            _, old_amt = self.win_24h.popleft()
            self.sum_24h -= old_amt

        txn_count_1h = float(len(self.win_1h))
        txn_sum_1h = float(max(0.0, round(self.sum_1h, 4)))
        txn_count_24h = float(len(self.win_24h))
        txn_sum_24h = float(max(0.0, round(self.sum_24h, 4)))

        n_hist = len(self.sorted_amounts)
        if n_hist == 0:
            amt_to_median_ratio = 1.0
        else:
            mid = n_hist // 2
            if n_hist % 2 == 1:
                med = self.sorted_amounts[mid]
            else:
                med = 0.5 * (self.sorted_amounts[mid - 1] + self.sorted_amounts[mid])
            amt_to_median_ratio = float(round(amount / max(med, 1.0), 6))

        secs_since_last_txn = (
            86400.0 if self.last_ts is None else float(max(0.0, round(ts_sec - self.last_ts, 3)))
        )
        is_new_country = 1.0 if country not in self.seen_countries else 0.0
        is_new_device = 1.0 if (device_id is not None and device_id != "" and device_id not in self.seen_devices) else 0.0

        feats: dict[str, float] = {
            "amount": float(amount),
            "mcc": float(mcc),
            "hour_of_day": float(hour_of_day),
            "channel_POS": 1.0 if channel == "POS" else 0.0,
            "channel_ECOM": 1.0 if channel == "ECOM" else 0.0,
            "channel_ATM": 1.0 if channel == "ATM" else 0.0,
            "txn_count_1h": txn_count_1h,
            "txn_sum_1h": txn_sum_1h,
            "txn_count_24h": txn_count_24h,
            "txn_sum_24h": txn_sum_24h,
            "amt_to_median_ratio": amt_to_median_ratio,
            "secs_since_last_txn": secs_since_last_txn,
            "is_new_country": is_new_country,
            "is_new_device": is_new_device,
        }

        # Mutate state AFTER computing features (strictly causal)
        self.win_1h.append((ts_sec, amount))
        self.sum_1h += amount
        self.win_24h.append((ts_sec, amount))
        self.sum_24h += amount
        bisect.insort(self.sorted_amounts, amount)
        self.last_ts = ts_sec
        self.seen_countries.add(country)
        if device_id is not None and device_id != "":
            self.seen_devices.add(device_id)

        return feats


class FeatureEngine:
    """Single causal feature engine shared across offline training and online serving."""

    def __init__(self) -> None:
        self._cards: dict[str, CardHistory] = {}

    def process_one(
        self,
        card_id: str,
        ts: datetime | pd.Timestamp | str,
        amount: float,
        mcc: int,
        channel: str,
        country: str,
        device_id: str | None = None,
    ) -> dict[str, float]:
        ts_pd = pd.Timestamp(ts)
        if ts_pd.tzinfo is None:
            ts_pd = ts_pd.tz_localize("UTC")
        ts_sec = float(ts_pd.timestamp())
        hour_of_day = int(ts_pd.hour)

        hist = self._cards.get(card_id)
        if hist is None:
            hist = CardHistory()
            self._cards[card_id] = hist

        clean_dev = None if (device_id is None or (isinstance(device_id, float) and np.isnan(device_id))) else str(device_id)
        return hist.compute_and_update(
            ts_sec=ts_sec,
            hour_of_day=hour_of_day,
            amount=float(amount),
            mcc=int(mcc),
            channel=str(channel),
            country=str(country),
            device_id=clean_dev,
        )

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assumes df is sorted chronologically by ts. Updates state in place."""
        records: list[dict[str, float]] = []
        for row in df.itertuples(index=False):
            feats = self.process_one(
                card_id=row.card_id,
                ts=row.ts,
                amount=row.amount,
                mcc=row.mcc,
                channel=row.channel,
                country=row.country,
                device_id=row.device_id,
            )
            records.append(feats)
        return pd.DataFrame(records, index=df.index)[FEATURE_COLUMNS]


def build_features_batch(df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureEngine]:
    engine = FeatureEngine()
    X = engine.transform_dataframe(df)
    return X, engine

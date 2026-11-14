from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

from needle.features import FeatureEngine
from needle.schemas import TransactionIn


@dataclass
class AlertRecord:
    txn_id: str
    card_id: str
    ts: str
    amount: float
    merchant_id: str
    mcc: int
    channel: str
    country: str
    device_id: str | None
    fraud_score: float
    features: dict[str, float]


@dataclass
class ScoringStateStore:
    """In-memory per-card history and alert store seeded with pre-test transactions."""

    feature_engine: FeatureEngine = field(default_factory=FeatureEngine)
    alerts: dict[str, AlertRecord] = field(default_factory=dict)

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str | Path,
        cutoff_ts: str | pd.Timestamp | None = None,
    ) -> ScoringStateStore:
        store = cls()
        path = Path(parquet_path)
        if not path.exists():
            return store

        df = pd.read_parquet(path)
        df = df.sort_values(["ts", "txn_id"], kind="mergesort").reset_index(drop=True)
        if cutoff_ts is not None:
            cutoff_pd = pd.Timestamp(cutoff_ts)
            if cutoff_pd.tzinfo is None:
                cutoff_pd = cutoff_pd.tz_localize("UTC")
            ts_series = pd.to_datetime(df["ts"], utc=True)
            df = df[ts_series < cutoff_pd].reset_index(drop=True)

        if not df.empty:
            store.feature_engine.transform_dataframe(df)
        return store

    def score_and_record_features(self, txn: TransactionIn) -> dict[str, float]:
        return self.feature_engine.process_one(
            card_id=txn.card_id,
            ts=txn.ts,
            amount=txn.amount,
            mcc=txn.mcc,
            channel=txn.channel,
            country=txn.country,
            device_id=txn.device_id,
        )

    def save_alert(self, txn: TransactionIn, fraud_score: float, features: dict[str, float]) -> None:
        self.alerts[txn.txn_id] = AlertRecord(
            txn_id=txn.txn_id,
            card_id=txn.card_id,
            ts=txn.ts.isoformat(),
            amount=txn.amount,
            merchant_id=txn.merchant_id,
            mcc=txn.mcc,
            channel=txn.channel,
            country=txn.country,
            device_id=txn.device_id,
            fraud_score=fraud_score,
            features=features,
        )

    def get_alert(self, txn_id: str) -> AlertRecord | None:
        return self.alerts.get(txn_id)

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import json
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
import xgboost as xgb

from needle.features import FEATURE_COLUMNS
from needle.schemas import ScoreResponse, TransactionIn
from needle.store import ScoringStateStore


def create_app(
    data_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    seed_history: bool = True,
) -> FastAPI:
    data_file = Path(data_path or os.getenv("NEEDLE_DATA_PATH", "data/transactions.parquet"))
    art_dir = Path(artifacts_dir or os.getenv("NEEDLE_ARTIFACTS_DIR", "artifacts"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        meta_path = art_dir / "model_meta.json"
        model_path = art_dir / "xgb_model.json"

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = {
                "model_version": "untrained-local",
                "feature_columns": FEATURE_COLUMNS,
                "alert_threshold": 0.5,
                "splits": {},
            }

        clf = xgb.XGBClassifier()
        if model_path.exists():
            clf.load_model(model_path)
            app.state.model = clf
        else:
            app.state.model = None

        app.state.meta = meta
        cutoff_ts = meta.get("splits", {}).get("test_min_ts")
        if seed_history and data_file.exists():
            app.state.store = ScoringStateStore.from_parquet(data_file, cutoff_ts=cutoff_ts)
        else:
            app.state.store = ScoringStateStore()
        yield

    app = FastAPI(title="Project Needle Card Fraud Scorer", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/score", response_model=ScoreResponse)
    def score_transaction(txn: TransactionIn) -> ScoreResponse:
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model artifact not loaded")

        store: ScoringStateStore = app.state.store
        meta: dict = app.state.meta
        feat_cols: list[str] = meta.get("feature_columns", FEATURE_COLUMNS)

        feats = store.score_and_record_features(txn)
        row_df = pd.DataFrame([feats])[feat_cols]
        prob = float(app.state.model.predict_proba(row_df)[0, 1])

        threshold = float(meta.get("alert_threshold", 0.5))
        decision = "REVIEW" if prob >= threshold else "ALLOW"
        if decision == "REVIEW":
            store.save_alert(txn, fraud_score=prob, features=feats)

        return ScoreResponse(
            txn_id=txn.txn_id,
            fraud_score=round(prob, 6),
            decision=decision,
            model_version=meta.get("model_version", "unknown"),
            features=feats,
        )

    return app


app = create_app()

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from needle.features import FEATURE_COLUMNS, build_features_batch


@dataclass
class TemporalSplits:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict 60/20/20 temporal split ensuring max(train.ts) < min(val.ts) <= max(val.ts) < min(test.ts)."""
    df_sorted = df.sort_values(["ts", "txn_id"], kind="mergesort").reset_index(drop=True)
    ts_series = pd.to_datetime(df_sorted["ts"], utc=True)

    t_min = ts_series.min()
    t_max = ts_series.max()
    total_span = t_max - t_min

    val_start_cut = t_min + 0.60 * total_span
    test_start_cut = t_min + 0.80 * total_span

    train_df = df_sorted[ts_series < val_start_cut].copy()
    val_df = df_sorted[(ts_series >= val_start_cut) & (ts_series < test_start_cut)].copy()
    test_df = df_sorted[ts_series >= test_start_cut].copy()

    # Guarantee strict inequality at boundaries even if identical timestamps exist
    if not train_df.empty and not val_df.empty:
        max_train_ts = pd.to_datetime(train_df["ts"], utc=True).max()
        val_df = val_df[pd.to_datetime(val_df["ts"], utc=True) > max_train_ts].copy()
    if not val_df.empty and not test_df.empty:
        max_val_ts = pd.to_datetime(val_df["ts"], utc=True).max()
        test_df = test_df[pd.to_datetime(test_df["ts"], utc=True) > max_val_ts].copy()

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_temporal_splits(df: pd.DataFrame) -> TemporalSplits:
    df_sorted = df.sort_values(["ts", "txn_id"], kind="mergesort").reset_index(drop=True)
    # Build causal features across the sorted stream (each row uses strictly past card txns)
    X_all, _ = build_features_batch(df_sorted)
    df_with_idx = df_sorted.copy()
    df_with_idx["_orig_idx"] = np.arange(len(df_sorted))

    train_df, val_df, test_df = split_by_time(df_with_idx)

    X_train = X_all.iloc[train_df["_orig_idx"].values].reset_index(drop=True)
    X_val = X_all.iloc[val_df["_orig_idx"].values].reset_index(drop=True)
    X_test = X_all.iloc[test_df["_orig_idx"].values].reset_index(drop=True)

    train_df = train_df.drop(columns=["_orig_idx"])
    val_df = val_df.drop(columns=["_orig_idx"])
    test_df = test_df.drop(columns=["_orig_idx"])

    return TemporalSplits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        X_train=X_train,
        y_train=train_df["is_fraud"].to_numpy(dtype=int),
        X_val=X_val,
        y_val=val_df["is_fraud"].to_numpy(dtype=int),
        X_test=X_test,
        y_test=test_df["is_fraud"].to_numpy(dtype=int),
    )


def train_models(
    data_path: str | Path = "data/transactions.parquet",
    artifacts_dir: str | Path = "artifacts",
    seed: int = 42,
) -> dict:
    df = pd.read_parquet(data_path)
    splits = prepare_temporal_splits(df)

    n_pos = int(splits.y_train.sum())
    n_neg = int(len(splits.y_train) - n_pos)
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    xgb_clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=25,
        random_state=seed,
        n_jobs=2,
    )
    xgb_clf.fit(
        splits.X_train[FEATURE_COLUMNS],
        splits.y_train,
        eval_set=[(splits.X_val[FEATURE_COLUMNS], splits.y_val)],
        verbose=False,
    )

    baseline_lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed)),
        ]
    )
    baseline_lr.fit(splits.X_train[FEATURE_COLUMNS], splits.y_train)

    art_path = Path(artifacts_dir)
    art_path.mkdir(parents=True, exist_ok=True)
    xgb_path = art_path / "xgb_model.json"
    lr_path = art_path / "baseline_lr.joblib"
    meta_path = art_path / "model_meta.json"

    xgb_clf.save_model(xgb_path)
    joblib.dump(baseline_lr, lr_path)

    meta = {
        "model_version": f"xgb-needle-v1-seed{seed}",
        "seed": seed,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "scale_pos_weight": round(scale_pos_weight, 4),
        "best_iteration": int(getattr(xgb_clf, "best_iteration", xgb_clf.n_estimators)),
        "splits": {
            "train_min_ts": pd.to_datetime(splits.train_df["ts"], utc=True).min().isoformat(),
            "train_max_ts": pd.to_datetime(splits.train_df["ts"], utc=True).max().isoformat(),
            "val_min_ts": pd.to_datetime(splits.val_df["ts"], utc=True).min().isoformat(),
            "val_max_ts": pd.to_datetime(splits.val_df["ts"], utc=True).max().isoformat(),
            "test_min_ts": pd.to_datetime(splits.test_df["ts"], utc=True).min().isoformat(),
            "test_max_ts": pd.to_datetime(splits.test_df["ts"], utc=True).max().isoformat(),
            "train_rows": int(len(splits.train_df)),
            "val_rows": int(len(splits.val_df)),
            "test_rows": int(len(splits.test_df)),
        },
        "alert_threshold": 0.5,
        "validation_expected_cost_inr": None,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(tpr[valid[-1]])


def evaluate_models(
    data_path: str | Path = "data/transactions.parquet",
    artifacts_dir: str | Path = "artifacts",
    reports_dir: str | Path = "reports",
) -> dict:
    df = pd.read_parquet(data_path)
    splits = prepare_temporal_splits(df)
    art_path = Path(artifacts_dir)

    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(art_path / "xgb_model.json")
    baseline_lr = joblib.load(art_path / "baseline_lr.joblib")

    xgb_probs = xgb_clf.predict_proba(splits.X_test[FEATURE_COLUMNS])[:, 1]
    lr_probs = baseline_lr.predict_proba(splits.X_test[FEATURE_COLUMNS])[:, 1]

    report = {
        "test_window": {
            "total_transactions": int(len(splits.y_test)),
            "fraud_cases": int(splits.y_test.sum()),
            "fraud_rate": round(float(splits.y_test.mean()), 6),
        },
        "xgboost": {
            "pr_auc": round(float(average_precision_score(splits.y_test, xgb_probs)), 6),
            "roc_auc": round(float(roc_auc_score(splits.y_test, xgb_probs)), 6),
            "recall_at_1pct_fpr": round(recall_at_fpr(splits.y_test, xgb_probs, 0.01), 6),
        },
        "logistic_regression_baseline": {
            "pr_auc": round(float(average_precision_score(splits.y_test, lr_probs)), 6),
            "roc_auc": round(float(roc_auc_score(splits.y_test, lr_probs)), 6),
            "recall_at_1pct_fpr": round(recall_at_fpr(splits.y_test, lr_probs, 0.01), 6),
        },
    }

    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)
    (rep_path / "metrics.json").write_text(json.dumps(report, indent=2))
    return report

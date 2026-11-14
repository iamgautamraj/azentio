from __future__ import annotations

from pathlib import Path
import json
import typer

from needle.generator import generate_transactions, save_transactions
from needle.model import evaluate_models, train_models

app = typer.Typer(help="Project Needle Card Fraud Scoring CLI")


@app.command()
def generate(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    out: Path = typer.Option(Path("data/transactions.parquet"), "--out", help="Output parquet path"),
) -> None:
    df = generate_transactions(seed=seed)
    saved = save_transactions(df, out)
    pattern_counts = df["fraud_pattern"].dropna().value_counts().to_dict()
    typer.echo(
        json.dumps(
            {
                "saved_to": str(saved),
                "rows": int(len(df)),
                "cards": int(df["card_id"].nunique()),
                "fraud_count": int(df["is_fraud"].sum()),
                "fraud_rate_pct": round(float(df["is_fraud"].mean() * 100), 4),
                "fraud_patterns": pattern_counts,
            },
            indent=2,
        )
    )


@app.command()
def train(
    data: Path = typer.Option(Path("data/transactions.parquet"), "--data", help="Input parquet path"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", help="Artifacts directory"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    meta = train_models(data_path=data, artifacts_dir=artifacts, seed=seed)
    typer.echo(f"Trained {meta['model_version']} (best_iteration={meta['best_iteration']})")


@app.command()
def evaluate(
    data: Path = typer.Option(Path("data/transactions.parquet"), "--data", help="Input parquet path"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", help="Artifacts directory"),
    reports: Path = typer.Option(Path("reports"), "--reports", help="Reports directory"),
) -> None:
    report = evaluate_models(data_path=data, artifacts_dir=artifacts, reports_dir=reports)
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()

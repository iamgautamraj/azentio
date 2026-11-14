from __future__ import annotations

from pathlib import Path
import json
import typer

from needle.explain import run_explain_eval
from needle.generator import generate_transactions, save_transactions
from needle.model import evaluate_models, train_models
from needle.threshold import generate_threshold_report, run_threshold_selection

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


@app.command()
def threshold(
    data: Path = typer.Option(Path("data/transactions.parquet"), "--data", help="Input parquet path"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", help="Artifacts directory"),
    costs: Path = typer.Option(Path("config/costs.yaml"), "--costs", help="Costs YAML path"),
) -> None:
    meta = run_threshold_selection(data_path=data, artifacts_dir=artifacts, costs_path=costs)
    typer.echo(
        json.dumps(
            {
                "alert_threshold": meta["alert_threshold"],
                "validation_expected_cost_inr": meta["validation_expected_cost_inr"],
            },
            indent=2,
        )
    )


@app.command()
def report(
    data: Path = typer.Option(Path("data/transactions.parquet"), "--data", help="Input parquet path"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", help="Artifacts directory"),
    costs: Path = typer.Option(Path("config/costs.yaml"), "--costs", help="Costs YAML path"),
    reports: Path = typer.Option(Path("reports"), "--reports", help="Reports directory"),
) -> None:
    res = generate_threshold_report(
        data_path=data, artifacts_dir=artifacts, costs_path=costs, reports_dir=reports
    )
    typer.echo(json.dumps(res["policy_comparison"], indent=2))


@app.command("explain-eval")
def explain_eval(
    data: Path = typer.Option(Path("data/transactions.parquet"), "--data", help="Input parquet path"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", help="Artifacts directory"),
    unfaithful: Path = typer.Option(
        Path("data/eval/unfaithful_explanations.jsonl"), "--unfaithful", help="Unfaithful eval JSONL"
    ),
    reports: Path = typer.Option(Path("reports"), "--reports", help="Reports directory"),
) -> None:
    res = run_explain_eval(
        data_path=data,
        artifacts_dir=artifacts,
        unfaithful_path=unfaithful,
        reports_dir=reports,
    )
    typer.echo(json.dumps({k: v for k, v in res.items() if k != "unfaithful_audit"}, indent=2))


if __name__ == "__main__":
    app()

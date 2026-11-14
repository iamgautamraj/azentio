# Project Needle — Leakage-Free Card Fraud Scoring & Grounded Explanation Service

An end-to-end card-fraud scoring and explainability service built for a retail banking transaction stream (`POS`, `ECOM`, `ATM`). The system enforces strict causal feature parity between offline training and online FastAPI inference, optimizes alert thresholds using rupee business costs and analyst daily review capacity, and verifies every LLM-generated alert narrative against XGBoost TreeSHAP contributions before exposure to analysts.

---

## Architecture Overview

```text
+-----------------------+     +----------------------------------------+
|  Synthetic Generator  | --> |  Single Causal Feature Engine          |
|  (generator.py)       |     |  (features.py: CardHistory state)      |
+-----------------------+     +-------------------+--------------------+
                                                  |
                         +------------------------+------------------------+
                         | Offline (60/20/20 ts split)                     | Online (FastAPI)
                         v                                                 v
               +-------------------+                             +-----------------+
               | XGBoost + LR      |                             | POST /score     |
               | (model.py)        |                             | (api.py)        |
               +---------+---------+                             +--------+--------+
                         |                                                |
                         v                                                v
               +-------------------+                             +-----------------+
               | Validation Cost   | ---> alert_threshold -----> | Alert Store     |
               | Sweep (threshold) |      (model_meta.json)      | (store.py)      |
               +-------------------+                             +--------+--------+
                                                                          |
                                                                          v
                                                            +---------------------------+
                                                            | TreeSHAP (pred_contribs)  |
                                                            | -> LLMProvider            |
                                                            | -> check_faithfulness()   |
                                                            | -> Retry / Template       |
                                                            | -> GET /alerts/{id}/exp   |
                                                            +---------------------------+
```

---

## Synthetic Dataset Summary (`seed=42`)

Generated via `fraud generate --seed 42 --out data/transactions.parquet`:

- **Total Transactions:** `160,075` across `3,500` cards over `100` days
- **Overall Fraud Cases:** `1,035` (`0.6466%` fraud rate; `188` fraud cases in the held-out test window)
- **Fraud Pattern Breakdown:**
  - `card_testing_burst`: `380` transactions (rapid small ECOM probes followed by a high-value ECOM cash-out)
  - `atm_cashout_burst`: `225` transactions (multiple rapid ATM withdrawals within 30 minutes)
  - `new_device_high_ecom`: `220` transactions (unseen `device_id` paired with high-value electronics/digital ECOM spend)
  - `foreign_usage`: `210` transactions (sudden international spend on non-traveler cards)

### Note on Separability & Test PR-AUC (`0.9815`)
Test PR-AUC is `0.98145` because burst patterns (`card_testing_burst`, `atm_cashout_burst`) create sharp spikes in 1-hour velocity (`txn_count_1h`, `txn_sum_1h`) and `amt_to_median_ratio` that tree splits isolate cleanly. To avoid trivial `1.0` separability, the generator injects overlapping **genuine look-alikes** (`3%` international traveler cards making legitimate foreign purchases, `2.5%` genuine high-value spending spikes, and `3%` genuine new-device ECOM orders), which is why precision-recall tradeoffs remain active at high thresholds and why the Logistic Regression baseline achieves only `0.6254` PR-AUC.

---

## Model Evaluation & Metric Rationale (`reports/metrics.json`)

| Model | Test PR-AUC | Test ROC-AUC | Recall @ 1% FPR | Test Fraud Cases |
| :--- | ---: | ---: | ---: | ---: |
| **XGBoost (`scale_pos_weight`)** | **0.9815** | **0.9995** | **0.9840** | 188 / 31,925 |
| Logistic Regression (Baseline) | 0.6254 | 0.9919 | 0.7766 | 188 / 31,925 |

**Why PR-AUC is the headline metric:** With a `0.59%` fraud prevalence in the test window, a naive all-negative classifier achieves `99.41%` accuracy, and ROC-AUC (`0.9919` even for the linear baseline) looks deceptively high because true negatives dominate the false-positive rate denominator ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$). PR-AUC evaluates precision directly against recall on the rare positive class, exposing false-alert fatigue that operations teams actually pay for.

---

## Cost- & Capacity-Aware Thresholding (`reports/threshold.md`)

Using `config/costs.yaml` (`chargeback_loss_rate = 1.0`, `analyst_review_cost_inr = 150`, `customer_friction_cost_inr = 300`, `max_alerts_per_day = 12.0`), `fraud threshold` selected **`0.965`** strictly on the validation window (`INR 233,302.23` validation cost).

### Test Window Policy Comparison

| Policy | Expected Cost (INR) | Precision | Recall | Amount-Weighted Recall | Avg Alerts / Day |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Chosen (`0.965`)** | **104,265.69** | **0.9681** | **0.9681** | **0.9856** | **9.40** |
| Default (`0.500`) | 224,250.00 | 0.3333 | 0.9840 | 0.9942 | 27.75 |
| Flag Nothing (`1.01`) | 5,161,204.37 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| Flag Everything (`0.0`) | 14,309,850.00 | 0.0059 | 1.0000 | 1.0000 | 1,596.49 |

### Weakest Fraud Patterns & Channel Breakdown
- **Most missed fraud patterns:** `card_testing_burst` (initial low-rupee probe transactions occur before velocity counters build up, so the model catches the large cash-out while missing early probes) and `foreign_usage` (overlaps with legitimate international traveler cards when transaction amounts are moderate, requiring a high `0.965` cut-off to stay within the `12 alerts/day` team budget).
- **Per-channel PR-AUC:** `ATM` and `POS` have slightly lower separability than `ECOM` because `device_id` is null on physical terminals, leaving fewer behavioral signals per event. Full breakdown tables are in `reports/threshold.md`.

---

## Grounded Explanations & Faithfulness Verifier (`reports/explain_eval.json`)

Every `REVIEW` alert computes top-5 feature attributions via XGBoost's native `pred_contribs=True` (TreeSHAP). Explanations pass through `check_faithfulness()` before being returned; if rejected, the provider retries once with failure reasons and falls back to a deterministic template (`explanation_source: "template"`).

### Sample `GET /alerts/{txn_id}/explanation` Response
```json
{
  "txn_id": "EX_CASHOUT",
  "narrative": "Transaction on channel ECOM for amount 70000.0 requires manual review because Amount relative to card trailing median is 1400.0 (raises_risk), Transaction amount (INR) is 70000 (raises_risk), Card transaction count in past hour is 3 (raises_risk).",
  "claims": [
    {"feature": "amt_to_median_ratio", "value": 1400, "direction": "raises_risk"},
    {"feature": "amount", "value": 70000, "direction": "raises_risk"},
    {"feature": "txn_count_1h", "value": 3, "direction": "raises_risk"}
  ],
  "evidence": [
    {"feature": "amt_to_median_ratio", "label": "Amount relative to card trailing median", "value": 1400, "contribution": 3.412, "direction": "raises_risk"},
    {"feature": "amount", "label": "Transaction amount (INR)", "value": 70000, "contribution": 2.185, "direction": "raises_risk"},
    {"feature": "txn_count_1h", "label": "Card transaction count in past hour", "value": 3, "contribution": 1.642, "direction": "raises_risk"},
    {"feature": "is_new_device", "label": "First-time device for this card", "value": 0, "contribution": -0.31, "direction": "lowers_risk"},
    {"feature": "channel_ECOM", "label": "E-commerce channel indicator", "value": 1, "contribution": 0.289, "direction": "raises_risk"}
  ],
  "explanation_source": "llm",
  "faithfulness": {"passed": true, "reasons": []}
}
```

### Known Limitations of the Rule-Based Faithfulness Checker
1. **Spurious causal connectors:** Regex and numeric checks verify that numbers and feature names match evidence, but cannot detect flawed causal phrasing (e.g., claiming feature A *caused* feature B).
2. **Synonym / paraphrase evasion on accusations:** Phrase matching blocks explicit accusations (`"confirmed fraud"`, `"scammer"`, `"stolen card"`), but creative euphemisms blaming the cardholder could slip past a static blocklist.
3. **Number coincidences across fields:** If an ungrounded number happens to equal an unrelated field value (e.g., `mcc` or `0`/`1` binary flags), numeric set membership treats it as grounded.
4. **Omission of countervailing signals:** The checker verifies that cited claims are faithful, not that the narrative balanced `raises_risk` drivers against strong `lowers_risk` factors.

---

## Stretch Goal: `/score` Sequential Latency Benchmark

Measured via `python scripts/benchmark_latency.py` over `1,000` sequential `POST /score` calls (including stateful rolling-window feature extraction and XGBoost inference; full output in `reports/latency_benchmark.json`):

| Metric | Latency (ms) |
| :--- | ---: |
| **p50** | **4.878 ms** |
| **p95** | **7.342 ms** |
| **p99** | **13.526 ms** |
| **Mean** | **5.330 ms** |

---

## Quickstart & Running Tests

```bash
# 1. Install package and dev dependencies
pip install -e ".[dev]"

# 2. Run offline test suite (no network or API keys required)
pytest -v

# 3. Reproduce full pipeline end-to-end
fraud generate --seed 42 --out data/transactions.parquet
fraud train --data data/transactions.parquet --artifacts artifacts --seed 42
fraud evaluate --data data/transactions.parquet --artifacts artifacts --reports reports
fraud threshold --data data/transactions.parquet --artifacts artifacts --costs config/costs.yaml
fraud report --data data/transactions.parquet --artifacts artifacts --costs config/costs.yaml --reports reports
fraud explain-eval --data data/transactions.parquet --artifacts artifacts --unfaithful data/eval/unfaithful_explanations.jsonl --reports reports

# 4. Start FastAPI server (defaults to LLM_PROVIDER=fake; set LLM_PROVIDER=ollama for local Ollama)
uvicorn needle.api:app --host 0.0.0.0 --port 8000
```

## What I Would Do Next
1. **Persistent Feature & Alert Store:** Replace in-memory `CardHistory` with SQLite or Redis sorted sets and write-ahead logging so state survives worker restarts and supports horizontal scaling.
2. **Two-Tier Action Policy (`REVIEW` vs `BLOCK`):** Add a high-precision auto-decline tier above `0.995` to stop high-value account takeovers immediately without consuming analyst review queue slots.
3. **Temporal Drift & Calibration Monitoring:** Track PSI on rolling velocity features and Platt/isotonic probability calibration across weekly windows so rupee cost optimization remains well-calibrated over time.
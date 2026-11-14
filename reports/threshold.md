# Cost- & Capacity-Aware Threshold Report (Test Window)

- **Selected Threshold (Validation):** `0.965`
- **Daily Analyst Capacity Limit:** `12.0` alerts/day
- **Cost Assumptions:** Chargeback = `100%` of txn amount | Review = `INR 150.0` | False-Alert Friction = `INR 300.0`

## Policy Comparison (Test Window)

| Policy | Expected Cost (INR) | Precision | Recall | Amount-Weighted Recall | Avg Alerts / Day |
| :--- | ---: | ---: | ---: | ---: | ---: |
| chosen (0.965) | 104,265.69 | 0.9681 | 0.9681 | 0.9856 | 9.40 |
| default (0.5) | 224,250.00 | 0.3333 | 0.9840 | 0.9942 | 27.75 |
| flag_nothing (1.01) | 5,161,204.37 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| flag_everything (0.0) | 14,309,850.00 | 0.0059 | 1.0000 | 1.0000 | 1596.49 |

## Recall by Fraud Pattern (Chosen Threshold)

| Fraud Pattern | Test Cases | Caught | Recall |
| :--- | ---: | ---: | ---: |
| atm_cashout_burst | 36 | 31 | 0.8611 |
| foreign_usage | 45 | 44 | 0.9778 |
| card_testing_burst | 72 | 72 | 1.0000 |
| new_device_high_ecom | 35 | 35 | 1.0000 |

## PR-AUC by Channel (Test Window)

| Channel | Transactions | Fraud Cases | PR-AUC |
| :--- | ---: | ---: | ---: |
| ATM | 3797 | 36 | 0.9211 |
| ECOM | 12880 | 125 | 0.9955 |
| POS | 15248 | 27 | 1.0000 |

## Top 3 Weakest Areas & Hypotheses

1. **Pattern: atm_cashout_burst (Recall=0.8611)** — Initial small probe transactions within a burst have low amounts before velocity counters build up.
2. **Pattern: foreign_usage (Recall=0.9778)** — Overlaps with genuine high-value shoppers or travelers, requiring a higher threshold to respect daily review capacity.
3. **Channel: ATM (PR-AUC=0.9211)** — Fewer behavioral signals (e.g. null device_id on POS/ATM) reduce separability compared to ECOM.

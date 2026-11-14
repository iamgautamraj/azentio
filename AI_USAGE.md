# AI Usage Log — Project Needle

## Tools Used
- **Gemini (Architectural Planning & Code Generation):** Used to break down the 2-hour sprint into vertical slices, design module boundaries, and generate initial implementations for the synthetic transaction generator, causal feature store, cost-sweep optimizer, faithfulness verifier, and pytest suites.
- **GitHub Codespaces:** Execution environment for running CLI pipelines, benchmarks, `pytest`, and managing the Git branching and Pull Request workflow.

## Key Prompts & Design Ownership
1. **Single Causal Feature Engine (`src/needle/features.py`):**
   - *Prompt intent:* Build one stateful `CardHistory` class (`collections.deque` for 1h/24h windows + sorted amounts via `bisect`) shared verbatim between batch training (`transform_dataframe`) and online FastAPI scoring (`process_one`), mutating card state only *after* computing features.
   - *Why it mattered:* Eliminated offline/online feature skew by construction and guaranteed zero future leakage.
2. **Realistic Synthetic Generator with Overlapping Look-Alikes (`src/needle/generator.py`):**
   - *Prompt intent:* Generate 160,000+ transactions across 3,500 cards over 100 days with 4 distinct fraud patterns (`card_testing_burst`, `atm_cashout_burst`, `new_device_high_ecom`, `foreign_usage`) alongside genuine hard negatives (international travelers, genuine high-value spenders, genuine new devices).
   - *Why it mattered:* Kept the problem non-trivial (`0.6466%` fraud rate) so precision-recall and rupee cost tradeoffs remain meaningful.
3. **Strict Temporal Split & Validation-Only Cost Sweep (`src/needle/model.py`, `src/needle/threshold.py`):**
   - *Prompt intent:* Enforce strict timestamp inequality (`max(train.ts) < min(val.ts) <= max(val.ts) < min(test.ts)`) and isolate `select_validation_threshold()` from test-window labels while enforcing `max_alerts_per_day`.
4. **Grounded Explanations & Faithfulness Verifier (`src/needle/explain.py`):**
   - *Prompt intent:* Extract top-5 attributions via XGBoost's native `pred_contribs=True` and enforce a deterministic `check_faithfulness()` gate (feature membership, contribution sign agreement, regex numeric grounding, banned accusation phrases, word limit) with 1-retry and template fallback.

## Where the AI Was Wrong & How I Caught It
1. **Embedded Digits in Feature Labels Triggering False Rejections:**
   - *How I caught it:* When running `fraud explain-eval`, `FakeProvider` initially achieved only an `82.5%` first-attempt pass rate (`0.825`) instead of `100%`. Inspecting `reports/explain_eval.json` showed rejections like `"Unverified number '1' in narrative not grounded in evidence or transaction fields"` caused by human-readable labels such as `"Card transaction count in past 1 hour"` containing the digit `"1"`.
   - *Fix:* Reworded those labels in `src/needle/features.py` to `"past hour"` and `"past day"` so `FakeProvider` achieved `1.0` (`40/40`) first-attempt grounding without weakening the regex verifier.
2. **Word-Count Test Case Under the 80-Word Limit:**
   - *How I caught it:* In `reports/explain_eval.json`, `unfaithful_detection_rate` initially came back as `0.875` (`7/8`), with `bad_7_over_word_limit` marked `"rejected": false`. Counting its tokens showed the AI-generated test string was 76 words long—just under the 80-word threshold.
   - *Fix:* Expanded `bad_7_over_word_limit` in `data/eval/unfaithful_explanations.jsonl` to >85 words and re-verified `8/8` (`1.0`) detection.
3. **Editable Install Packaging Artifact (`*.egg-info`) in Git Tree:**
   - *How I caught it:* After running `pip install -e ".[dev]"`, `src/needle/project_needle.egg-info/` appeared in the file explorer.
   - *Fix:* Appended `*.egg-info/` to `.gitignore` immediately so packaging build artifacts were never committed.
4. **README Contradicted the Generated Threshold Report & Hypotheses Were Hard-Coded:**
   - *How I caught it:* While reviewing the README against `reports/threshold.md`, the README said `card_testing_burst` was the most-missed fraud pattern, but the report showed it at recall `1.0000`. The weakest pattern was actually `atm_cashout_burst` (`0.8611`), then `foreign_usage` (`0.9778`). The README also claimed POS had lower separability than ECOM, but POS PR-AUC was `1.0000`. The cause was that the "weakest areas" hypotheses in `threshold.py` were hard-coded strings, so they never reflected the results.
   - *Fix:* Replaced the hard-coded text with hypotheses computed from the missed rows (median `txn_count_1h`, amount and `amt_to_median_ratio`), regenerated `reports/threshold.md`, and corrected the README to match.
5. **Label-Invariance Test Did Not Change Any Labels:**
   - *How I caught it:* The test meant to prove that test-window labels don't affect the chosen threshold only called `select_validation_threshold` twice with identical inputs, so it could never fail.
   - *Fix:* Added a test that flips `is_fraud` on all test-window rows, reruns `run_threshold_selection`, and asserts the threshold is unchanged.
6. **SHAP Contributions Could Use Different Trees Than the Score:**
   - *How I caught it:* The saved model has 116 trees but `best_iteration=90`. `predict_proba` stops at the best iteration, while `pred_contribs=True` could use all trees, so the evidence might not explain the returned score.
   - *Fix:* Passed `iteration_range=(0, best_iteration + 1)` in `extract_top_evidence`.
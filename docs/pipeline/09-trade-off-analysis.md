# Stage 9 — Trade-off Analysis

**Status:** Implemented, not yet run (depends on Stage 7 producing `results/*.json`
for all six runs).

## Purpose
Aggregate all six per-(architecture, task) evaluation results into the
accuracy-vs-interpretability comparison called for by Objective 5 in the proposal, and
surface where architectures disagree in ranking — without the script itself declaring
a single "winner."

## Inputs
- `results/{architecture}_{task}.json` for each of the 6 (architecture, task) pairs
  (Stage 7). Missing files are tolerated, not required — the script runs on whatever
  subset exists and shows gaps as `None` rather than erroring.

## Process
`src/trade_off.py`:
1. `load_results()` reads whichever of the 6 JSON files exist under `results/`.
2. `build_comparison_table()` builds one row per architecture with: seven-class
   accuracy and Cohen's Kappa, ISIC2018 held-out accuracy, Grad-CAM faithfulness
   (mean IoU/Dice), binary-task accuracy, and `binary_skin_tone_accuracy_spread` —
   the max-minus-min accuracy across DDI's skin-tone groups (smaller = more
   consistent across skin tones; a direct, single-number way to report whether the
   bias-mitigation goal from the Social Issues section is actually working).
3. `rank_by()` ranks architectures on any single metric, skipping architectures with
   a missing value for that metric rather than crashing the whole ranking.
4. `summarise()` ranks by seven-class accuracy and by faithfulness (mean IoU), then:
   - if the two rankings agree on the top architecture, says so explicitly (with a
     caveat to reconfirm once all 6 runs are in).
   - if they disagree, states the trade-off directly ("X leads on accuracy but Y
     leads on faithfulness") and explicitly defers the final choice to
     student+supervisor judgement, rather than combining the two into an opaque
     single score.

**Design note (worth restating in Chapter 3):** this script deliberately does not
pick an "optimal" model via a weighted combined score. Selecting the accuracy/
interpretability balance is framed as a judgement call for discussion in the
dissertation, not something to automate away — this is the honest way to satisfy
Objective 5 ("determining the optimal CNN model... balancing both aspects") without
hiding a value judgement inside arbitrary metric weights.

## CLI usage
```
python -m src.trade_off --results-dir results/
```

## Outputs
- `results/trade_off_summary.json` — `{"comparison_table": [...], "summary": "..."}`.
- Printed summary to stdout.

## Code references
- `src/trade_off.py :: load_results()`, `build_comparison_table()`, `rank_by()`,
  `summarise()`, `_skin_tone_spread()`.

## Tests
- `tests/test_trade_off.py`.

## Notes / open items
- Run this only after all 6 Stage 7 evaluations are complete for a meaningful
  comparison — it will run on a partial set, but the ranking/trade-off narrative is
  only as complete as the input results are.

# SYNTHETIC DATA. NOT A RESULT.

> **SYNTHETIC DATA. NOT A RESULT.** Everything in this folder was made from the evaluator's synthetic generator, not from real runs. It exercises the code of issue 21 and says nothing about real models. The report on real data is `docs/results/baselines.md` (issue 23), not this folder.

| File | What it is |
|---|---|
| `report.json` | the evaluator's report: the tabular predictor inside `ScoredPolicy` (calibrated and not, at tau 0.7, 0.8, 0.9), the rule baseline and the hindsight oracle, against every single configuration, on 190 test tasks. Every verdict sentence starts with the stamp |
| `report.svg` | the frontier plot of that report, watermarked |
| `calibration.md`, `calibration.json` | accuracy and calibration before and after calibration, with intervals, reliability tables and the gap near each threshold |

The model card is `docs/MODEL-CARD-TABULAR.md`. No trained model is checked in. Regenerate everything with `scripts/synthetic_tabular_report.sh`; the files differ only in the report's `generated_at` field.

**SYNTHETIC DATA. NOT A RESULT.**

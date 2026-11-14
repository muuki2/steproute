# SYNTHETIC DATA. NOT A RESULT.

> **SYNTHETIC DATA. NOT A RESULT.** Everything in this folder was made from the evaluator's synthetic generator, not from real runs. It exercises the external-baseline adapters of issue 22 and says nothing about real models or about the published routers. The report on real data is issue 23, not this folder.

| Folder | What it is |
|---|---|
| `paretobandit/` | ParetoBandit, `full` and `online` training at five values of λ_c |
| `graphrouter/` | LLMRouter's `graphrouter` at five values of β |
| `routellm/` | RouteLLM `mf` between the cheapest and the strongest configuration, at five strong-call shares |

Each folder holds `report.json` and `report.svg`. `report.json` is the evaluator's report for that baseline's policies, the rule baseline and the hindsight oracle, against every single configuration. Every verdict sentence in it starts with the stamp. `report.svg` is the watermarked frontier plot. The 1,000-task synthetic grid (seed 0) and its 190 test tasks are the same as in `docs/synthetic/tabular-predictor/`.

How each adapter was mapped, and the fairness notes, are in `docs/BASELINES.md`. No trained model is checked in. Regenerate everything with `scripts/synthetic_baselines_report.sh` (it needs the `baselines` extra); the files differ only in the report's `generated_at` field.

**SYNTHETIC DATA. NOT A RESULT.**

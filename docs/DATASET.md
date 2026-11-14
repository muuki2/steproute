# Dataset release

`steproute dataset` turns benchmark replay output into a release directory: scrubbed traces as Parquet, the JSON Schema of the records, fixed splits by task, a datasheet and a manifest with checksums. Code: `src/steproute/dataset/`. Issue 19; the replay output it reads is described in `docs/REPLAY.md` section 4, the task set in `docs/TASKSET.md`.

The tool writes a local directory and nothing else. It never uploads. Publishing is a manual step (see the end of this page).

Parquet needs the `dataset` extra (`pip install 'steproute[dataset]'`, which installs pyarrow). The rest of the package does not import it; without the extra, `steproute dataset` says what is missing.

## Commands

```
steproute dataset build --from RUN_DIR [--from RUN_DIR ...] --out RELEASE_DIR
                        [--manifest FILE] [--catalog FILE] [--seed N]
                        [--fractions TRAIN VALIDATION TEST] [--salt S]
                        [--licence NAME] [--allow-mock]
steproute dataset check RELEASE_DIR      # checksums, splits, datasheet tables against the data
```

```python
from steproute.dataset import load

rel = load("release")  # verifies every checksum first
rel.splits["train"].runs, rel.splits["train"].steps  # lists of row dicts
rel.splits["holdout_model"]  # every run of the held-out model
rel.summary_block() == rel.datasheet_block()  # the datasheet's table, recomputed
```

`RELEASE_DIR` must not exist. The build writes to `RELEASE_DIR.partial` and renames it at the end, so a failed build leaves nothing behind. The default task set is the built-in one (`swebench-verified-v1`); `--manifest` names another.

## What a release contains

```
DATASHEET.md                  generated; the summary tables are computed from the data
manifest.json                 versions, source identities, scrub report, sha256 and size of every other file
splits.json                   every task of the task set: its stratum and its split; seed and fractions
schema/trace.schema.json      JSON Schema of the Decision and Outcome records (docs/SCHEMA.md)
schema/tables.json            the columns of the two Parquet tables, with types and meanings
data/runs.parquet             one row per run (task-level)
data/steps-train.parquet      one row per step: decision joined to its outcome
data/steps-validation.parquet
data/steps-test.parquet
data/steps-holdout_model.parquet
```

### Parquet layout, and why

Two tables, joined by `run_id`:

- **`runs`**: one row per replay run (one task under one fixed model and effort): task, stratum, split, model, effort, `success` (null when the check gave no result: not a failure), status, steps, cost as billed, cost at list price, latency. This is the table for task-level work (the grid of `docs/EVALUATION.md`).
- **`steps`**: one row per decision, with its outcome joined in. Decision and outcome are one row because the replay writes them one to one and every analysis needs both; a decision without an outcome has `has_outcome = false`. Nested parts that vary in shape (`candidates`, the surviving `features`, `signals`) are JSON strings (`candidates_json`, `features_json`, `signals_json`) so the schema stays flat and stable.

Steps are split into one file per split on purpose: reading `steps-test.parquet` is a deliberate act, and a glob over `steps-*.parquet` is the only way to mix splits by accident. Every file has the same columns. `schema/tables.json` lists them; `schema/trace.schema.json` is the schema of the original records.

## Where the rows come from

For every `--from` directory the build reads `plan.json`, `grid.jsonl`, `grid-holdout.jsonl`, `traces/` and `traces-holdout/`. It does not read `runs/`: patches, trajectories and workspace diffs hold repository code and model text and are never part of a release. `runs.jsonl` and `ledger.jsonl` are provenance and stay local.

- A step belongs to a run through its session id. Steps of attempts that are no longer in the grid (a cell that was retried) are left out and counted in the datasheet.
- Only the last outcome of each decision is kept; earlier ones (the replay reports a step twice: once from the proxy, once with the task label) are counted as superseded.
- Run directories can be combined (for example the pilot and the sparse grid). A run, or a cell of (task, model, effort, repeat), that appears twice is refused.

## Splits

Splits are **by task**: every run and step of a task, under every model, is in the same split, and no task is in two.

- The unit is the task set, not the data: `train`, `validation` and `test` are assigned to every task of the manifest (default 60 / 20 / 20), whether or not it has runs yet. A task therefore keeps its split in every release made from the same task set, seed and fractions, and a later release cannot move a task from `train` into `test`.
- Within each stratum (easy, medium, hard, unsolved) tasks are ordered by `sha256("steproute-split:SEED:TASK_ID")` and cut in that order. The number per split is the largest-remainder share of the stratum; leftover seats go to the split that is furthest behind its overall share. The default seed is 19. The result does not depend on the order of the manifest.
- **`holdout_model`** holds every run and step of the held-out model (`devstral:24b`, the catalog's `holdout = true`), for the cold-start experiment. These rows are in no other split, whatever the task's split; the column `task_split` keeps the split the task has for the other models, so the same task is held out consistently.
- Sizes on the built-in set: the 150 tasks give 90 / 30 / 30 (train: easy 24, medium 27, hard 30, unsolved 9; validation and test: 8, 9, 10, 3 each). The 20 pilot tasks, inside that split, fall 9 / 9 / 2 by chance (the pilot is not stratified like the split). Split on their own the 20 would give 12 / 4 / 4.

The loader refuses a release in which a task appears in two of the three splits.

## Scrubbing

Records go through `steproute.core.privacy.scrub_decision` and `scrub_outcome` (`docs/PRIVACY.md`): task text dropped, decision and session ids replaced by keyed hashes, timestamps rounded down to one-hour buckets, `features` and `signals` reduced to booleans, numbers and nulls. One deliberate exception: **task ids are kept**, because they are public benchmark ids, the join key to the task set and what makes the splits checkable.

The salt for the hashes is derived from the checksums of the sources, so the same sources give the same release. It is not a secret: replay ids carry nothing private. `--salt` overrides it. What the scrubbing removed (counts, and the names of the dropped feature and signal keys) is in the datasheet and in `manifest.json`.

## What the build refuses

It stops at the first problem, names the file (and line) and the record id, and writes nothing. Exit status is 2.

| Refused | Why |
|---|---|
| A decision whose `source` is not `replay` | Decision D5: only benchmark replays are published. Traces from personal Claude Code or dsh use stay local. There is no flag to include them |
| An outcome with no decision | Its source is unknown |
| A run directory without a complete `plan.json`, or one made for another task set | The plan identity is unknown |
| A record whose schema version is not the one this tool reads, or a record that does not parse | The format is unknown |
| A record in which a secret pattern matches, or one that still fails the scrubber afterwards | Replay output is redacted when it is written, so a match means a leak upstream |
| A plan run against the mock upstream (`mock = true`) | Mock data is not a measurement. `--allow-mock` exists for tests and fixtures; such a release says so on the first line of the datasheet |
| A model that is not in the catalog, or the held-out model in `traces/` (or a pool model in `traces-holdout/`) | The held-out split must stay clean |
| A run or step for a task that is not in the task set, or one marked `broken` | Its label is wrong |
| An output directory that exists | A release is never overwritten |

## Versions and checks

`manifest.json` records the steproute version, trace schema version, grid version and release format version, the task set (name, dataset revision, seed), the split seed and fractions, the identity of every source (plan name, the plan's cell checksum, the checksum of `plan.json` and of the grids; no paths), and the sha256 and size of every file. `load` checks all of them and refuses a file that was changed or added. `steproute dataset check` also recomputes the datasheet's summary tables from the Parquet files and compares them to the text.

## What is not covered

- Nothing in this tool has seen real replay data; it is tested on directories written by the replay runner against the mock upstream (`tests/test_dataset.py`). The first real release needs a dry look at the datasheet before anything is published.
- Trajectories are not published. If that is ever wanted, it needs its own scrubbing and a separate decision.
- The licence line in the datasheet is whatever `--licence` says (default `MIT`). The maintainer decides it; the tool does not add a licence file.

## Publishing (manual)

The tool stops at a local directory. To publish:

1. Build from the real run directories, without `--allow-mock`, and run `steproute dataset check`.
2. Read `DATASHEET.md` end to end, in particular the scrub report (no dropped key you did not expect) and the known biases. Confirm the licence line.
3. Choose the host (for example a Hugging Face dataset repository) and upload the directory as it is, `manifest.json` included. Do it by hand, with your own credentials; nothing in steproute does it.
4. After uploading, run `steproute dataset check` on a fresh download.

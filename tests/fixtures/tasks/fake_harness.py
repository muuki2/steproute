"""Stand-in for `python -m swebench.harness.run_evaluation` in tests (no Docker).

It parses the same arguments, records what it was given, and writes what the real harness
writes (report.json, run_instance.log) according to a marker in the patch text.
"""

import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_name")
ap.add_argument("--split")
ap.add_argument("--predictions_path")
ap.add_argument("--instance_ids", nargs="+")
ap.add_argument("--run_id")
ap.add_argument("--max_workers")
ap.add_argument("--timeout")
args = ap.parse_args()

rows = json.loads(Path(args.dataset_name).read_text())
pred = json.loads(Path(args.predictions_path).read_text().splitlines()[0])
iid = pred["instance_id"]
patch = pred["model_patch"]
Path("seen.json").write_text(
    json.dumps({"image": rows[0]["image"], "ids": args.instance_ids, "timeout": args.timeout})
)

base = "logs/evaluation" if "NEWLAYOUT" in patch else "logs/run_evaluation"
inst = Path(base) / args.run_id / pred["model_name_or_path"] / iid
inst.mkdir(parents=True)


def report(**fields):
    full = {"patch_is_None": False, "patch_exists": True, "infra_failure": False}
    full.update(fields)
    (inst / "report.json").write_text(json.dumps({iid: full}))


if "CRASH" in patch:
    sys.exit(1)
if "NOAPPLY" in patch:
    (inst / "run_instance.log").write_text(">>>>> Patch Apply Failed:\nerror: corrupt patch\n")
elif "HANG" in patch:
    (inst / "run_instance.log").write_text(f"Test timed out after {args.timeout} seconds.\n")
elif "INFRA" in patch:
    report(
        patch_successfully_applied=True,
        resolved=False,
        infra_failure=True,
        infra_failure_reason="environment: pip install failed",
    )
elif "UNRESOLVED" in patch:
    report(patch_successfully_applied=True, resolved=False)
else:
    report(patch_successfully_applied=True, resolved=True)

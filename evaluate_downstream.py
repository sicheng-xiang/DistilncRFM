"""Evaluate the four downstream tasks with all five saved models."""

import argparse
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
TASKS = {"nrc": "family", "circ": "localization", "spl": "splice", "ires": "ires"}
MODELS = {
    "student": "structRFM_11M", "teacher": "structRFM_86M",
    "dis": "DistilncRFM-EW", "dis_w": "DistilncRFM",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")

    out = ROOT / (args.output_dir or f"results/downstream_eval_{datetime.now():%Y%m%d_%H%M%S_%f}")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error(f"Use an empty output directory: {out}")
    rinalmo = read_json(ROOT / "migration/rinalmo_models.json")
    common = ["--device", args.device]
    summary = {}
    for task in dict.fromkeys(args.tasks):
        dest = out / task
        entry = ["tasks/IRES/run_ires.py"] if task == "ires" else ["run_downstream.py", task]
        subprocess.run([sys.executable, *entry, "eval", "all", *common,
                        "--batch_size", str(args.batch_size), "--output_dir", str(dest)],
                       cwd=ROOT, check=True)
        external = dest / "RiNALMo-Micro"
        subprocess.run([sys.executable, "tasks/rinalmo_downstream/run_rinalmo.py",
                        "--task", TASKS[task], "--eval_checkpoint", rinalmo[TASKS[task]],
                        *common, "--eval_batch_size", str(args.batch_size), "--output_dir", str(external)],
                       cwd=ROOT, check=True)
        metric_file = "test_results.json" if task == "spl" else "metrics.json"
        summary[task] = {
            name: {"metrics": read_json(dest / key / metric_file),
                   "efficiency": read_json(dest / key / "inference_efficiency.json")}
            for key, name in MODELS.items()
        }
        summary[task]["RiNALMo-Micro"] = {
            "metrics": read_json(external / "test_metrics.json")["metrics"],
            "efficiency": read_json(external / "inference_efficiency.json"),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()

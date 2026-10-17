
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from mcare.data import ROOT, load_tables  # noqa: E402
from mcare.experiment import run_task  # noqa: E402
from mcare.frames import build_frames  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tasks", nargs="+", default=["question", "response"])
    a = ap.parse_args()
    t = load_tables()
    mapping, q, r = build_frames(t)
    frames = {"question": q, "response": r}
    out = {}
    for task in a.tasks:
        logf = open(ROOT / "results" / f"run_log_{task}.txt", "w")

        def log(msg):
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()

        summ, per, pooled = run_task(task, frames[task], ROOT / "results" / task, log=log, repeats=a.repeats)
        out[task] = {"pooled_rep0": pooled}
        log(summ[["model", "macro_f1_mean", "macro_f1_sd", "macro_f1_ci_lo", "macro_f1_ci_hi"]].to_string())
    (ROOT / "results" / "experiments_done.json").write_text(json.dumps(out, indent=2))

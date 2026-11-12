"""
V11: the hybrid with an external archive.

Up to v10, the hybrid (anchors and ACE-SQP ladder in the initial population of SR-NSGA-II) returned the final
population of NSGA-II, so crowding-distance truncation could discard points of the ladder it started from, and its
front could be worse than its own initial population. From v11 on (methods.run_hybrid), an external archive keeps every
non-dominated solution evaluated, and the returned front is the archive reduced to 100 points by hypervolume
contribution. The hybrid is therefore ACE-SQP followed by evolutionary refinement with the remaining budget.

Blocks (each replaces the hybrid records of results/raw_<block>.pkl; the v10 file is kept as raw_<block>_v10.pkl;
skipped when results/v11_<block>.done exists):
  tune    the four hybrid settings of tuning.py on the tuning scenarios (archive version) -> results/tuning_v11_hybrid.csv
  e1      30 evaluation scenarios, M = 10 (incl. the convergence history of the base scenario)
  e4      M = 20, 30 (raw_e4) and M = 50 (raw_e4b), first ten evaluation seeds
  fresh50 ten fresh scenarios with M = 50
  stress  heterogeneous devices
  path    path loss and Rayleigh fading
Usage: python experiments_v11_hybrid.py [blocks]
"""
import json, os, pickle, shutil, sys, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
import methods as M
from model import Scenario
import experiments_rev2 as XR2

HYB = "SR-NSGA-II+SQP"
SEEDS = list(range(5))
PROCS = 10
R = M.RESULTS


def path(n):
    return os.path.join(R, f"raw_{n}.pkl")


def done(n):
    return os.path.exists(os.path.join(R, f"v11_{n}.done"))


def replace_hybrid(n, new):
    if not os.path.exists(path(n + "_v10")):
        shutil.copy2(path(n), path(n + "_v10"))
    old = pickle.load(open(path(n + "_v10"), "rb"))
    keys = {(o["kw"]["M"], o["kw"]["seed"]) for o in new}
    assert keys == {(o["kw"]["M"], o["kw"]["seed"]) for o in old if o["name"] == HYB}, n
    out = [o for o in old if o["name"] != HYB] + new
    pickle.dump(out, open(path(n), "wb"))
    open(os.path.join(R, f"v11_{n}.done"), "w").write("ok")
    print(n, "written,", len(new), "hybrid records replaced,", len(out), "in total", time.strftime("%H:%M:%S"), flush=True)


def tune_job(a):
    par, s, seed = a
    F = M.nondom(M.run_hybrid(Scenario(M=10, seed=s), seed, **par)[0])
    return json.dumps(par), s, seed, F


def kws_of(n):
    return sorted({tuple(sorted(o["kw"].items())) for o in pickle.load(open(path(n), "rb")) if o["name"] == HYB})


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    which = sys.argv[1:] or ["tune", "e1", "e4", "fresh50", "stress", "path"]
    t0 = time.time()
    if "tune" in which and not done("tune"):
        import tuning
        grid = [p for n, p, _ in tuning.GRID if n == HYB]
        jobs = [(p, s, d) for p in grid for s in tuning.SCEN for d in tuning.SEEDS]
        out = XR2.run_pool(tune_job, jobs)
        rows = []
        for s in tuning.SCEN:
            sub = [o for o in out if o[1] == s]
            ref = M.nondom(np.vstack([o[3] for o in sub]))
            lo, hi = ref.min(0), ref.max(0)
            hv = HV(ref_point=np.array([1.1, 1.1]))
            rows += [dict(setting=o[0], scenario=s, seed=o[2], HV=hv((o[3] - lo) / (hi - lo))) for o in sub]
        tab = pd.DataFrame(rows).groupby("setting").HV.agg(["mean", "std", "count"]).reset_index()
        tab.to_csv(os.path.join(R, "tuning_v11_hybrid.csv"), index=False)
        print(tab.to_string(), flush=True)
        best = json.loads(tab.sort_values("mean").iloc[-1].setting)
        cfg = json.load(open(os.path.join(R, "config.json")))
        print("selected", best, "config.json has", cfg["hybrid"], "->", "unchanged" if best == cfg["hybrid"] else "DIFFERENT", flush=True)
        open(os.path.join(R, "v11_tune.done"), "w").write(json.dumps(best))
    for n in ("e1", "e4", "fresh50", "stress", "path"):
        if n not in which or done(n):
            continue
        blocks = ("e4", "e4b") if n == "e4" else (n,)
        kws = {b: [dict(k) for k in kws_of(b + "_v10" if os.path.exists(path(b + "_v10")) else b)] for b in blocks}
        jobs = [(HYB, kw, d) for b in blocks for kw in kws[b] for d in SEEDS]
        jobs.sort(key=lambda j: -j[1]["M"])
        out = XR2.run_pool(XR2.job_run, jobs)
        for b in blocks:
            replace_hybrid(b, [o for o in out if o["kw"] in kws[b]])
    print("ALL DONE", round(time.time() - t0), "s", flush=True)

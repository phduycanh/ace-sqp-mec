"""
Select the settings of every method on tuning scenarios that are disjoint from the evaluation
scenarios, by mean hypervolume. All candidate settings are written to results/tuning.csv.
Output: results/config.json, used by experiments.py.
"""
import json, os, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
import methods as M
from model import Scenario

SCEN = [3001, 3002, 3003, 3004, 3005]
SEEDS = [0, 1, 2]
OPS = [dict(sbx_eta=a, pm_eta=b) for a in (5, 15, 30) for b in (5, 20)]
GRID = []
for shape in (1.0, 1.5, 2.0, 3.0):
    GRID.append(("ACE-SQP", dict(shape=shape), [0]))
for K in (12, 25, 50):
    GRID.append(("ACE-SQP (K)", dict(K=K), [0]))
for K, rho in ((12, 0.45), (12, 1.0), (50, 1.0), (98, 1.0)):
    GRID.append(("SR-NSGA-II+SQP", dict(K=K, rho=rho), SEEDS))
for name in ("SR-NSGA-II", "NSGA-II+S", "SPEA2+S", "WS-GA+S"):
    for o in OPS:
        GRID.append((name, o, SEEDS))
for nb in (10, 15, 30):
    for pr in (0.7, 0.9):
        GRID.append(("MOEA/D+S", dict(n_neighbors=nb, prob_neighbor_mating=pr), SEEDS[:2]))


def job(a):
    name, par, s, seed = a
    sc = Scenario(M=10, seed=s)
    if name == "ACE-SQP":
        F = M.nondom(M.run_ace(sc, seed, shape=par["shape"])[0])
    elif name == "ACE-SQP (K)":
        F = M.nondom(M.run_ace(sc, seed, K=par["K"])[0])
    elif name == "SR-NSGA-II+SQP":
        F = M.nondom(M.run_hybrid(sc, seed, **par)[0])
    elif name == "MOEA/D+S":
        F = M.run_method(name, sc, seed, dict(moead=par))["F"]
    else:
        F = M.run_method(name, sc, seed, dict(ops={name: par}))["F"]
    return name, json.dumps(par), s, seed, F


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(M.RESULTS, exist_ok=True)
    with Pool(10) as p:
        M.ensure_anchors([dict(M=10, seed=s) for s in SCEN], p)
    print("anchors ready", round(time.time() - t0), "s", flush=True)
    jobs = [(n, par, s, d) for n, par, seeds in GRID for s in SCEN for d in seeds]
    print(len(jobs), "runs", flush=True)
    with Pool(10, initializer=M.init_worker) as p:
        out = []
        for i, r in enumerate(p.imap_unordered(job, jobs, chunksize=1)):
            out.append(r)
            if i % 100 == 0:
                print(i, round(time.time() - t0), "s", flush=True)
    rows = []
    for s in SCEN:
        sub = [o for o in out if o[2] == s]
        ref = M.nondom(np.vstack([o[4] for o in sub]))
        lo, hi = ref.min(0), ref.max(0)
        hv = HV(ref_point=np.array([1.1, 1.1]))
        for o in sub:
            rows.append(dict(method=o[0], setting=o[1], scenario=s, seed=o[3], HV=hv((o[4] - lo) / (hi - lo))))
    df = pd.DataFrame(rows)
    tab = df.groupby(["method", "setting"]).HV.agg(["mean", "std", "count"]).reset_index()
    tab.to_csv(os.path.join(M.RESULTS, "tuning.csv"), index=False)
    best = {n: json.loads(g.sort_values("mean").iloc[-1].setting) for n, g in tab.groupby("method") if n != "ACE-SQP (K)"}
    cfg = dict(ace=best["ACE-SQP"], hybrid=best["SR-NSGA-II+SQP"],
               moead=best["MOEA/D+S"],
               ops={"SR-NSGA-II": best["SR-NSGA-II"], "SR-NSGA-II+SQP": best["SR-NSGA-II"],
                    "NSGA-II+S": best["NSGA-II+S"], "NSGA-II": best["NSGA-II+S"],
                    "SPEA2+S": best["SPEA2+S"], "WS-GA+S": best["WS-GA+S"]})
    json.dump(cfg, open(os.path.join(M.RESULTS, "config.json"), "w"), indent=1)
    pd.set_option("display.width", 160)
    print(tab.round(4).to_string())
    print(json.dumps(cfg, indent=1))
    print("total", round(time.time() - t0), "s")

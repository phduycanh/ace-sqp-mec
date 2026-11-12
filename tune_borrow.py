"""Selection of the budget-borrowing factor of ACE-SQP on DEVELOPMENT scenarios only
(M = 10: seeds 4001-4020; M = 50: seeds 5001-5010). The evaluation scenarios are not used.
Output: results/tune_borrow.json"""
import json, os, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import methods as M
from model import Scenario
from analysis import metric_fns

BORROW = (1.0, 2.0, 4.0, 8.0)
SETS = {10: list(range(4001, 4021)), 50: list(range(5001, 5011))}
CFG = json.load(open(os.path.join(M.RESULTS, "config.json")))


def job(a):
    m, s, b = a
    sc = Scenario(M=m, seed=s)
    F, X, used, st = M.run_ace(sc, 0, shape=CFG["ace"]["shape"], borrow=b)
    return dict(M=m, s=s, b=b, F=M.nondom(F), evals=used, conv=sum(x["status"] == 0 for x in st["levels"]) / len(st["levels"]))


if __name__ == "__main__":
    t0 = time.time()
    kws = [dict(M=m, seed=s) for m, ss in SETS.items() for s in ss]
    with Pool(10) as p:
        M.ensure_anchors(kws, p)
    with Pool(10, initializer=M.init_worker) as p:
        out = list(p.imap_unordered(job, [(m, s, b) for m, ss in SETS.items() for s in ss for b in BORROW], chunksize=1))
    rows = []
    for (m, s), g in pd.DataFrame(out).groupby(["M", "s"]):
        ref = M.nondom(np.vstack(list(g.F)))
        ev, _ = metric_fns(ref)
        for _, r in g.iterrows():
            h, i_ = ev(r.F)
            rows.append(dict(M=m, s=s, b=r.b, HV=h, IGD=i_, evals=r.evals, conv=r.conv))
    d = pd.DataFrame(rows)
    tab = d.groupby(["M", "b"]).agg(HV=("HV", "mean"), HV_min=("HV", "min"), IGD=("IGD", "mean"), evals=("evals", "mean"),
                                    conv=("conv", "mean")).reset_index()
    print(tab.to_string(), "\n", round(time.time() - t0), "s")
    json.dump(tab.to_dict("records"), open(os.path.join(M.RESULTS, "tune_borrow.json"), "w"), indent=1)
    # selection: the largest mean HV on the M = 50 development scenarios (M = 10 is insensitive to the factor)
    best = float(tab[tab.M == 50].sort_values("HV").iloc[-1].b)
    cfg_path = os.path.join(M.RESULTS, "config.json")
    cfg = json.load(open(cfg_path)); cfg["ace"]["borrow"] = best
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    print("selected borrow factor", best)

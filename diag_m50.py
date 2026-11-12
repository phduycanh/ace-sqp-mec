"""Diagnostic of the first evaluation round (even budget split, rho_b = 1): on the scenario M = 50, seed 2026,
ACE-SQP is rerun with the per-level allowance doubled (budget 60 000, rho_b = 1) and compared with the first-round
runs of raw_e4b_v1.pkl. Output: results/diag_m50_2026.json (quoted in the scalability experiment E4 of the paper)."""
import json, os, pickle
import numpy as np
import methods as M
from model import Scenario
from analysis import metric_fns

if __name__ == "__main__":
    M.init_worker()
    sc = Scenario(M=50, seed=2026)
    F, X, ev, st = M.run_ace(sc, budget=60000, borrow=1.0)
    conv = sum(l["status"] == 0 for l in st["levels"])
    v1 = pickle.load(open(os.path.join(M.RESULTS, "raw_e4b_v1.pkl"), "rb"))
    sub = [o for o in v1 if o["kw"]["M"] == 50 and o["kw"]["seed"] == 2026]
    ref = M.nondom(np.vstack([o["F"] for o in sub] + [M.nondom(F)]))
    hv, _ = metric_fns(ref)
    r = dict(evals=ev, conv=conv, solves=len(st["levels"]), HV2x=hv(M.nondom(F))[0])
    for n in (M.PROP, "SR-NSGA-II+SQP", "WS-SQP"):
        r[n] = float(np.mean([hv(o["F"])[0] for o in sub if o["name"] == n]))
    json.dump(r, open(os.path.join(M.RESULTS, "diag_m50_2026.json"), "w"), indent=1)
    print(r)

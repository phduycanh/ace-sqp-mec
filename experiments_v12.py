"""
V12: reviewer items M5 (deterministic scalarization baselines) and M9 (cost of a single operating point).

Blocks (each skipped when results/v12_<block>.done exists; earlier files are kept as raw_<block>_v11.pkl):
  e1      NBI-SQP and AEC-SQP on the 30 evaluation scenarios (M = 10)            -> appended to raw_e1.pkl
  e4      the same for M = 20, 30 (raw_e4) and M = 50 (raw_e4b), first ten seeds -> appended
  fresh50 the same on the fresh M = 50 scenarios                                 -> appended to raw_fresh50.pkl
  stress  the same on the heterogeneous scenarios                                -> appended to raw_stress.pkl
  path    the same under the path-loss channel                                   -> appended to raw_path.pkl
  single  one epsilon-constraint problem at (1 + delta) T*, delta in {0.05, 0.10, 0.25}, started from the latency
          anchor (as an operator with a single service level would solve it), with and without the activation
          restart, on the 30 evaluation scenarios and the M = 50 scenarios of E4; energy and wall-clock time
          (sequential, one thread) compared with the ACE-SQP front                -> raw_single.pkl
Usage: python experiments_v12.py [blocks]
"""
import json, os, pickle, shutil, sys, time
from multiprocessing import Pool
import numpy as np
import methods as M
from model import Scenario, objectives
from sqp_core import Counter, from_genes, model, solve_eps
import experiments_rev as XR
import experiments_rev2 as XR2

NEW = ("NBI-SQP", "AEC-SQP")
R = M.RESULTS
CFG = json.load(open(os.path.join(R, "config.json")))
S10 = [2026] + list(range(6001, 6010))
BLOCKS = {
    "e1": [dict(M=10, seed=s) for s in XR2.SCEN_E1],
    "e4": [dict(M=m, seed=s) for m in (20, 30) for s in S10],
    "e4b": [dict(M=50, seed=s) for s in S10],
    "fresh50": [dict(M=50, seed=s) for s in range(9001, 9011)],
    "stress": [dict(M=10, seed=s, **XR.STRESS) for s in XR.SCEN_STRESS],
    "path": [dict(M=10, seed=s, channel="pathloss") for s in range(9301, 9311)],
}


def path(n):
    return os.path.join(R, f"raw_{n}.pkl")


def done(n):
    return os.path.exists(os.path.join(R, f"v12_{n}.done"))


def append(n, new):
    if not os.path.exists(path(n + "_v11")):
        shutil.copy2(path(n), path(n + "_v11"))
    old = [o for o in pickle.load(open(path(n + "_v11"), "rb")) if o["name"] not in NEW]
    pickle.dump(old + new, open(path(n), "wb"))
    open(os.path.join(R, f"v12_{n}.done"), "w").write("ok")
    print(n, "written,", len(new), "new records,", len(old) + len(new), "in total", time.strftime("%H:%M:%S"), flush=True)


def single_job(kw):
    """Sequential in each worker; timings are therefore indicative, the M = 50 timings are repeated in timing_v12.py."""
    sc = Scenario(**kw)
    T, x_lat = M.latency_optimal(sc)
    E0 = float(objectives(x_lat[None], sc)[0, 1])
    t0 = time.perf_counter()
    Fa = M.run_ace(sc, 0, shape=CFG["ace"]["shape"], borrow=CFG["ace"]["borrow"])[0]
    t_ace = time.perf_counter() - t0
    out = dict(kw=kw, T=T, E0=E0, t_ace=t_ace, pts=[])
    for d in (0.05, 0.10, 0.25):
        eps = T * (1 + d)
        ladder = float(Fa[Fa[:, 0] <= eps * (1 + 1e-6), 1].min())
        for act in (False, True):
            cnt = Counter(limit=M.TOTAL)
            t0 = time.perf_counter()
            z, res, nit = solve_eps(from_genes(x_lat, sc), eps * (1 + 1e-9), sc, cnt, M.TOTAL, E0)
            E = float(model(z, sc)["E"])
            if act:
                za = M.activated_start(z, sc)
                if za is not None:
                    z2, res2, _ = solve_eps(za, eps * (1 + 1e-9), sc, cnt, M.TOTAL - cnt.n, E0)
                    m2 = model(z2, sc)
                    if max(m2["Tl"].max(), (m2["Tt"] + m2["Tm"]).max()) <= eps * (1 + 1e-6) and m2["E"] < E:
                        E = float(m2["E"])
            out["pts"].append(dict(delta=d, act=act, E=E, ladder=ladder, evals=cnt.n, status=int(res.status),
                                   time=time.perf_counter() - t0))
    return out


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    which = sys.argv[1:] or ["e1", "e4", "fresh50", "stress", "path", "single"]
    t0 = time.time()
    for n in ("e1", "e4", "fresh50", "stress", "path"):
        if n not in which:
            continue
        blocks = ("e4", "e4b") if n == "e4" else (n,)
        todo = [b for b in blocks if not done(b)]
        if not todo:
            continue
        jobs = [(m, kw, 0) for b in todo for kw in BLOCKS[b] for m in NEW]
        jobs.sort(key=lambda j: -j[1]["M"])
        out = XR2.run_pool(XR2.job_run, jobs)
        for b in todo:
            append(b, [o for o in out if o["kw"] in BLOCKS[b]])
    if "single" in which and not done("single"):
        kws = BLOCKS["e1"] + BLOCKS["e4b"]
        with Pool(XR2.PROCS, initializer=M.init_worker) as p:
            res = list(p.imap_unordered(single_job, kws, chunksize=1))
        pickle.dump(res, open(path("single"), "wb"))
        open(os.path.join(R, "v12_single.done"), "w").write("ok")
        print("single written", len(res), time.strftime("%H:%M:%S"), flush=True)
    print("ALL DONE", round(time.time() - t0), "s", flush=True)

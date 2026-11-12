"""
Main experiments of the paper (first evaluation round). Run after tuning.py. Each block stores its raw results in
results/raw_<block>.pkl and is skipped when that file already exists.

E1/E2  30 scenarios (M = 10): all methods; 5 runs per scenario for stochastic methods.
E3     10 scenarios: ACE-SQP under F_max in {5,10,15} GHz and B_max in {10,20,40} MHz.
E4     10 scenarios each for M = 20, 30: ACE-SQP, hybrid, SR-NSGA-II, NSGA-II+S.
ABL    30 scenarios: ablation of ACE-SQP.
GLOB   10 scenarios: multistart check of the local solutions at five latency levels.
"""
import json, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
import methods as M
from model import Scenario, objectives
from sqp_core import Counter, from_genes, model, solve_eps, eps_levels, to_genes
from model import W_MIN

SCEN_E1 = [2026] + list(range(6001, 6030))     # 30 scenarios; 3001-3005 tuning, 4001-4029 and 5001-5029 development
SCEN_10 = SCEN_E1[:10]
SEEDS = list(range(5))
CFG = json.load(open(os.path.join(M.RESULTS, "config.json")))
PROCS = 10


def job_e1(a):
    name, kw, seed = a
    r = M.run_method(name, Scenario(**kw), seed, CFG)
    keepX = name == M.PROP
    return dict(name=name, kw=kw, seed=seed, F=r["F"], X=r["X"] if keepX else None, time=r["time"],
                evals=r.get("evals"), sqp_evals=r.get("sqp_evals"), stats=r.get("sqp_stats"),
                hist=[(n, f) for n, f in r["hist"]] if kw.get("seed") == 2026 and seed < 5 else [])


def job_abl(a):
    label, s = a
    sc = Scenario(M=10, seed=s)
    t0 = time.time()
    F, X, used, st = M.run_ace(sc, 0, shape=CFG["ace"]["shape"], **M.ABLATIONS[label])
    return dict(label=label, s=s, F=M.nondom(F), evals=used, time=time.time() - t0,
                conv=sum(x["status"] == 0 for x in st["levels"]), levels=len(st["levels"]), kept=st["n_kept"])


def job_glob(a):
    """Compare the ladder point with the best of 20 random starts at the same latency level."""
    s, k = a
    sc = Scenario(M=10, seed=s)
    T, x_lat = M.latency_optimal(sc)
    E_ref = float(objectives(x_lat[None], sc)[0, 1])
    levels = eps_levels(T, M.t_nadir(sc), M.K_FRONT, CFG["ace"]["shape"])
    F = M.run_ace(sc, 0, shape=CFG["ace"]["shape"])[0]
    ladder_E = float(F[F[:, 0] <= levels[k] * (1 + 1e-6), 1].min())
    rng = np.random.default_rng(s * 100 + k)
    best, n_ok, vals = np.inf, 0, []
    for _ in range(20):
        x = rng.random(4 * sc.M); x[sc.M:3 * sc.M] = np.maximum(x[sc.M:3 * sc.M], W_MIN)
        zz, res, _ = solve_eps(from_genes(x, sc), levels[k] * (1 + 1e-9), sc, Counter(), 1200, E_ref)
        m = model(zz, sc)
        if res.status == 0 and max(m["Tl"].max(), (m["Tt"] + m["Tm"]).max()) <= levels[k] * (1 + 1e-6):
            n_ok += 1
            vals.append(m["E"])
            best = min(best, m["E"])
    return dict(s=s, k=k, eps=float(levels[k]), ladder=float(ladder_E), best=float(best), n_ok=n_ok,
                spread=float(np.std(vals) / np.mean(vals)) if vals else None)


def block(name, fn):
    path = os.path.join(M.RESULTS, f"raw_{name}.pkl")
    if os.path.exists(path):
        print(name, "already done", flush=True)
        return
    t0 = time.time()
    out = fn()
    pickle.dump(out, open(path, "wb"))
    print(name, "done in", round(time.time() - t0), "s,", len(out), "records", flush=True)


def run_pool(fn, jobs):
    with Pool(PROCS, initializer=M.init_worker) as p:
        out = []
        for i, r in enumerate(p.imap_unordered(fn, jobs, chunksize=1)):
            out.append(r)
            if i % 200 == 0:
                print("  ", fn.__name__, i, "/", len(jobs), flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["anchors", "abl", "glob", "e3", "e1", "e4"]
    base = [dict(M=10, seed=s) for s in SCEN_E1]
    e3 = [dict(M=10, seed=s, F_max=v * 1e9) for s in SCEN_10 for v in (5.0, 15.0)] + \
         [dict(M=10, seed=s, B_max=v * 1e6) for s in SCEN_10 for v in (10.0, 40.0)]
    e4 = [dict(M=m, seed=s) for m in (20, 30) for s in SCEN_10]
    t0 = time.time()
    if "anchors" in which:
        with Pool(PROCS) as p:
            M.ensure_anchors(base + e3, p)
        print("anchors M=10 ready", round(time.time() - t0), "s", flush=True)
    if "abl" in which:
        block("abl", lambda: run_pool(job_abl, [(l, s) for l in M.ABLATIONS for s in SCEN_E1]))
    if "glob" in which:
        block("glob", lambda: run_pool(job_glob, [(s, k) for s in SCEN_10 for k in (10, 30, 60, 90)]))
    if "e3" in which:
        block("e3", lambda: run_pool(job_e1, [(M.PROP, kw, 0) for kw in base[:10] + e3]))
    if "e1" in which:
        jobs = [(M.PROP, kw, 0) for kw in base] + [(n, kw, d) for n in M.STOCHASTIC for kw in base for d in SEEDS]
        block("e1", lambda: run_pool(job_e1, jobs))
    if "e4" in which:
        with Pool(PROCS) as p:
            M.ensure_anchors(e4, p)
        print("anchors M=20,30 ready", round(time.time() - t0), "s", flush=True)
        jobs = [(M.PROP, kw, 0) for kw in e4] + [(n, kw, d) for n in ("SR-NSGA-II+SQP", "SR-NSGA-II", "NSGA-II+S") for kw in e4 for d in SEEDS]
        block("e4", lambda: run_pool(job_e1, jobs))
    print("ALL DONE", round(time.time() - t0), "s", flush=True)

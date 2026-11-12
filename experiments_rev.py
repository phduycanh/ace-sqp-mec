"""
Additional experiments of the revision rounds (run after experiments.py).
Each block stores results/raw_<block>.pkl and is skipped when the file exists.

E4B    scalability: WS-SQP and MOEA/D+S added for M = 20, 30 (same ten scenarios as E4);
       M = 50: ACE-SQP, hybrid, WS-SQP, SR-NSGA-II, NSGA-II+S, MOEA/D+S on the same ten scenario seeds.
STRESS ten heterogeneous scenarios (task sizes 2-20 MB, frequency ceilings 0.5-2.0 GHz), where about half
       of the UDs are idle at the latency anchor: ACE-SQP, ACE-SQP without activation, WS-SQP, hybrid, SR-NSGA-II.
FDBIG  finite-difference gradients with the budget multiplied by 4M+1 (cost of one forward-difference gradient).
EXACT  global optimum of the epsilon-constraint problems by SCIP for M = 3, 4, 5 (resources scaled with M),
       compared with the ACE-SQP front at the same latency levels.
"""
import json, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
import methods as M
from model import Scenario, objectives
from sqp_core import eps_levels

SCEN_10 = [2026] + list(range(6001, 6010))
SEEDS = list(range(5))
CFG = json.load(open(os.path.join(M.RESULTS, "config.json")))
PROCS = 10
STRESS = dict(TS_range=(2e6, 20e6), fmax_range=(0.5e9, 2.0e9))
SCEN_STRESS = list(range(7001, 7011))
EXACT_SETS = {3: list(range(8001, 8011)), 4: list(range(8101, 8111)), 5: list(range(8201, 8206))}
EXACT_LEVELS = (1, 5, 15, 30, 50, 80)
EXACT_LIMIT = {3: 300.0, 4: 300.0, 5: 600.0}


def small_kw(m, s):
    return dict(M=m, seed=s, F_max=m * 1e9, B_max=2 * m * 1e6)     # the per-UD resources of the base case


def job_run(a):
    name, kw, seed = a
    sc = Scenario(**kw)
    t0 = time.time()
    if name == "ACE-SQP w/o activation":
        F, X, used, st = M.run_ace(sc, 0, shape=CFG["ace"]["shape"], activation=False)
        r = dict(F=M.nondom(F), evals=used, sqp_stats=st)
    elif name == "ACE-SQP FD x(4M+1)":
        F, X, used, st = M.run_ace(sc, 0, shape=CFG["ace"]["shape"], use_grad=False, budget=M.TOTAL * (4 * sc.M + 1))
        r = dict(F=M.nondom(F), evals=used, sqp_stats=st)
    else:
        r = M.run_method(name, sc, seed, CFG)
    st = r.get("sqp_stats")
    return dict(name=name, kw=kw, seed=seed, F=r["F"], time=time.time() - t0, evals=r.get("evals"),
                conv=(sum(x["status"] == 0 for x in st["levels"]), len(st["levels"])) if st else None,
                n_act=st["n_activated"] if st else None)


def job_exact(a):
    from global_scip import solve_global
    m, s, k = a
    sc = Scenario(**small_kw(m, s))
    T = M.latency_optimal(sc)[0]
    lev = eps_levels(T, M.t_nadir(sc), M.K_FRONT, CFG["ace"]["shape"])
    F = M.run_ace(sc, 0, shape=CFG["ace"]["shape"])[0]
    ace = float(F[F[:, 0] <= lev[k] * (1 + 1e-6), 1].min())
    g = solve_global(sc, float(lev[k]), time_limit=EXACT_LIMIT[m])
    return dict(M=m, s=s, k=k, eps=float(lev[k]), ace=ace, **g)


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
    with Pool(PROCS, initializer=M.init_worker, maxtasksperchild=20) as p:
        out = []
        for i, r in enumerate(p.imap_unordered(fn, jobs, chunksize=1)):
            out.append(r)
            if i % 20 == 0:
                print("  ", fn.__name__, i, "/", len(jobs), time.strftime("%H:%M:%S"), flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["anchors", "stress", "fdbig", "e4b", "exact"]
    t0 = time.time()
    e4_old = [dict(M=m, seed=s) for m in (20, 30) for s in SCEN_10]
    e50 = [dict(M=50, seed=s) for s in SCEN_10]
    stress = [dict(M=10, seed=s, **STRESS) for s in SCEN_STRESS]
    small = [small_kw(m, s) for m, ss in EXACT_SETS.items() for s in ss]
    if "anchors" in which:
        with Pool(PROCS) as p:
            M.ensure_anchors(e50 + stress + small, p)
        print("anchors ready", round(time.time() - t0), "s", flush=True)
    if "stress" in which:
        jobs = [(n, kw, 0) for n in (M.PROP, "ACE-SQP w/o activation") for kw in stress] + \
               [(n, kw, d) for n in ("WS-SQP", "SR-NSGA-II+SQP", "SR-NSGA-II") for kw in stress for d in SEEDS]
        block("stress", lambda: run_pool(job_run, jobs))
    if "fdbig" in which:
        base = [dict(M=10, seed=s) for s in [2026] + list(range(6001, 6030))]
        block("fdbig", lambda: run_pool(job_run, [("ACE-SQP FD x(4M+1)", kw, 0) for kw in base]))
    if "e4b" in which:
        jobs = [(n, kw, d) for n in ("WS-SQP", "MOEA/D+S") for kw in e4_old for d in SEEDS] + \
               [(M.PROP, kw, 0) for kw in e50] + \
               [(n, kw, d) for n in ("SR-NSGA-II+SQP", "WS-SQP", "SR-NSGA-II", "NSGA-II+S", "MOEA/D+S") for kw in e50 for d in SEEDS]
        jobs.sort(key=lambda j: -j[1]["M"])            # longest jobs first
        block("e4b", lambda: run_pool(job_run, jobs))
    if "exact" in which:
        jobs = [(m, s, k) for m in (5, 4, 3) for s in EXACT_SETS[m] for k in EXACT_LEVELS]
        block("exact", lambda: run_pool(job_exact, jobs))
    print("ALL DONE", round(time.time() - t0), "s", flush=True)

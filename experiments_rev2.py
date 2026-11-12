"""
Second revision: ACE-SQP with the budget-borrowing factor selected on development scenarios (tune_borrow.py),
the BCD-eps baseline, fresh M = 50 scenarios and a path-loss/fading channel.
Every run that involves the SQP ladder (ACE-SQP, its variants and the hybrid) is repeated; runs of methods that do
not use it are reused from the first round. The first-round files are kept as raw_<block>_v1.pkl.

Blocks (each writes results/raw_<block>.pkl, skipped if it exists):
  e1      ACE-SQP, hybrid (5 seeds) and BCD-eps on the 30 evaluation scenarios, merged with the other methods
  e3      ACE-SQP on the resource variations
  abl     ablation (incl. finite differences with a larger budget -> raw_fdbig.pkl)
  glob    multistart check
  e4      M = 20, 30 (raw_e4) and M = 50 (raw_e4b): ACE-SQP, hybrid, BCD-eps, merged with the other methods
  fresh50 ten fresh scenarios with M = 50 (seeds 9001-9010), all six methods and BCD-eps
  stress  heterogeneous scenarios: ACE-SQP, ACE-SQP w/o activation, hybrid, BCD-eps, merged with WS-SQP, SR-NSGA-II
  path    path loss and Rayleigh fading (seeds 9301-9310): ACE-SQP, w/o activation, BCD-eps, WS-SQP, hybrid, SR-NSGA-II
  exact   ACE-SQP values at the SCIP levels recomputed (the SCIP results are reused)
"""
import json, os, pickle, shutil, sys, time
from multiprocessing import Pool
import numpy as np
import methods as M
from model import Scenario
from sqp_core import eps_levels
import experiments as X1
import experiments_rev as XR

CFG = json.load(open(os.path.join(M.RESULTS, "config.json")))
SH, BR = CFG["ace"]["shape"], CFG["ace"]["borrow"]
PROCS = 10
SEEDS = list(range(5))
NEW = (M.PROP, "SR-NSGA-II+SQP", "BCD-eps", "ACE-SQP w/o activation")     # methods that are (re)run
SCEN_E1 = X1.SCEN_E1
SCEN_10 = SCEN_E1[:10]
PATH = dict(channel="pathloss")


def job_run(a):
    name, kw, seed = a
    sc = Scenario(**kw)
    t0 = time.time()
    if name == "ACE-SQP w/o activation":
        F, X, used, st = M.run_ace(sc, 0, shape=SH, borrow=BR, activation=False)
        r = dict(F=M.nondom(F), X=X, evals=used, sqp_stats=st)
    elif name == "ACE-SQP FD x(4M+1)":
        F, X, used, st = M.run_ace(sc, 0, shape=SH, borrow=BR, use_grad=False, budget=M.TOTAL * (4 * sc.M + 1))
        r = dict(F=M.nondom(F), X=None, evals=used, sqp_stats=st)
    else:
        r = M.run_method(name, sc, seed, CFG)
    st = r.get("sqp_stats")
    return dict(name=name, kw=kw, seed=seed, F=r["F"], X=r.get("X") if name == M.PROP else None, time=time.time() - t0,
                evals=r.get("evals"), sqp_evals=r.get("sqp_evals"), stats=st,
                conv=(sum(x["status"] == 0 for x in st["levels"]), len(st["levels"])) if st else None,
                n_act=st["n_activated"] if st else None,
                hist=[(n, f) for n, f in r.get("hist", [])] if kw.get("seed") == 2026 and kw.get("M") == 10 and len(kw) == 2 else [])


def job_abl(a):
    label, s = a
    sc = Scenario(M=10, seed=s)
    t0 = time.time()
    kw = dict(M.ABLATIONS[label]); kw.setdefault("borrow", BR)
    F, X, used, st = M.run_ace(sc, 0, shape=SH, **kw)
    return dict(label=label, s=s, F=M.nondom(F), evals=used, time=time.time() - t0,
                conv=sum(x["status"] == 0 for x in st["levels"]), levels=len(st["levels"]), kept=st["n_kept"])


def job_glob(a):
    s, k = a
    from sqp_core import Counter, from_genes, model, solve_eps
    from model import objectives, W_MIN
    sc = Scenario(M=10, seed=s)
    T, x_lat = M.latency_optimal(sc)
    E_ref = float(objectives(x_lat[None], sc)[0, 1])
    levels = eps_levels(T, M.t_nadir(sc), M.K_FRONT, SH)
    F = M.run_ace(sc, 0, shape=SH, borrow=BR)[0]
    ladder_E = float(F[F[:, 0] <= levels[k] * (1 + 1e-6), 1].min())
    rng = np.random.default_rng(s * 100 + k)
    best, n_ok, vals = np.inf, 0, []
    for _ in range(20):
        x = rng.random(4 * sc.M); x[sc.M:3 * sc.M] = np.maximum(x[sc.M:3 * sc.M], W_MIN)
        zz, res, _ = solve_eps(from_genes(x, sc), levels[k] * (1 + 1e-9), sc, Counter(), 1200, E_ref)
        m = model(zz, sc)
        if res.status == 0 and max(m["Tl"].max(), (m["Tt"] + m["Tm"]).max()) <= levels[k] * (1 + 1e-6):
            n_ok += 1; vals.append(m["E"]); best = min(best, m["E"])
    return dict(s=s, k=k, eps=float(levels[k]), ladder=float(ladder_E), best=float(best), n_ok=n_ok,
                spread=float(np.std(vals) / np.mean(vals)) if vals else None)


def job_exact_ace(a):
    m, s = a
    sc = Scenario(**XR.small_kw(m, s))
    T = M.latency_optimal(sc)[0]
    lev = eps_levels(T, M.t_nadir(sc), M.K_FRONT, SH)
    F = M.run_ace(sc, 0, shape=SH, borrow=BR)[0]
    return {(m, s, k): float(F[F[:, 0] <= lev[k] * (1 + 1e-6), 1].min()) for k in XR.EXACT_LEVELS}


def path(n):
    return os.path.join(M.RESULTS, f"raw_{n}.pkl")


def keep_v1(n):
    if os.path.exists(path(n)) and not os.path.exists(path(n + "_v1")):
        shutil.copy2(path(n), path(n + "_v1"))


def done(n):
    return os.path.exists(os.path.join(M.RESULTS, f"rev2_{n}.done"))


def finish(n, out):
    pickle.dump(out, open(path(n), "wb"))
    open(os.path.join(M.RESULTS, f"rev2_{n}.done"), "w").write("ok")
    print(n, "written,", len(out), "records", time.strftime("%H:%M:%S"), flush=True)


def run_pool(fn, jobs):
    with Pool(PROCS, initializer=M.init_worker, maxtasksperchild=20) as p:
        out = []
        for i, r in enumerate(p.imap_unordered(fn, jobs, chunksize=1)):
            out.append(r)
            if i % 20 == 0:
                print("  ", fn.__name__, i, "/", len(jobs), time.strftime("%H:%M:%S"), flush=True)
    return out


def merged(n, new, drop=NEW):
    old = pickle.load(open(path(n + "_v1"), "rb"))
    return [o for o in old if o["name"] not in drop] + new


if __name__ == "__main__":
    which = sys.argv[1:] or ["anchors", "e1", "e3", "abl", "glob", "e4", "stress", "path", "exact", "fresh50"]
    os.environ["OMP_NUM_THREADS"] = "1"
    t0 = time.time()
    base = [dict(M=10, seed=s) for s in SCEN_E1]
    stress = [dict(M=10, seed=s, **XR.STRESS) for s in XR.SCEN_STRESS]
    pathl = [dict(M=10, seed=s, **PATH) for s in range(9301, 9311)]
    fresh = [dict(M=50, seed=s) for s in range(9001, 9011)]
    for n in ("e1", "e3", "abl", "glob", "e4", "e4b", "stress", "fdbig", "exact"):
        keep_v1(n)
    if "anchors" in which:
        with Pool(PROCS) as p:
            M.ensure_anchors(pathl + fresh, p)
        print("anchors ready", round(time.time() - t0), "s", flush=True)
    if "e1" in which and not done("e1"):
        jobs = [(n, kw, 0) for n in (M.PROP, "BCD-eps") for kw in base] + [("SR-NSGA-II+SQP", kw, d) for kw in base for d in SEEDS]
        finish("e1", merged("e1", run_pool(job_run, jobs)))
    if "e3" in which and not done("e3"):
        e3kw = [o["kw"] for o in pickle.load(open(path("e3_v1"), "rb"))]
        finish("e3", run_pool(job_run, [(M.PROP, kw, 0) for kw in e3kw]))
    if "abl" in which and not done("abl"):
        finish("abl", run_pool(job_abl, [(l, s) for l in M.ABLATIONS for s in SCEN_E1]))
        finish("fdbig", run_pool(job_run, [("ACE-SQP FD x(4M+1)", kw, 0) for kw in base]))
    if "glob" in which and not done("glob"):
        finish("glob", run_pool(job_glob, [(s, k) for s in SCEN_10 for k in (10, 30, 60, 90)]))
    if "e4" in which and not done("e4"):
        kws = [dict(M=m, seed=s) for m in (20, 30, 50) for s in SCEN_10]
        jobs = [(n, kw, 0) for n in (M.PROP, "BCD-eps") for kw in kws] + [("SR-NSGA-II+SQP", kw, d) for kw in kws for d in SEEDS]
        jobs.sort(key=lambda j: -j[1]["M"])
        out = run_pool(job_run, jobs)
        finish("e4b", merged("e4b", [o for o in out if o["kw"]["M"] == 50]))
        finish("e4", merged("e4", [o for o in out if o["kw"]["M"] in (20, 30)]))
    if "stress" in which and not done("stress"):
        jobs = [(n, kw, 0) for n in (M.PROP, "ACE-SQP w/o activation", "BCD-eps") for kw in stress] + \
               [("SR-NSGA-II+SQP", kw, d) for kw in stress for d in SEEDS]
        finish("stress", merged("stress", run_pool(job_run, jobs)))
    if "path" in which and not done("path"):
        jobs = [(n, kw, 0) for n in (M.PROP, "ACE-SQP w/o activation", "BCD-eps") for kw in pathl] + \
               [(n, kw, d) for n in ("WS-SQP", "SR-NSGA-II+SQP", "SR-NSGA-II") for kw in pathl for d in SEEDS]
        finish("path", run_pool(job_run, jobs))
    if "exact" in which and not done("exact"):
        old = pickle.load(open(path("exact_v1"), "rb"))
        vals = {}
        for d in run_pool(job_exact_ace, sorted({(r["M"], r["s"]) for r in old})):
            vals.update(d)
        new = [dict(r, ace_v1=r["ace"], ace=vals[(r["M"], r["s"], r["k"])]) for r in old]
        finish("exact", new)
    if "fresh50" in which and not done("fresh50"):
        jobs = [(n, kw, 0) for n in (M.PROP, "BCD-eps") for kw in fresh] + \
               [(n, kw, d) for n in ("SR-NSGA-II+SQP", "WS-SQP", "SR-NSGA-II", "NSGA-II+S", "MOEA/D+S") for kw in fresh for d in SEEDS]
        jobs.sort(key=lambda j: j[0] not in ("SR-NSGA-II+SQP", "WS-SQP"))
        finish("fresh50", run_pool(job_run, jobs))
    print("ALL DONE", round(time.time() - t0), "s", flush=True)

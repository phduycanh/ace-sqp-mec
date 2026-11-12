"""
V10: comparison at equal wall-clock time (E8).

The common budget of 30 000 evaluations counts one call of the Jacobian as one evaluation. For M >= 20 an SQP-based run
therefore takes much longer than a run of the evolutionary methods (M = 50: about 36 s against 2 s). This script gives
the fast methods the run time of ACE-SQP and more:

  calib   sequential timing on the development scenarios 5001-5003 (one BLAS thread, anchors cached, as in timing.py):
          ACE-SQP, the hybrid, WS-SQP and BCD-eps once per scenario; SR-NSGA-II and NSGA-II+S for G_CAL generations with
          the wall-clock time of every generation; two interleaved repetitions per scenario. The time per generation b
          is the median increment after generation 100 (robust to the clock-speed changes of a laptop), and the
          equal-time number of generations is G_eq = (median ACE-SQP time) / b, ignoring the set-up cost of the
          evolutionary run in its favor; BCD-eps gets the budget 30 000 x (median ACE-SQP time / median BCD-eps time).
          -> results/v10_calib.json
  runs    SR-NSGA-II and NSGA-II+S on the scenarios of E4 (M = 20, 30, 50, first ten evaluation seeds) and on the fresh
          M = 50 scenarios, five runs each, with the same seeds and operator settings as in E4, for 2 G_eq generations;
          the front is stored after 300 generations (the original budget, which reproduces the E4 runs), G_eq and
          2 G_eq generations. BCD-eps once per scenario with the equal-time budget.  -> results/raw_v10wt.pkl
Because the generation counts and the BCD budget are fixed in advance from the sequential calibration, the runs
themselves can be executed in parallel.
Usage: python experiments_v10_walltime.py [calib] [runs]
"""
import json, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
import methods as M
from model import Scenario, MECProblem, SlackRepair
from model_ext import seed_population_exact

CFG = json.load(open(os.path.join(M.RESULTS, "config.json")))
SIZES = (20, 30, 50)
CAL_SEEDS = [5001, 5002, 5003]
G_CAL = 1000
REPS = 2
EVO = ("SR-NSGA-II", "NSGA-II+S")
SQP = (M.PROP, "SR-NSGA-II+SQP", "WS-SQP", "BCD-eps")
S10 = [2026] + list(range(6001, 6010))
FRESH = list(range(9001, 9011))
SEEDS = list(range(5))
PROCS = 10
CAL_PATH = os.path.join(M.RESULTS, "v10_calib.json")
RAW_PATH = os.path.join(M.RESULTS, "raw_v10wt.pkl")
PARTS = os.path.join(M.RESULTS, "v10wt_parts")


class Stamp(Callback):
    """Wall-clock time after every generation and the front at the requested generations."""
    def __init__(self, keep=()):
        super().__init__(); self.t, self.keep, self.fronts = [], set(keep), {}

    def notify(self, algorithm):
        self.t.append(time.perf_counter())
        g = algorithm.n_gen
        if g in self.keep:
            self.fronts[g] = M.nondom(algorithm.opt.get("F").copy())


def evo(name, sc, seed, n_gen, keep=()):
    """SR-NSGA-II or NSGA-II+S exactly as in methods.run_method, but for n_gen generations."""
    ops = CFG["ops"][name]
    seeded = seed_population_exact(sc, M.POP, np.random.default_rng(seed))
    kw = dict(repair=SlackRepair()) if name == "SR-NSGA-II" else {}
    alg = NSGA2(pop_size=M.POP, sampling=seeded, **M.ops_kwargs(ops), **kw)
    cb = Stamp(keep)
    t0 = time.perf_counter()
    res = minimize(MECProblem(sc), alg, ("n_gen", n_gen), seed=seed, callback=cb, verbose=False)
    return res, cb, t0


def calibrate():
    import highqos
    print("power throttling disabled:", highqos.enable(), flush=True)   # otherwise Windows slows the process 3-4x at times
    M.init_worker()
    out = dict(G_cal=G_CAL, seeds=CAL_SEEDS, reps=REPS, runs={})
    for m in SIZES:
        for s in CAL_SEEDS:
            sc = Scenario(M=m, seed=s)
            M.latency_optimal(sc); M.energy_optimal(sc)        # cached: the anchors are a common one-off cost
            for rep in range(REPS):                            # interleaved repetitions against drifting clock speeds
                for n in SQP:
                    t0 = time.perf_counter(); r = M.run_method(n, sc, 0, CFG); t = time.perf_counter() - t0
                    out["runs"][f"{m}|{s}|{n}|{rep}"] = dict(time=t, evals=r.get("evals"))
                    print(m, s, rep, n, round(t, 2), flush=True)
                for n in EVO:
                    _, cb, t0 = evo(n, sc, 0, G_CAL)
                    t = np.array(cb.t) - t0                          # time after generation 1, 2, ...
                    b = float(np.median(np.diff(t[99:])))            # robust time per generation
                    out["runs"][f"{m}|{s}|{n}|{rep}"] = dict(b=b, t300=float(t[299]), tG=float(t[-1]))
                    print(m, s, rep, n, "b(ms)", round(1e3 * b, 3), "t300", round(t[299], 2), flush=True)
    plan = {}
    med = lambda m, n, k: float(np.median([out["runs"][f"{m}|{s}|{n}|{r}"][k] for s in CAL_SEEDS for r in range(REPS)]))
    for m in SIZES:
        t_ace, t_bcd = med(m, M.PROP, "time"), med(m, "BCD-eps", "time")
        p = dict(t_ace=t_ace, t_bcd=t_bcd, bcd_budget=int(round(M.TOTAL * t_ace / t_bcd)))
        for n in EVO:
            b = med(m, n, "b")
            # the set-up cost before the first generation is ignored, which slightly favors the evolutionary methods
            p[n] = dict(b=b, t300=med(m, n, "t300"), G_eq=int(round(t_ace / b)))
        for n in SQP:
            p[f"t|{n}"] = med(m, n, "time")
        plan[str(m)] = p
    out["plan"] = plan
    json.dump(out, open(CAL_PATH, "w"), indent=1)
    print(json.dumps(plan, indent=1), flush=True)


def job(a):
    name, kw, seed, plan = a
    sc = Scenario(**kw)
    p = plan[str(sc.M)]
    t0 = time.time()
    if name == "BCD-eps":
        import bcd
        F, X, used, st = bcd.run_bcd(sc, budget=p["bcd_budget"], shape=CFG["ace"]["shape"], borrow=CFG["ace"]["borrow"])
        return dict(name="BCD-eps (equal time)", kw=kw, seed=0, F=M.nondom(F), evals=used, time=time.time() - t0)
    G = p[name]["G_eq"]
    res, cb, _ = evo(name, sc, seed, 2 * G, keep=(300, G, 2 * G))
    return dict(name=name, kw=kw, seed=seed, fronts=cb.fronts, G=(300, G, 2 * G), time=time.time() - t0)


def runs():
    plan = json.load(open(CAL_PATH))["plan"]
    kws = [dict(M=m, seed=s) for m in SIZES for s in S10] + [dict(M=50, seed=s) for s in FRESH]
    jobs = [(n, kw, d, plan) for n in EVO for kw in kws for d in SEEDS] + [("BCD-eps", kw, 0, plan) for kw in kws]
    jobs.sort(key=lambda j: -(j[1]["M"] * (2 * j[3][str(j[1]["M"])][j[0]]["G_eq"] if j[0] in EVO else 1000)))
    # every finished run is stored at once in results/v10wt_parts/, so an interrupted batch resumes where it stopped
    os.makedirs(PARTS, exist_ok=True)
    part = lambda j: os.path.join(PARTS, f"{j[0].replace('/', '_')}_M{j[1]['M']}_s{j[1]['seed']}_r{j[2]}.pkl")
    todo = [j for j in jobs if not os.path.exists(part(j))]
    print(len(jobs), "runs,", len(jobs) - len(todo), "already done", time.strftime("%H:%M:%S"), flush=True)
    with Pool(PROCS, initializer=M.init_worker, maxtasksperchild=10) as pool:
        for i, (j, r) in enumerate(pool.imap_unordered(job_keyed, todo, chunksize=1)):
            pickle.dump(r, open(part(j), "wb"))
            if i % 10 == 0:
                print("  ", i, "/", len(todo), time.strftime("%H:%M:%S"), flush=True)
    out = [pickle.load(open(part(j), "rb")) for j in jobs]
    pickle.dump(out, open(RAW_PATH, "wb"))
    print("written", len(out), "records", time.strftime("%H:%M:%S"), flush=True)


def job_keyed(a):
    return a, job(a)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    which = sys.argv[1:] or ["calib", "runs"]
    if "calib" in which:
        calibrate()
    if "runs" in which:
        runs()

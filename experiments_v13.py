"""
V13: which ingredient makes the difference (reviewer items M3, M4 of the critique of main_v12, 12 Nov 2026)?
Diagnostic variants (methods.VARIANTS), run on every set where NBI-SQP and AEC-SQP were run:
  ACE-SQP (random restart)  restart of ACE-SQP from a random point instead of the activated start (5 seeds)
  NBI-SQP+act               NBI-SQP with the activation restart of ACE-SQP (deterministic)
  NBI-SQP+rand              NBI-SQP with one random start per subproblem (5 seeds; more evaluations and run time)
The main raw files are not touched; the variants are stored in results/raw_v13_<block>.pkl and analyzed by
analysis_v13.py together with ACE-SQP, ACE-SQP w/o activation (where available), NBI-SQP, AEC-SQP and WS-SQP.
Each block is skipped when results/v13_<block>.done exists.
Usage: python experiments_v13.py [blocks]
"""
import os, pickle, sys, time
import methods as M
import experiments_rev as XR
import experiments_rev2 as XR2

R = M.RESULTS
S10 = [2026] + list(range(6001, 6010))
SEEDS = list(range(5))
BLOCKS = {
    "e1": [dict(M=10, seed=s) for s in XR2.SCEN_E1],
    "stress": [dict(M=10, seed=s, **XR.STRESS) for s in XR.SCEN_STRESS],
    "path": [dict(M=10, seed=s, channel="pathloss") for s in range(9301, 9311)],
    "e4": [dict(M=m, seed=s) for m in (20, 30) for s in S10],
    "e4b": [dict(M=50, seed=s) for s in S10],
    "fresh50": [dict(M=50, seed=s) for s in range(9001, 9011)],
}
STOCH = {"ACE-SQP (random restart)", "NBI-SQP+rand"}


def jobs_of(block):
    return [(n, kw, d) for kw in BLOCKS[block] for n in M.VARIANTS for d in (SEEDS if n in STOCH else [0])]


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    which = sys.argv[1:] or list(BLOCKS)
    t0 = time.time()
    for b in which:
        if os.path.exists(os.path.join(R, f"v13_{b}.done")):
            continue
        jobs = jobs_of(b)
        jobs.sort(key=lambda j: -j[1]["M"])
        out = XR2.run_pool(XR2.job_run, jobs)
        pickle.dump(out, open(os.path.join(R, f"raw_v13_{b}.pkl"), "wb"))
        open(os.path.join(R, f"v13_{b}.done"), "w").write("ok")
        print(b, "written,", len(out), "records", time.strftime("%H:%M:%S"), round(time.time() - t0), "s", flush=True)
    print("ALL DONE", round(time.time() - t0), "s", flush=True)

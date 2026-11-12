"""
Complete pipeline that regenerates every number, table and figure of the paper, in the order in which the study
was carried out (first round, revision rounds). Each step runs in a fresh process; logs go to results/.
Each experiment block is skipped when its output file exists, so the pipeline can be resumed.

Approximate run times on a 12-thread laptop (Intel i7-1355U) with OMP_NUM_THREADS=1: 6-8 hours in total,
dominated by the M = 50 runs and the global solutions of SCIP.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
STEPS = [
    ("check_gradients.py",),                              # closed-form gradients vs complex step
    ("tuning.py",),                                       # operator settings on the tuning scenarios -> config.json
    ("experiments.py", "anchors", "abl", "glob", "e3", "e1"),   # first round, M = 10
    ("experiments.py", "e4"),                             # first round, M = 20, 30 (separate process: memory)
    ("check_fast_anchors.py",),                           # fast anchors reproduce the cached ones
    ("experiments_rev.py", "anchors", "stress", "fdbig"),
    ("experiments_rev.py", "e4b"),                        # WS-SQP, MOEA/D+S for M = 20, 30; M = 50
    ("experiments_rev.py", "exact"),                      # SCIP global solutions (M = 3, 4, 5)
    ("tune_borrow.py",),                                  # borrowing factor on development scenarios
    ("experiments_rev2.py",),                             # final ACE-SQP, BCD-eps, fresh M = 50, path loss;
                                                          # keeps the first-round files as raw_<block>_v1.pkl
    ("diag_m50.py",),                                     # diagnostic of the first-round failure at M = 50
                                                          # (reads raw_e4b_v1.pkl, so it runs after experiments_rev2)
    ("experiments_rev3.py",),                             # hard budget cap for WS-SQP and BCD-eps
    ("experiments_v10_walltime.py", "calib"),             # E8: sequential calibration (idle machine)
    ("experiments_v10_walltime.py", "runs"),              # E8: equal-time runs
    ("experiments_v11_hybrid.py",),                       # ACE-SQP+EA with external archive (reproduces the runs above
                                                          # when the current methods.py was used throughout)
    ("experiments_v12.py",),                              # NBI-SQP, AEC-SQP in E1, E4, fresh M = 50, heterogeneous,
                                                          # path loss; single operating point (raw_single.pkl)
    ("analysis.py",), ("paper_stats.py",), ("analysis_rev.py",), ("analysis_v10_walltime.py",),
    ("timing.py",), ("timing_v11_hybrid.py",), ("timing_v12.py",), ("cost_bench.py",),   # sequential, idle machine
    ("analysis_v12.py",),                                 # single operating point vs the ladder -> summary_v12.json
    ("experiments_v13.py",),                              # restart variants (methods.VARIANTS) on every NBI-SQP set
    ("timing_v13.py",),                                   # sequential timing of the variants (idle machine)
    ("analysis_v13.py",),                                 # variants, E1 diagnosis -> summary_v13.json
    ("experiments_v13_ga9.py",),                          # latency-only GA as in [9]: gap to T* -> ga9_latency.json
]

if __name__ == "__main__":
    os.makedirs(R, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for step in STEPS:
        log = os.path.join(R, "log_" + "_".join(s.replace(".py", "") for s in step) + ".txt")
        print("running", " ".join(step), flush=True)
        with open(log, "w") as f:
            code = subprocess.call([sys.executable, "-u", *step], cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env)
        if code:
            sys.exit(f"step {' '.join(step)} failed with code {code}; see {log}")
    print("finished: results/summary_v7.json, summary_rev.json, paper_stats.json, timing.json, cost_bench.json, fig_*.pdf")

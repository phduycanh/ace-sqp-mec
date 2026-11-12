"""
Hard cap on the evaluation budget: WS-SQP and BCD-eps occasionally exceeded 30 000 evaluations by up to 1%
(the last SLSQP solve was allowed to finish). The counter now stops a solver at the cap and keeps its last point.
ACE-SQP and the hybrid never reached the cap, so their results are unchanged; every WS-SQP and BCD-eps run is
repeated here and replaces the previous records. The second-round files are kept as raw_<block>_v2.pkl.
"""
import os, pickle, shutil, time
from multiprocessing import Pool
import methods as M
from experiments_rev2 import job_run, SEEDS

REDO = ("WS-SQP", "BCD-eps")
BLOCKS = ("e1", "e4", "e4b", "fresh50", "stress", "path")


def path(n):
    return os.path.join(M.RESULTS, f"raw_{n}.pkl")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    t0 = time.time()
    data, jobs = {}, []
    for b in BLOCKS:
        if not os.path.exists(path(b + "_v2")):
            shutil.copy2(path(b), path(b + "_v2"))
        data[b] = pickle.load(open(path(b + "_v2"), "rb"))
        for o in data[b]:
            if o["name"] in REDO:
                jobs.append((b, (o["name"], o["kw"], o["seed"])))
    print(len(jobs), "runs to repeat", flush=True)
    jobs.sort(key=lambda j: -j[1][1]["M"])
    with Pool(10, initializer=M.init_worker, maxtasksperchild=20) as p:
        res = list(p.imap(job_run, [j[1] for j in jobs], chunksize=1))
    for b in BLOCKS:
        new = [r for (bb, _), r in zip(jobs, res) if bb == b]
        out = [o for o in data[b] if o["name"] not in REDO] + new
        pickle.dump(out, open(path(b), "wb"))
        print(b, len(new), "replaced; max evals", max((r["evals"] or 0) for r in new), flush=True)
    print("ALL DONE", round(time.time() - t0), "s", flush=True)

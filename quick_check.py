"""
Quick check (a few seconds): recomputes the base scenario of the paper (M = 10, seed 2026) and compares it with
the stored result summary, without running the 6-8 hour pipeline and without writing any file.

    python quick_check.py

Checked: the exact latency anchor T*, the exact energy anchor E*, and the device energy of the ACE-SQP front within
latency budgets of +5 %, +10 % and +25 % (results/summary_v7.json, "base_tradeoff"), plus the evaluation count.
"""
import json, os, sys, time
import numpy as np
import model as mdl
import model_ext as ext
import methods as M

HERE = os.path.dirname(os.path.abspath(__file__))
REL_TOL = 1e-4          # results are deterministic; the tolerance only absorbs floating-point differences between platforms


def close(a, b):
    return abs(a - b) <= REL_TOL * max(1.0, abs(b))


def main():
    ref = json.load(open(os.path.join(HERE, "results", "summary_v7.json")))["base_tradeoff"]
    sc = mdl.Scenario(M=10, seed=2026)
    t0 = time.time()
    T, _ = mdl.latency_optimal(sc)
    E = ext.energy_optimal(sc)[0]
    F, _, evals, _ = M.run_ace(sc, borrow=2.0)
    F = np.asarray(F)
    rows = [("T* (s)", T, ref["T_star"]), ("E at T* (J)", float(F[F[:, 0] <= T * (1 + 1e-9), 1].min()), ref["E_at_T"])]
    for p in ref["points"]:
        if p["point"].startswith("+"):
            d = float(p["point"].strip("+%")) / 100.0
            rows.append((f"E within {p['point']} (J)", float(F[F[:, 0] <= (1 + d) * T + 1e-9, 1].min()), p["E"]))
    ok = True
    print(f"{'quantity':<22}{'recomputed':>14}{'stored':>14}")
    for name, val, want in rows:
        good = close(val, want); ok &= good
        print(f"{name:<22}{val:>14.6f}{want:>14.6f}  {'OK' if good else 'MISMATCH'}")
    print(f"E* (J) = {E:.6f}; ACE-SQP: {len(F)} points, {evals} evaluations; {time.time() - t0:.1f} s")
    print("quick check passed" if ok else "quick check FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

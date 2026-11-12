"""Compare the closed-form gradients of sqp_core.py with complex-step derivatives,
which are free of the subtractive round-off error of finite differences."""
import numpy as np
from model import Scenario
from sqp_core import bounds_z, energy_grad, time_jacobians, split


def values(z, sc):
    P, f, F, B = split(z, sc.M)
    R = B * np.log2(1 + sc.gamma * sc.B_ref / B)
    Tt = P * sc.bits / R
    E = np.sum(sc.kappa * f ** 2 * (1 - P) * sc.C + sc.p_tx * Tt)
    return E, (1 - P) * sc.C / f, Tt + P * sc.C / F


for M, seed in [(10, 2026), (10, 3001), (30, 3002)]:
    sc = Scenario(M=M, seed=seed)
    lb, ub = bounds_z(sc)
    rng = np.random.default_rng(1)
    worst = np.zeros(3)
    for _ in range(50):
        z = lb + (ub - lb) * rng.uniform(0.05, 0.95, lb.size)
        z[2 * M:3 * M] = z[2 * M:3 * M] / z[2 * M:3 * M].sum() * sc.F_max / 1e9
        z[3 * M:4 * M] = z[3 * M:4 * M] / z[3 * M:4 * M].sum() * sc.B_max / 1e6
        g, (Jl, Jo) = energy_grad(z, sc), time_jacobians(z, sc)
        G, L, O = np.zeros_like(g), np.zeros_like(Jl), np.zeros_like(Jo)
        h = 1e-30
        for j in range(z.size):
            zc = z.astype(complex)
            zc[j] += 1j * h
            e, tl, to = values(zc, sc)
            G[j], L[:, j], O[:, j] = e.imag / h, tl.imag / h, to.imag / h
        rel = lambda x, y: np.max(np.abs(x - y) / (1e-12 + np.abs(y)))
        worst = np.maximum(worst, [rel(g, G), rel(Jl, L), rel(Jo, O)])
    print(f"M={M} seed={seed}: max relative error energy={worst[0]:.2e} Tl={worst[1]:.2e} To={worst[2]:.2e}")

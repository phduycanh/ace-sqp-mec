# ACE-SQP: Energy–latency trade-off in mobile edge computing

Code and data for the paper

> *Energy–Latency Trade-Off in Mobile Edge Computing: Exact Anchors and Gradient-Based ε-Constraint Continuation*
> (submitted to IEEE Access).

The package computes the Pareto front between the minimax completion time and the total device energy of a
partial-offloading MEC system with bandwidth-dependent rates and DVFS, with the proposed **ACE-SQP**
(anchored continuation ε-constraint SQP), ten baselines (NSGA-II, SR-NSGA-II, NSGA-II+S, SPEA2+S, MOEA/D+S, WS-GA+S,
WS-SQP, NBI-SQP, AEC-SQP, BCD-ε), random search, an evolutionary refinement of the ACE-SQP front (ACE-SQP+EA,
`SR-NSGA-II+SQP` in the code) and the restart variants of Section VI-I, and regenerates every table and figure of
the paper.

## Experiment labels

This README uses the labels of the paper. The scripts keep the labels of the first submission in their docstrings
and in the names of some result keys:

| Paper | Code | Content |
|---|---|---|
| E1, E2 | E1, E2 | comparison on the 30 evaluation scenarios; convergence on the base scenario |
| E3 | E5 | latency–energy exchange rate, single operating point |
| E4 | E3 | sensitivity to the edge resources |
| E5 | E4 | scalability (M = 20, 30, 50, fresh M = 50) |
| E6 | E8 | comparison at equal run time |
| E7 | E6 | heterogeneous devices, path-loss channel |
| E8 | E7 | global check by SCIP |

## Requirements

Tested with Python 3.14.3 and the package versions pinned in `requirements.txt`:

```
pip install -r requirements.txt
```

`pyscipopt` (SCIP 10) is needed only for the global solutions of E8 (`global_scip.py`). `highqos.py` (power
throttling off during sequential timing) works on Windows only; elsewhere it does nothing.

## Quick start

```python
from model import Scenario
import methods as M

sc = Scenario(M=10, seed=2026)                 # base scenario of the paper
F, X, evals, stats = M.run_ace(sc, borrow=2.0) # F: (latency, energy) of the front, X: decisions
print(F[:5], evals)
```

`Scenario` accepts `M`, `F_max`, `B_max`, `TS_range`, `fmax_range`, `channel="pathloss"` and the other model
parameters of the simulation-parameter table of the paper; all random quantities are drawn from `seed`.

## Quick checks

```
python quick_check.py              # base scenario against results/summary_v7.json (a few seconds)
python check_paper_numbers_v13.py  # numbers of Sections VI-B and VI-I against the stored result summaries
```

## Paper ↔ files

| Paper | Scripts | Result files (`results/`) |
|---|---|---|
| Table 4 (E1) | `experiments.py`, `experiments_rev2.py`, `experiments_rev3.py`, `experiments_v11_hybrid.py`, `experiments_v12.py` → `analysis.py`, `analysis_rev.py`, `paper_stats.py`; run times `timing.py`, `timing_v11_hybrid.py`, `timing_v12.py` | `summary_v7.json` (`table`, `friedman`, `ace`), `E1_scenario_means.csv`, `summary_rev.json` (`e1_hvlow`), `paper_stats.json` (`a12`), `timing.json` |
| Table 5 (energy saving within a budget) | `analysis.py`, `paper_stats.py` | `summary_v7.json` (`budget`), `paper_stats.json` (`budget_n`) |
| Fig. 2 (fronts) | `analysis.py` | `fig_pareto.pdf` |
| E1 text: diagnosis of the margin over NBI-SQP, HV₂T* differences | `analysis_v13.py` | `summary_v13.json` (`e1_diagnosis`, `e1_hvlow_vs_nbi`) |
| E1 text: latency-only GA as in Elkawkagy et al. | `experiments_v13_ga9.py` | `ga9_latency.json` |
| E3 (exchange rate, single operating point) | `analysis.py`, `experiments_v12.py`, `analysis_v12.py`, `timing_v12.py` | `summary_v7.json` (`base_tradeoff`), `summary_v12.json`, `timing.json` (`single`) |
| Table 6 (E5) | `experiments_rev.py`, `experiments_rev2.py`, `experiments_v12.py` → `analysis_rev.py` | `summary_rev.json` (`e4x`, `fresh50`) |
| E6 and Table S3 (equal run time) | `experiments_v10_walltime.py`, `analysis_v10_walltime.py` | `v10_calib.json`, `summary_v10.json` |
| Table 7, Fig. 3 (E7) | `experiments_rev.py`, `experiments_rev2.py`, `experiments_v12.py` → `analysis_rev.py` | `summary_rev.json` (`stress`, `path`, `e1_trap`), `fig_trap.pdf` |
| Table 8 (ablation) | `experiments.py`, `experiments_rev2.py` → `analysis.py`, `analysis_rev.py` | `summary_rev.json` (`ablation`) |
| Table 9 (restart versus scalarization) | `experiments_v13.py`, `timing_v13.py` → `analysis_v13.py` | `summary_v13.json`, `timing_v13.json` |
| Table 10, E8 text (global check, multistart, DVFS grid) | `global_scip.py`, `experiments_rev.py` → `analysis_rev.py`; `analysis.py` | `summary_rev.json` (`exact`, `exact_all`), `summary_v7.json` (`multistart`, `discrete`) |
| Table S1, Fig. S2 (E4) | `analysis.py` | `summary_v7.json` (`e3`), `fig_sensitivity.pdf` |
| Table S2, Fig. S3 (operating points, per-UD decisions) | `analysis.py` | `summary_v7.json` (`base_tradeoff`, `decisions`), `fig_decisions.pdf` |
| Fig. S1 (E2, convergence) | `analysis.py` | `fig_convergence.pdf` |
| Section V (cost of one evaluation), tuning, ρ_b | `cost_bench.py`, `tuning.py`, `tune_borrow.py` | `cost_bench.json`, `config.json`, `tuning.csv`, `tune_borrow.json` |

## Files

| File | Content |
|---|---|
| `model.py` | System model, objectives, slack repair (Prop. 3), exact latency anchor (Alg. 1, Prop. 1) |
| `model_ext.py` | Exact energy anchor with knapsack pruning (Prop. 2), fast anchor implementation, seeding |
| `sqp_core.py` | Closed-form gradients and Jacobians, scaling, SLSQP solve of one ε-constraint problem, budget counter |
| `methods.py` | ACE-SQP (Alg. 2), all baselines except BCD-ε, the restart variants; dispatcher `run_method` |
| `bcd.py` | BCD-ε baseline (block coordinate descent on the same ε ladder) |
| `global_scip.py` | Global solution of the ε-constraint problem by SCIP |
| `check_gradients.py`, `check_fast_anchors.py` | Complex-step check of the gradients; check of the fast anchors |
| `tuning.py`, `tune_borrow.py` | Settings of all methods on tuning scenarios; budget-borrowing factor on development scenarios |
| `experiments.py`, `experiments_rev.py`, `experiments_rev2.py`, `experiments_rev3.py` | E1–E8 except E6 (first round and revision rounds) |
| `diag_m50.py` | Diagnostic of the first-round failure with an even budget split |
| `analysis.py`, `analysis_rev.py`, `paper_stats.py` | Tables, statistics and figures |
| `timing.py`, `cost_bench.py` | Sequential run times; cost of one counted evaluation |
| `experiments_v10_walltime.py`, `analysis_v10_walltime.py` | E6: comparison at equal wall-clock time (calibration `calib`, runs `runs`) |
| `experiments_v11_hybrid.py`, `timing_v11_hybrid.py` | ACE-SQP+EA with an external archive: reruns of all its runs (and a check of its tuned setting) and its sequential timing |
| `experiments_v12.py`, `timing_v12.py`, `analysis_v12.py` | NBI-SQP and AEC-SQP in E1, E5, E7; a single operating point solved directly |
| `experiments_v13.py`, `timing_v13.py`, `analysis_v13.py` | Restart variants (ACE-SQP with a random restart point, NBI-SQP with the activation restart, NBI-SQP with a random second start per subproblem); their sequential run times; diagnosis of the E1 margin over NBI-SQP |
| `experiments_v13_ga9.py` | Latency-only GA as in Elkawkagy et al. (random initial population, 30 000 evaluations) on the 30 E1 scenarios |
| `check_paper_numbers_v13.py` | Check of the numbers stated in Sections VI-B and VI-I |
| `highqos.py` | Windows only: exempts the process from power throttling during sequential timing |
| `reproduce.py` | Complete pipeline in the order of the study |
| `quick_check.py` | Check of the base scenario against the stored results in a few seconds |

## Scenario sets

| Seeds | Use |
|---|---|
| 3001–3005 | tuning of all methods |
| 4001–4029, 5001–5029 | development of ACE-SQP; borrowing factor (4001–4020, 5001–5010); timing (5001–5003) |
| 2026, 6001–6029 | evaluation (E1, E2, E3, ablation; E4 and E5 use 2026, 6001–6009) |
| 7001–7010 | heterogeneous devices (E7) |
| 9301–9310 | path-loss channel (E7) |
| 9001–9010 | fresh scenarios with M = 50 (E5) |
| 8001–8010, 8101–8110, 8201–8205 | small instances for SCIP (E8) |

## Reproducing the paper

```
python reproduce.py
```

runs every step in the order in which the study was carried out and writes the result summaries and figures listed
above. `reproduce.py` runs every step with one BLAS thread. **Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and
`MKL_NUM_THREADS=1` also when you run a script on its own**: a multi-threaded BLAS sums in a different order, and for
the hard problems with 50 UDs this can change the path of SLSQP and hence the run (e.g., 22 953 instead of 27 859
evaluations in `diag_m50.py`). With one thread, the stored results are reproduced exactly. Experiment blocks whose output exists are skipped. On a 12-thread laptop with `OMP_NUM_THREADS=1`, a complete
run takes about 12–14 hours: 6–8 hours for the original pipeline, about 2.5 hours for the equal-run-time runs of E6,
0.5 hours for the reruns of ACE-SQP+EA, 0.2 hours for NBI-SQP and AEC-SQP, and about 2.5 hours for the restart
variants and their sequential timing. Timing steps should run on an otherwise idle machine.

Raw run records (`results/raw_*.pkl`, about 74 MB) are not included in the repository; `python reproduce.py`
regenerates them. Open `.pkl` files only from a trusted source. The folder `results/` contains the result summaries
and figures of the paper.

Notes on the history of the study, which the pipeline reproduces:

* The first round used an even split of the evaluation budget among the ε-levels (`borrow = 1`). After it
  revealed a failure with 50 UDs, the borrowing factor was selected on development scenarios (`tune_borrow.py`,
  `borrow = 2`), every run involving the SQP ladder was repeated (`experiments_rev2.py`) and ten fresh scenarios
  with 50 UDs were added. Runs of methods that do not use the ladder are reused from the first round.
* `experiments_rev3.py` enforces a hard cap of 30 000 evaluations for WS-SQP and BCD-ε, which could exceed it
  by up to 1 % in their last solve. Earlier raw files are kept as `results/raw_<block>_v1.pkl` and `_v2.pkl`.
* Run times depend on the machine; they were measured sequentially with one BLAS thread.
* E6 (first revision): `python experiments_v10_walltime.py calib` measures the run time of ACE-SQP and the time per
  generation of SR-NSGA-II and NSGA-II+S sequentially (on Windows with power throttling disabled; without it,
  timings varied by a factor of 3–4); `python experiments_v10_walltime.py runs` then reruns the evolutionary methods
  for up to twice the equal-time number of generations and BCD-ε with the equal-time budget (finished runs are
  stored in `results/v10wt_parts/`, and an interrupted batch resumes); `python analysis_v10_walltime.py` writes
  `results/summary_v10.json`.
* ACE-SQP+EA (second revision): up to v10, the hybrid returned the final population of NSGA-II, so crowding-distance
  truncation could discard points of the ACE-SQP ladder it started from. `methods.run_hybrid` now keeps an external
  archive of every non-dominated solution evaluated and returns it reduced to 100 points by hypervolume contribution;
  the hybrid is treated as an extension of ACE-SQP (`methods.EXTENSIONS`): it is excluded from the tests against
  ACE-SQP, from the Friedman ranks and from the reference front without ACE-SQP. `experiments_v11_hybrid.py` reran
  all its runs (the v10 raw files are kept as `results/raw_<block>_v10.pkl`) and confirmed its tuned setting
  (`results/tuning_v11_hybrid.csv`); `timing_v11_hybrid.py` re-timed it (input `timing_v10.json`). In a complete
  rerun with the current code these two steps reproduce what the earlier steps already computed.
* NBI-SQP and AEC-SQP (third revision): `experiments_v12.py` appends their runs to the raw files of E1, E5, E7 and
  the fresh M = 50 set (the earlier raw files are kept as `results/raw_<block>_v11.pkl`) and solves one ε-constraint
  problem at (1 + δ) T*, δ ∈ {0.05, 0.10, 0.25}, with and without the activation restart (`results/raw_single.pkl`);
  `timing_v12.py` adds their sequential run times to `results/timing.json` (input `timing_v11.json`), and
  `analysis_v12.py` compares the single operating point with the ladder.
* Restart variants (fourth revision): `experiments_v13.py` runs the three variants of `methods.VARIANTS` on E1, the
  heterogeneous and path-loss sets, M = 20/30 and both M = 50 sets (stochastic variants: five runs) and stores them in
  `results/raw_v13_<block>.pkl` (the main raw files are not touched); `analysis_v13.py` compares them with ACE-SQP,
  ACE-SQP without restart, NBI-SQP, AEC-SQP and WS-SQP (reference front per scenario from this group) and writes
  `results/summary_v13.json`. The run times stored in `raw_v13_*.pkl` were measured in a parallel pool and are not
  comparable; the paper uses the sequential times of `timing_v13.py` (`results/timing_v13.json`).

## Evaluation accounting

Every call of an objective, a constraint vector, a gradient or a Jacobian counts as one evaluation for the SQP-based
methods; one golden-section step counts as two for BCD-ε; an evolutionary method counts one evaluation per solution.
`cost_bench.py` reports the wall-clock cost of each kind of call.

## Citation

Please cite the paper (reference to be completed after publication).

## License

MIT License; see `LICENSE`.

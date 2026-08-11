# Local Search Algorithms — Solving an Underdetermined System of Equations

Practical assignment comparing three local-search optimization algorithms — **Stochastic Hill Climbing**, **Random-Restart Hill Climbing**, and **Simulated Annealing** — on the task of solving an underdetermined system of linear equations (more unknowns than equations, read from `coefficient.txt`), where the traditional algebraic approach doesn't apply and many candidate solutions exist.

Each equation is a row of coefficients followed by the right-hand-side value. The cost function is the sum of squared errors (LHS − RHS) across all equations; the goal is to find the unknowns that minimize it.

## Files
- **PAOStochasticHillClimbing.py** — hill climbing with a dynamically adjusted step size (grows on improvement, shrinks otherwise) and value clipping to keep unknowns within `[-1000, 1000]`.
- **PAORandomRestartHillClimbing.py** — runs hill climbing from multiple random starting points (10 restarts) and keeps the best solution found, to reduce the risk of getting stuck in a local optimum.
- **PAOSimulatedAnnealingAlgorithm.py** — accepts worse neighbors with a temperature-dependent probability (`exp((current_cost - neighbor_cost) / temperature)`) that decays via a cooling schedule, allowing broader exploration of the solution space.
- **coefficient.txt** — the system of equations (one per line, comma-separated: coefficients..., RHS).
- **results_chart.png** — solution quality (final cost, log scale) and computation time per algorithm.
- **ReportPracticalAssignmentOne.pdf** — full write-up (in Persian) of the neighbor-generation strategy, cost function design, parameter tuning, and comparative analysis.

## Results summary
| Algorithm | Final cost | Runtime |
|---|---|---|
| Stochastic Hill Climbing | ~1.11e+07 | ~0.09s |
| Random Restart Hill Climbing | ~1.38e-05 | ~1.82s |
| Simulated Annealing | ~4.89e+06 | ~0.11s |

Random Restart Hill Climbing found by far the best solution in this run, at the cost of ~20x the runtime of the single-pass methods, by exploring multiple starting points instead of committing to one greedy trajectory.

## Run
```
python PAOStochasticHillClimbing.py
python PAORandomRestartHillClimbing.py
python PAOSimulatedAnnealingAlgorithm.py
```
No external dependencies beyond the Python standard library (`random`, `math`, `time`).

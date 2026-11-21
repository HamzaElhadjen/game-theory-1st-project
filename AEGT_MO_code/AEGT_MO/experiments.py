import csv
import os
import time
import numpy as np
from config import DEConfig, GameConfig
from differential_evolution import DifferentialEvolution
from problems import BENCHMARKS

def run_experiments(problems, dimension, runs, budget, methods, output_dir, seed):
    os.makedirs(output_dir, exist_ok=True)
    raw_rows = []
    convergence_rows = []

    for problem_name in problems:
        objective, lower, upper, optimum = BENCHMARKS[problem_name]
        for method in methods:
            for run_id in range(runs):
                run_seed = seed + run_id * 1009 + methods.index(method) * 100000
                de_cfg = DEConfig(
                    population_size=40,
                    max_evaluations=budget,
                    lower_bound=lower,
                    upper_bound=upper,
                    seed=run_seed
                )
                game_cfg = GameConfig(method=method)
                solver = DifferentialEvolution(
                    objective, dimension, (lower, upper), de_cfg, game_cfg
                )
                start = time.perf_counter()
                result = solver.run()
                elapsed = time.perf_counter() - start

                raw_rows.append({
                    "problem": problem_name,
                    "method": method,
                    "run": run_id,
                    "seed": run_seed,
                    "dimension": dimension,
                    "budget": budget,
                    "best_fitness": result["best_fitness"],
                    "error_to_optimum": abs(result["best_fitness"] - optimum),
                    "success": int(result["best_fitness"] <= optimum + 1e-6),
                    "evaluations": result["evaluations"],
                    "runtime_seconds": elapsed,
                })
                for point in result["history"]:
                    convergence_rows.append({
                        "problem": problem_name, "method": method,
                        "run": run_id, **point
                    })
                print(f"{problem_name:12s} | {method:10s} | run {run_id+1:02d}/{runs} "
                      f"| best={result['best_fitness']:.6g} | {elapsed:.2f}s")

    _write_csv(os.path.join(output_dir, "raw_runs.csv"), raw_rows)
    _write_csv(os.path.join(output_dir, "convergence.csv"), convergence_rows)
    return raw_rows, convergence_rows

def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def summarize(raw_rows, output_dir):
    groups = {}
    for row in raw_rows:
        key = (row["problem"], row["method"])
        groups.setdefault(key, []).append(row["best_fitness"])
    summary = []
    for (problem, method), vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        summary.append({
            "problem": problem, "method": method,
            "n": len(arr), "best": float(np.min(arr)),
            "mean": float(np.mean(arr)), "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "success_rate": float(np.mean(arr <= 1e-6)),
        })
    _write_csv(os.path.join(output_dir, "summary.csv"), summary)
    return summary

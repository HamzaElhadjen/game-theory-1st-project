import csv
import os
import time
import numpy as np
from config import DEConfig, GameConfig
from differential_evolution import DifferentialEvolution
from problems import BENCHMARKS


def run_experiments(problems, dimension, runs, budget, methods, output_dir,
                    seed, success_tolerance=1e-3):
    os.makedirs(output_dir, exist_ok=True)
    raw_rows, convergence_rows = [], []

    for problem_name in problems:
        objective, lower, upper, optimum = BENCHMARKS[problem_name]
        for method in methods:
            for run_id in range(runs):
                run_seed = seed + run_id * 1009 + methods.index(method) * 100000
                de_cfg = DEConfig(
                    population_size=40, max_evaluations=budget,
                    lower_bound=lower, upper_bound=upper, seed=run_seed
                )
                game_cfg = GameConfig(method=method)
                solver = DifferentialEvolution(
                    objective, dimension, (lower, upper), de_cfg, game_cfg
                )
                start = time.perf_counter()
                result = solver.run()
                elapsed = time.perf_counter() - start
                error = abs(result["best_fitness"] - optimum)
                probs = result["final_probabilities"]

                raw_rows.append({
                    "problem": problem_name, "method": method,
                    "run": run_id + 1, "seed": run_seed,
                    "dimension": dimension, "budget": budget,
                    "best_fitness": result["best_fitness"],
                    "error_to_optimum": error,
                    "success": int(error <= success_tolerance),
                    "evaluations": result["evaluations"],
                    "runtime_seconds": elapsed,
                    "final_p_rand1": float(probs[0]),
                    "final_p_current_best": float(probs[1]),
                    "final_p_rand2": float(probs[2]),
                })
                for point in result["history"]:
                    convergence_rows.append({
                        "problem": problem_name, "method": method,
                        "run": run_id + 1, **point
                    })
                print(
                    f"{problem_name:12s} | {method:10s} | "
                    f"run {run_id + 1:02d}/{runs} | "
                    f"best={result['best_fitness']:.7g} | "
                    f"eval={result['evaluations']:5d} | {elapsed:.2f}s"
                )

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


def summarize(raw_rows, output_dir, success_tolerance=1e-3):
    groups = {}
    for row in raw_rows:
        groups.setdefault((row["problem"], row["method"]), []).append(row)

    summary = []
    for (problem, method), rows in groups.items():
        vals = np.asarray([r["best_fitness"] for r in rows], dtype=float)
        summary.append({
            "problem": problem, "method": method, "n": len(vals),
            "best": float(np.min(vals)), "worst": float(np.max(vals)),
            "mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in rows])),
            "mean_evaluations": float(np.mean([r["evaluations"] for r in rows])),
            "mean_p_rand1": float(np.mean([r["final_p_rand1"] for r in rows])),
            "mean_p_current_best": float(np.mean([r["final_p_current_best"] for r in rows])),
            "mean_p_rand2": float(np.mean([r["final_p_rand2"] for r in rows])),
        })

    _write_csv(os.path.join(output_dir, "summary.csv"), summary)
    write_text_report(summary, output_dir, success_tolerance)
    return summary


def write_text_report(summary, output_dir, success_tolerance):
    lines = [
        "AEGT-MO - RAPPORT TEXTUEL DES RESULTATS",
        "=" * 76,
        f"Tolerance de succes : erreur absolue <= {success_tolerance:g}",
        "Les valeurs de fitness sont a minimiser.", ""
    ]
    for problem in sorted({r["problem"] for r in summary}):
        rows = sorted(
            [r for r in summary if r["problem"] == problem],
            key=lambda r: r["mean"]
        )
        lines.extend([
            f"PROBLEME : {problem.upper()}", "-" * 76,
            f"{'Methode':<13}{'Meilleur':>12}{'Moyenne':>12}"
            f"{'Mediane':>12}{'Ecart-type':>12}{'Succes':>10}"
        ])
        for r in rows:
            lines.append(
                f"{r['method']:<13}{r['best']:>12.6g}{r['mean']:>12.6g}"
                f"{r['median']:>12.6g}{r['std']:>12.6g}"
                f"{r['success_rate']:>9.1%}"
            )
        lines.append("")
        for r in rows:
            lines.append(
                f"{r['method']} : pire={r['worst']:.7g} ; "
                f"temps moyen={r['mean_runtime_seconds']:.3f} s ; "
                f"evaluations moyennes={r['mean_evaluations']:.0f} ; "
                f"runs={r['n']}"
            )
            lines.append(
                "  Probabilites finales moyennes : "
                f"rand/1={r['mean_p_rand1']:.3f} ; "
                f"current-to-best/1={r['mean_p_current_best']:.3f} ; "
                f"rand/2={r['mean_p_rand2']:.3f}"
            )
        lines.append(
            f"Meilleure moyenne : {rows[0]['method']} "
            f"(fitness moyenne = {rows[0]['mean']:.7g})."
        )
        lines.append("")

    lines.extend([
        "NOTES D'INTERPRETATION",
        "- Une moyenne plus faible indique une meilleure performance moyenne.",
        "- Un ecart-type faible indique des resultats plus reguliers.",
        "- Le taux de succes utilise la tolerance indiquee en debut de rapport.",
        "- Le classement par moyenne ne remplace pas un test statistique.",
        "- Les evaluations des trois operateurs sont comptees dans le budget.",
        "", "FICHIERS COMPLEMENTAIRES",
        "- raw_runs.csv : resultats de chaque execution.",
        "- summary.csv : statistiques agregees.",
        "- convergence.csv : convergence, probabilites et gains.",
        "- figures/ : graphiques facultatifs."
    ])
    path = os.path.join(output_dir, "rapport_resultats.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path

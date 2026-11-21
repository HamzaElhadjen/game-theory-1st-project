import os
import numpy as np
import matplotlib.pyplot as plt

def create_figures(convergence_rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not convergence_rows:
        return
    problems = sorted(set(r["problem"] for r in convergence_rows))
    methods = sorted(set(r["method"] for r in convergence_rows))

    for problem in problems:
        plt.figure(figsize=(9, 5))
        for method in methods:
            subset = [r for r in convergence_rows
                      if r["problem"] == problem and r["method"] == method]
            by_run = {}
            for row in subset:
                by_run.setdefault(row["run"], []).append(
                    (row["evaluations"], row["best_fitness"])
                )
            curves = []
            common_x = None
            for points in by_run.values():
                points.sort()
                x = np.array([p[0] for p in points])
                y = np.array([p[1] for p in points])
                if common_x is None:
                    common_x = x
                if len(x) == len(common_x) and np.all(x == common_x):
                    curves.append(y)
            if curves:
                mean = np.mean(np.vstack(curves), axis=0)
                plt.plot(common_x, mean, label=method)
        plt.title(f"Convergence moyenne — {problem}")
        plt.xlabel("Nombre d'évaluations")
        plt.ylabel("Meilleure fitness (minimisation)")
        plt.yscale("symlog")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"convergence_{problem}.png"), dpi=160)
        plt.close()

    # Évolution des probabilités pour chaque méthode et problème.
    for problem in problems:
        for method in methods:
            subset = [r for r in convergence_rows
                      if r["problem"] == problem and r["method"] == method]
            if not subset:
                continue
            by_run = {}
            for row in subset:
                by_run.setdefault(row["run"], []).append(row)
            curves = []
            for points in by_run.values():
                points.sort(key=lambda z: z["evaluations"])
                curves.append(points)
            min_len = min(map(len, curves))
            x = [curves[0][i]["evaluations"] for i in range(min_len)]
            plt.figure(figsize=(9, 5))
            for key, label in [
                ("p_rand1", "DE/rand/1"),
                ("p_current_best", "DE/current-to-best/1"),
                ("p_rand2", "DE/rand/2"),
            ]:
                values = np.array([[c[i][key] for i in range(min_len)] for c in curves])
                plt.plot(x, values.mean(axis=0), label=label)
            plt.title(f"Proportions des stratégies — {problem} / {method}")
            plt.xlabel("Nombre d'évaluations")
            plt.ylabel("Probabilité")
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"strategies_{problem}_{method}.png"), dpi=160)
            plt.close()

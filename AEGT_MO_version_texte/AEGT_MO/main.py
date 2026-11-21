import argparse
import os
from experiments import run_experiments, summarize
from visualization import create_figures
from problems import BENCHMARKS


def parse_args():
    parser = argparse.ArgumentParser(
        description="AEGT-MO - Differential Evolution et theorie des jeux"
    )
    parser.add_argument("--problem", choices=["rastrigin", "rosenbrock", "all"], default="all")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--method", choices=["uniform", "replicator", "regret", "all"], default="all")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--success-tol", type=float, default=1e-3,
                        help="Tolerance absolue sur l'erreur a l'optimum")
    parser.add_argument("--output", default="results")
    parser.add_argument("--no-plots", action="store_true",
                        help="Desactive la creation des graphiques")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dim < 2:
        raise SystemExit("Erreur : --dim doit etre >= 2.")
    if args.runs < 1:
        raise SystemExit("Erreur : --runs doit etre >= 1.")
    if args.budget <= 40:
        raise SystemExit("Erreur : --budget doit etre superieur a 40 evaluations.")
    if args.success_tol < 0:
        raise SystemExit("Erreur : --success-tol doit etre positif ou nul.")

    problems = list(BENCHMARKS) if args.problem == "all" else [args.problem]
    methods = ["uniform", "replicator", "regret"] if args.method == "all" else [args.method]
    os.makedirs(args.output, exist_ok=True)

    print("\nAEGT-MO - LANCEMENT DES EXPERIENCES")
    print(f"Problemes : {', '.join(problems)}")
    print(f"Methodes : {', '.join(methods)}")
    print(f"Dimension : {args.dim} | Runs : {args.runs} | Budget : {args.budget}")
    print(f"Tolerance de succes : {args.success_tol:g}")
    print("Les trois operateurs sont evalues a chaque generation ; couts comptes.\n")

    raw, convergence = run_experiments(
        problems, args.dim, args.runs, args.budget, methods,
        args.output, args.seed, args.success_tol
    )
    summary = summarize(raw, args.output, args.success_tol)

    print("\n" + "=" * 94)
    print("RESULTATS TEXTUELS - SYNTHESE")
    print("=" * 94)
    for problem in problems:
        rows = sorted(
            [r for r in summary if r["problem"] == problem],
            key=lambda r: r["mean"]
        )
        print(f"\nPROBLEME : {problem.upper()}")
        print(f"{'Methode':<13}{'Meilleur':>13}{'Moyenne':>13}{'Mediane':>13}"
              f"{'Ecart-type':>13}{'Succes':>11}{'Temps moy.':>13}")
        print("-" * 89)
        for r in rows:
            print(f"{r['method']:<13}{r['best']:>13.6g}{r['mean']:>13.6g}"
                  f"{r['median']:>13.6g}{r['std']:>13.6g}"
                  f"{r['success_rate']:>10.1%}{r['mean_runtime_seconds']:>10.3f} s")
            print("    Probabilites finales moyennes : "
                  f"rand/1={r['mean_p_rand1']:.3f}, "
                  f"current-to-best/1={r['mean_p_current_best']:.3f}, "
                  f"rand/2={r['mean_p_rand2']:.3f}")
        if rows:
            print(f"Meilleure moyenne : {rows[0]['method']} ({rows[0]['mean']:.7g}).")

    if not args.no_plots:
        create_figures(convergence, os.path.join(args.output, "figures"))
        print("\nGraphiques crees dans figures/.")
    print(f"Rapport textuel : {os.path.join(args.output, 'rapport_resultats.txt')}")
    print(f"Donnees detaillees : {os.path.join(args.output, 'raw_runs.csv')}")
    print(f"Dossier de sortie : {args.output}")


if __name__ == "__main__":
    main()

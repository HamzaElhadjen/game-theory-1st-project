import argparse
import os
from experiments import run_experiments, summarize
from visualization import create_figures
from problems import BENCHMARKS

def parse_args():
    parser = argparse.ArgumentParser(description="AEGT-MO — expériences DE + théorie des jeux")
    parser.add_argument("--problem", choices=["rastrigin", "rosenbrock", "all"], default="all")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--method", choices=["uniform", "replicator", "regret", "all"], default="all")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="results")
    return parser.parse_args()

def main():
    args = parse_args()
    problems = list(BENCHMARKS) if args.problem == "all" else [args.problem]
    methods = ["uniform", "replicator", "regret"] if args.method == "all" else [args.method]
    os.makedirs(args.output, exist_ok=True)

    raw, convergence = run_experiments(
        problems, args.dim, args.runs, args.budget, methods, args.output, args.seed
    )
    summary = summarize(raw, args.output)
    create_figures(convergence, os.path.join(args.output, "figures"))

    print("\n=== RÉSUMÉ ===")
    for row in summary:
        print(f"{row['problem']:12s} | {row['method']:10s} | "
              f"mean={row['mean']:.6g} | std={row['std']:.4g} | "
              f"success={row['success_rate']:.1%}")
    print(f"\nRésultats enregistrés dans : {args.output}")

if __name__ == "__main__":
    main()

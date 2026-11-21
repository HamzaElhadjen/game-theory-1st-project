# AEGT-MO — Adaptive Evolutionary Game Theory for Metaheuristic Optimization

Version initiale exécutable : optimisation continue par Differential Evolution (DE).

## Fonctionnalités
- DE avec trois stratégies : `rand/1`, `current-to-best/1`, `rand/2`.
- Sélection uniforme, dynamique du réplicateur et Regret Matching à information complète.
- Gains multi-critères : amélioration de fitness, diversité et congestion.
- Coefficient de diversité temporel, mémoire EMA et détection de stagnation optionnelle.
- Benchmarks Rastrigin et Rosenbrock.
- Expériences reproductibles, CSV bruts, statistiques descriptives et graphiques.
- Tests unitaires de base.

## Installation
Python 3.8+ recommandé.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Lancer une expérience rapide
```bash
python main.py --problem rastrigin --dim 10 --runs 3 --budget 3000 --method all
```

## Lancer le protocole principal
```bash
python main.py --problem all --dim 10 --runs 30 --budget 10000 --method all
```

Les fichiers sont enregistrés dans `results/` :
- `raw_runs.csv` : résultats par exécution.
- `convergence.csv` : trajectoires agrégées.
- `figures/` : graphiques.

## Important : limites de cette version
Cette version constitue un socle fonctionnel pour la famille continue. Elle n'implémente pas encore :
- l'algorithme génétique mémétique TSP ;
- le chargement TSPLIB ;
- jDE et Probability Matching ;
- une variante bandit de Regret Matching à feedback partiel ;
- les tests Wilcoxon/Friedman et la correction de Holm.

Le Regret Matching fourni utilise une information complète estimée par simulation des trois opérateurs sur les mêmes individus. Ces évaluations supplémentaires sont comptabilisées dans le budget. Cela rend la comparaison plus coûteuse, mais explicite. Pour une comparaison stricte, toutes les méthodes doivent respecter le même budget total d'évaluations.

Les résultats ne constituent pas une preuve de supériorité. Utiliser 30 exécutions indépendantes et analyser les résultats avant toute conclusion.

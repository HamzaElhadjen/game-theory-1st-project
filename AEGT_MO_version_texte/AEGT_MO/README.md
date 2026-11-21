# AEGT-MO — Adaptive Evolutionary Game Theory for Metaheuristic Optimization

Version continue : Differential Evolution avec sélection d'opérateurs guidée par un jeu évolutif.

## Fonctionnalités
- Differential Evolution avec `rand/1`, `current-to-best/1` et `rand/2`.
- Trois contrôleurs : sélection uniforme, dynamique du réplicateur et Regret Matching.
- Gains multicritères : amélioration, diversité et congestion.
- Mémoire EMA, coefficient de diversité dynamique et option stagnation.
- Benchmarks Rastrigin et Rosenbrock.
- Résultats textuels dans la console et dans `rapport_resultats.txt`.
- Données détaillées CSV et graphiques facultatifs.

## Installation (Windows)
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Exécution rapide
```bat
python main.py --problem rastrigin --dim 10 --runs 3 --budget 3000 --method all
```

## Protocole principal
```bat
python main.py --problem all --dim 10 --runs 30 --budget 10000 --method all
```

## Affichage uniquement textuel
```bat
python main.py --problem all --dim 10 --runs 30 --budget 10000 --method all --no-plots
```

## Régler le seuil de succès
Le seuil par défaut est `1e-3` sur l'erreur absolue à l'optimum.
```bat
python main.py --problem rastrigin --runs 30 --budget 10000 --success-tol 0.01
```

## Fichiers produits dans `results/`
- `rapport_resultats.txt` : synthèse lisible, classement par moyenne, écart-type, succès et temps.
- `summary.csv` : statistiques agrégées par problème et méthode.
- `raw_runs.csv` : une ligne par exécution indépendante.
- `convergence.csv` : convergence, probabilités, gains et regrets au fil des générations.
- `figures/` : graphiques si `--no-plots` n'est pas activé.

## Point méthodologique important
Pour fournir un feedback complet à Regret Matching, chaque génération évalue les trois opérateurs sur les mêmes individus. Seul le candidat correspondant à l'opérateur tiré est intégré à la population, mais les trois évaluations sont toutes comptabilisées dans le budget. Cette conception rend les gains contrefactuels observables et le coût de cette information explicite.

Le classement par moyenne ne remplace pas un test de significativité. Cette version ne contient pas encore les tests Wilcoxon/Friedman, la correction de Holm, l'indice A12 ni l'extension TSP.

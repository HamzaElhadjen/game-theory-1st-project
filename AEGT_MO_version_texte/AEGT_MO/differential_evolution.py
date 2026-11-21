import numpy as np
from game_theory import OperatorGame, normalize_minmax, entropy

STRATEGY_NAMES = ["rand/1", "current-to-best/1", "rand/2"]


class DifferentialEvolution:
    """Differential Evolution avec sélection d'opérateurs par théorie des jeux.

    Chaque génération évalue les trois opérateurs sur les mêmes individus.
    Les gains contrefactuels sont donc complets, et toutes les évaluations sont
    comptabilisées dans le budget de la fonction objectif.
    """

    def __init__(self, objective, dimension, bounds, de_config, game_config):
        self.objective = objective
        self.dimension = int(dimension)
        self.lower, self.upper = map(float, bounds)
        self.de = de_config
        self.gc = game_config
        self.rng = np.random.default_rng(de_config.seed)
        self.population_size = max(de_config.population_size, 8)
        self.population = self.rng.uniform(
            self.lower, self.upper, (self.population_size, self.dimension)
        )
        self.fitness = np.array([objective(x) for x in self.population], dtype=float)
        self.evaluations = self.population_size
        self.best_index = int(np.argmin(self.fitness))
        self.best_x = self.population[self.best_index].copy()
        self.best_f = float(self.fitness[self.best_index])
        self.game = OperatorGame(
            3, method=game_config.method,
            learning_rate=game_config.learning_rate,
            exploration_floor=game_config.exploration_floor
        )
        self.ema = np.zeros(3, dtype=float)
        self.ema_initialized = np.zeros(3, dtype=bool)
        self.no_improvement = 0
        self.history = []

    def _mutant(self, i, strategy):
        candidates = np.delete(np.arange(self.population_size), i)
        if strategy == 0:
            r1, r2, r3 = self.rng.choice(candidates, 3, replace=False)
            v = self.population[r1] + self.de.differential_weight * (
                self.population[r2] - self.population[r3]
            )
        elif strategy == 1:
            r1, r2 = self.rng.choice(candidates, 2, replace=False)
            v = self.population[i] + self.de.differential_weight * (
                self.best_x - self.population[i]
            ) + self.de.differential_weight * (
                self.population[r1] - self.population[r2]
            )
        else:
            r1, r2, r3, r4, r5 = self.rng.choice(candidates, 5, replace=False)
            v = self.population[r1] + self.de.differential_weight * (
                self.population[r2] - self.population[r3]
            ) + self.de.differential_weight * (
                self.population[r4] - self.population[r5]
            )
        return np.clip(v, self.lower, self.upper)

    def _trial(self, i, strategy):
        mutant = self._mutant(i, strategy)
        mask = self.rng.random(self.dimension) < self.de.crossover_rate
        mask[self.rng.integers(self.dimension)] = True
        return np.where(mask, mutant, self.population[i])

    def _diversity_score(self, children):
        if len(children) == 0:
            return 0.0
        combined = np.vstack([self.population, children])
        scale = max(self.upper - self.lower, 1e-12)
        center = np.mean(combined, axis=0)
        distances = np.linalg.norm((children - center) / scale, axis=1)
        return float(np.mean(distances))

    def run(self):
        budget = int(self.de.max_evaluations)
        if budget <= self.population_size:
            raise ValueError(
                "Le budget doit être supérieur à la taille de population "
                f"({self.population_size}) pour permettre une génération."
            )

        while self.evaluations + 3 <= budget:
            remaining = budget - self.evaluations
            batch_n = min(self.population_size, remaining // 3)
            if batch_n <= 0:
                break

            chosen_strategies = [self.game.sample(self.rng) for _ in range(batch_n)]
            candidates_by_strategy = [[] for _ in range(3)]
            fitness_by_strategy = [[] for _ in range(3)]

            # Feedback complet : évaluer chaque opérateur sur les mêmes parents.
            for i in range(batch_n):
                for strategy in range(3):
                    trial = self._trial(i, strategy)
                    value = float(self.objective(trial))
                    candidates_by_strategy[strategy].append(trial)
                    fitness_by_strategy[strategy].append(value)

            self.evaluations += 3 * batch_n
            fitness_matrix = np.asarray(fitness_by_strategy, dtype=float)
            parent_fitness = self.fitness[:batch_n]
            raw_reward = np.mean(
                (parent_fitness[np.newaxis, :] - fitness_matrix)
                / (np.abs(parent_fitness[np.newaxis, :]) + 1e-12), axis=1
            )
            diversity = np.array([
                self._diversity_score(np.asarray(candidates_by_strategy[k]))
                for k in range(3)
            ], dtype=float)
            reward_norm = normalize_minmax(raw_reward)
            div_norm = normalize_minmax(diversity)

            progress = min(1.0, self.evaluations / max(1, budget))
            lam = (self.gc.lambda_max * (1.0 - progress) ** self.gc.gamma
                   if self.gc.use_dynamic_diversity else self.gc.lambda_max)
            if not self.gc.use_diversity:
                lam = 0.0
            congestion = self.gc.congestion if self.gc.use_congestion else 0.0
            payoffs = reward_norm + lam * div_norm - congestion * self.game.probabilities

            if self.gc.use_memory:
                for k in range(3):
                    if not self.ema_initialized[k]:
                        self.ema[k] = payoffs[k]
                        self.ema_initialized[k] = True
                    else:
                        self.ema[k] = (self.gc.ema_beta * self.ema[k]
                                       + (1.0 - self.gc.ema_beta) * payoffs[k])
                payoffs_for_game = self.ema.copy()
            else:
                payoffs_for_game = payoffs.copy()

            if self.gc.use_stagnation and self.no_improvement >= self.gc.stagnation_window:
                payoffs_for_game += self.gc.stagnation_boost * div_norm
            self.game.update(payoffs_for_game)

            # Seul l'enfant correspondant à la stratégie tirée est retenu.
            improved_global = False
            for local_i, strategy in enumerate(chosen_strategies):
                candidate = candidates_by_strategy[strategy][local_i]
                value = fitness_by_strategy[strategy][local_i]
                if value <= self.fitness[local_i]:
                    self.population[local_i] = candidate
                    self.fitness[local_i] = value
                if value < self.best_f:
                    self.best_f = value
                    self.best_x = candidate.copy()
                    improved_global = True

            self.no_improvement = 0 if improved_global else self.no_improvement + 1
            self.best_index = int(np.argmin(self.fitness))
            self.history.append({
                "evaluations": self.evaluations,
                "best_fitness": self.best_f,
                "mean_fitness": float(np.mean(self.fitness)),
                "diversity": float(np.mean(np.std(self.population, axis=0))),
                "entropy": entropy(self.game.probabilities),
                "p_rand1": float(self.game.probabilities[0]),
                "p_current_best": float(self.game.probabilities[1]),
                "p_rand2": float(self.game.probabilities[2]),
                "regret_rand1": float(self.game.cumulative_regret[0]),
                "regret_current_best": float(self.game.cumulative_regret[1]),
                "regret_rand2": float(self.game.cumulative_regret[2]),
                "gain_rand1": float(payoffs_for_game[0]),
                "gain_current_best": float(payoffs_for_game[1]),
                "gain_rand2": float(payoffs_for_game[2]),
            })

        return {
            "best_fitness": self.best_f,
            "best_solution": self.best_x.copy(),
            "evaluations": self.evaluations,
            "history": self.history,
            "final_probabilities": self.game.probabilities.copy(),
        }

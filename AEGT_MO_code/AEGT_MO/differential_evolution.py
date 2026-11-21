import numpy as np
from game_theory import OperatorGame, normalize_minmax, entropy

STRATEGY_NAMES = ["rand/1", "current-to-best/1", "rand/2"]

class DifferentialEvolution:
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
        if strategy == 0:  # DE/rand/1
            r1, r2, r3 = self.rng.choice(candidates, 3, replace=False)
            v = self.population[r1] + self.de.differential_weight * (
                self.population[r2] - self.population[r3]
            )
        elif strategy == 1:  # DE/current-to-best/1
            r1, r2 = self.rng.choice(candidates, 2, replace=False)
            v = self.population[i] + self.de.differential_weight * (
                self.best_x - self.population[i]
            ) + self.de.differential_weight * (
                self.population[r1] - self.population[r2]
            )
        else:  # DE/rand/2
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
        # Diversité moyenne des enfants par rapport au barycentre de la population.
        combined = np.vstack([self.population, children])
        scale = max(self.upper - self.lower, 1e-12)
        center = np.mean(combined, axis=0)
        distances = np.linalg.norm((children - center) / scale, axis=1)
        return float(np.mean(distances))

    def run(self):
        budget = self.de.max_evaluations
        while self.evaluations < budget:
            remaining = budget - self.evaluations
            batch_n = min(self.population_size, remaining)
            # Chaque individu reçoit une stratégie selon la distribution courante.
            selected = [self.game.sample(self.rng) for _ in range(batch_n)]
            children = []
            child_f = []
            parents = []
            # Génération de candidats et évaluation.
            for j, strategy in enumerate(selected):
                i = j % self.population_size
                trial = self._trial(i, strategy)
                children.append(trial)
                child_f.append(self.objective(trial))
                parents.append(i)
            children = np.asarray(children)
            child_f = np.asarray(child_f)
            self.evaluations += batch_n

            # Gains par opérateur : amélioration relative moyenne et diversité.
            raw_reward = np.zeros(3)
            diversity = np.zeros(3)
            counts = np.zeros(3)
            for child, f_child, parent_i, strategy in zip(
                children, child_f, parents, selected
            ):
                f_parent = self.fitness[parent_i]
                improvement = max(0.0, (f_parent - f_child) /
                                  (abs(f_parent) + 1e-12))
                raw_reward[strategy] += improvement
                counts[strategy] += 1
            for k in range(3):
                if counts[k] > 0:
                    raw_reward[k] /= counts[k]
                    diversity[k] = self._diversity_score(children[
                        np.array(selected) == k
                    ]) if np.any(np.array(selected) == k) else 0.0

            reward_norm = normalize_minmax(raw_reward)
            div_norm = normalize_minmax(diversity)
            progress = min(1.0, self.evaluations / max(1, budget))
            if self.gc.use_dynamic_diversity:
                lam = self.gc.lambda_max * (1.0 - progress) ** self.gc.gamma
            else:
                lam = self.gc.lambda_max
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
                        self.ema[k] = (
                            self.gc.ema_beta * self.ema[k]
                            + (1.0 - self.gc.ema_beta) * payoffs[k]
                        )
                payoffs_for_game = self.ema.copy()
            else:
                payoffs_for_game = payoffs.copy()

            if self.gc.use_stagnation and self.no_improvement >= self.gc.stagnation_window:
                payoffs_for_game += self.gc.stagnation_boost * div_norm

            self.game.update(payoffs_for_game)

            # Sélection environnementale DE : remplacement si amélioration.
            improved_global = False
            for child, f_child, parent_i in zip(children, child_f, parents):
                if f_child <= self.fitness[parent_i]:
                    self.population[parent_i] = child
                    self.fitness[parent_i] = f_child
                if f_child < self.best_f:
                    self.best_f = float(f_child)
                    self.best_x = child.copy()
                    improved_global = True

            if improved_global:
                self.no_improvement = 0
            else:
                self.no_improvement += 1

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
            })

        return {
            "best_fitness": self.best_f,
            "best_solution": self.best_x.copy(),
            "evaluations": self.evaluations,
            "history": self.history,
        }

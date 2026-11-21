import numpy as np


class OperatorGame:
    """Contrôleur de sélection des opérateurs par théorie des jeux.

    Regret Matching suppose que le gain de chaque action est observé à chaque
    tour. L'algorithme DE fournit ces gains complets par évaluation des trois
    opérateurs sur les mêmes individus.
    """

    def __init__(self, n_actions, method="regret", learning_rate=0.15,
                 exploration_floor=0.03):
        self.n = int(n_actions)
        self.method = method
        self.eta = float(learning_rate)
        self.floor = float(exploration_floor)
        self.probabilities = np.ones(self.n, dtype=float) / self.n
        self.cumulative_regret = np.zeros(self.n, dtype=float)
        self.last_payoffs = np.zeros(self.n, dtype=float)

    def _project(self, p):
        p = np.maximum(np.asarray(p, dtype=float), 0.0)
        p = p / p.sum() if p.sum() > 1e-15 else np.ones(self.n) / self.n
        floor = min(self.floor, 0.99 / self.n)
        p = floor + (1.0 - self.n * floor) * p
        return p / p.sum()

    def update(self, payoffs):
        u = np.asarray(payoffs, dtype=float)
        if u.shape != (self.n,) or not np.all(np.isfinite(u)):
            raise ValueError("Vecteur de gains invalide.")
        self.last_payoffs = u.copy()

        if self.method == "uniform":
            self.probabilities = np.ones(self.n) / self.n
        elif self.method == "replicator":
            average = float(np.dot(self.probabilities, u))
            logits = np.log(np.maximum(self.probabilities, 1e-15))
            logits += self.eta * (u - average)
            logits -= np.max(logits)
            self.probabilities = self._project(np.exp(logits))
        elif self.method == "regret":
            average = float(np.dot(self.probabilities, u))
            self.cumulative_regret = np.maximum(
                0.0, self.cumulative_regret + u - average
            )
            p = (self.cumulative_regret / self.cumulative_regret.sum()
                 if self.cumulative_regret.sum() > 1e-15
                 else np.ones(self.n) / self.n)
            self.probabilities = self._project(p)
        else:
            raise ValueError(f"Méthode inconnue : {self.method}")
        return self.probabilities.copy()

    def sample(self, rng):
        return int(rng.choice(self.n, p=self.probabilities))


def normalize_minmax(values):
    values = np.asarray(values, dtype=float)
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def entropy(probabilities):
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

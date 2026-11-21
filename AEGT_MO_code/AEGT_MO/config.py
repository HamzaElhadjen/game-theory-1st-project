from dataclasses import dataclass

@dataclass
class DEConfig:
    population_size: int = 40
    differential_weight: float = 0.5
    crossover_rate: float = 0.9
    max_evaluations: int = 10000
    lower_bound: float = -5.12
    upper_bound: float = 5.12
    seed: int = 42

@dataclass
class GameConfig:
    method: str = "regret"  # uniform, replicator, regret
    learning_rate: float = 0.15
    congestion: float = 0.15
    lambda_max: float = 0.5
    gamma: float = 1.0
    ema_beta: float = 0.7
    use_memory: bool = True
    use_dynamic_diversity: bool = True
    use_stagnation: bool = False
    stagnation_window: int = 12
    stagnation_boost: float = 0.25
    exploration_floor: float = 0.03
    use_congestion: bool = True
    use_diversity: bool = True

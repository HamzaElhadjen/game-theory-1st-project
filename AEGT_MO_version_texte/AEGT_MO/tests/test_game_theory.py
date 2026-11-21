import numpy as np
from game_theory import OperatorGame, entropy, normalize_minmax

def test_probabilities_sum_to_one():
    for method in ["uniform", "replicator", "regret"]:
        game = OperatorGame(3, method=method)
        for gains in ([0.2, 0.5, 0.1], [1.0, 1.0, 1.0], [-1.0, 0.0, 2.0]):
            p = game.update(gains)
            assert np.all(p >= 0)
            assert abs(p.sum() - 1.0) < 1e-10

def test_normalization_constant():
    assert np.allclose(normalize_minmax([2, 2, 2]), [0, 0, 0])

def test_entropy_uniform():
    assert entropy([1/3, 1/3, 1/3]) > 1.0

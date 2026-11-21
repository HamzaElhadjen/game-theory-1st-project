from problems import rastrigin, rosenbrock

def test_known_optima():
    assert abs(rastrigin([0, 0, 0])) < 1e-12
    assert abs(rosenbrock([1, 1, 1])) < 1e-12

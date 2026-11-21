import numpy as np

def rastrigin(x):
    x = np.asarray(x, dtype=float)
    d = x.size
    return float(10.0 * d + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))

def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float((1.0 - x[0]) ** 2)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

BENCHMARKS = {
    "rastrigin": (rastrigin, -5.12, 5.12, 0.0),
    "rosenbrock": (rosenbrock, -5.0, 10.0, 0.0),
}

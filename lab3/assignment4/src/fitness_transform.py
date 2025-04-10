import numpy as np
from numpy.typing import NDArray


def linear_baseline(fitness: NDArray[np.float64]) -> NDArray[np.float64]:
	baseline = max(fitness)
	return 2 - fitness / baseline


def sqrt_linear_baseline(fitness: NDArray[np.float64]) -> NDArray[np.float64]:
	baseline = max(fitness)
	return np.sqrt(2 - fitness / baseline)


def exponential_baseline(fitness: NDArray[np.float64]) -> NDArray[np.float64]:
	baseline = max(fitness)
	return 2 - (fitness / baseline) ** 2


def inverse_sqrt(fitness: NDArray[np.float64]) -> NDArray[np.float64]:
	return np.sqrt(1 / fitness)

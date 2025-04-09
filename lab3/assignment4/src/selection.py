import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable


class SelectionBase:
	fitness: NDArray[np.float64]
	transform: Callable[[NDArray[np.float64]], NDArray[np.float64]]
	size: int

	def __init__(self, fitness: NDArray[np.float64], transform: Callable[[NDArray[np.float64]], NDArray[np.float64]]):
		self.fitness = fitness
		self.transform = transform
		self.size = len(fitness)

	def __call__(self) -> tuple[int, int]:
		raise NotImplementedError('Subclasses must implement __call__ method')


class RouletteWheel(SelectionBase):
	prob: NDArray[np.float64]

	def __init__(
		self,
		fitness: NDArray[np.float64],
		transform: Callable[[NDArray[np.float64]], NDArray[np.float64]],
	) -> None:
		super().__init__(fitness, transform)
		self.prob = transform(fitness)
		self.prob /= self.prob.sum()

	def __call__(self) -> tuple[int, int]:
		a, b = np.random.choice(self.size, size=2, p=self.prob)
		return a, b


class Tournament(SelectionBase):
	tournament_size: int

	def __init__(
		self,
		fitness: NDArray[np.float64],
		transform: Callable[[NDArray[np.float64]], NDArray[np.float64]],
		tournament_size: int = 2,
	) -> None:
		super().__init__(fitness, transform)
		self.tournament_size = tournament_size

	def __call__(self) -> tuple[int, int]:
		stage_1 = np.random.choice(self.size, size=self.tournament_size)
		a = stage_1[np.argmax(self.fitness[stage_1])]
		b = stage_1[np.argmin(self.fitness[stage_1])]
		return a, b

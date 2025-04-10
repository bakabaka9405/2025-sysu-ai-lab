import numpy as np
from numpy.typing import NDArray


class Selector:
	size: int

	def __init__(
		self,
		size: int,
		**_kwargs,
	) -> None:
		self.size = size

	def __call__(self, _count: int) -> NDArray:
		raise NotImplementedError('Subclasses must implement __call__ method')


class RouletteWheel(Selector):
	prob: NDArray[np.float64]

	def __init__(
		self,
		size: int,
		**kwargs,
	) -> None:
		super().__init__(size)
		self.prob = kwargs['transform']((kwargs['fitness']))
		self.prob /= self.prob.sum()

	def __call__(self, count: int) -> NDArray:
		return np.random.choice(self.size, size=(count, 2), p=self.prob)


class Tournament(Selector):
	tournament_size: int

	def __init__(
		self,
		size: int,
		**kwargs,
	) -> None:
		super().__init__(size)
		self.tournament_size = kwargs['tournament_size']

	def __call__(self, count: int) -> NDArray:
		"""
		针对性优化：因为每轮结束都要对 Individual 基于 fitness 排序，因此下标小的 Individual 适应度必定更高。
		第一轮的 fitness 是随机的，但是第一轮就无所谓了。
		可以省下存 fitness 的空间
		"""
		rnd = np.random.randint(0, self.size, size=(count * 2, self.tournament_size))
		return np.min(rnd, axis=1).reshape(count, 2)


if __name__ == '__main__':
	tournament = Tournament(10, tournament_size=5)
	print(tournament(5))

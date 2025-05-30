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
		"""
		单次调用 choice 的复杂度是线性的，因此选择一次性产生多个 choice
		理论上复杂度不超过 n log n，不知道 numpy 怎么实现的，按理说自己写可以达到线性？
		"""
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
		可以省下存 fitness 的空间
		"""
		rnd = np.random.randint(0, self.size, size=(count * 2, self.tournament_size))
		return np.min(rnd, axis=1).reshape(count, 2)


if __name__ == '__main__':
	tournament = Tournament(10, tournament_size=5)
	print(tournament(5))

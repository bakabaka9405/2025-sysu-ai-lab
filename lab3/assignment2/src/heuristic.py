from collections.abc import Callable
from typing import Union


def multiple_aug(ratio: float):
	def aug(x: int) -> float:
		return ratio * x

	return aug


def power_aug(ratio: float):
	def aug(x: int) -> float:
		return ratio**x

	return aug


def manhattan(aug: Callable[[int], Union[int, float]]):
	"""
	曼哈顿距离
	"""

	def h(s: list[int]) -> int:
		return int(aug(sum([abs((s[i] >> 2) - (i >> 2)) + abs((s[i] & 3) - (i & 3)) for i in range(16) if s[i] != 15])))

	return h


def weighted_manhattan(aug: Callable[[int], Union[int, float]]):
	"""
	带权曼哈顿距离
	理论上能刺激 A* 算法按 1,2,3... 的顺序还原 puzzle，从而提高在宽松步数限制下的计算效率
	"""

	def h(s: list[int]) -> int:
		return int(aug(sum([(16 - s[i]) * (abs((s[i] >> 2) - (i >> 2)) + abs((s[i] & 3) - (i & 3))) for i in range(16) if s[i] != 15])))

	return h


def misplaced(aug: Callable[[int], Union[int, float]]):
	"""
	错位数
	"""

	def h(s: list[int]) -> int:
		return int(aug(sum([1 for i in range(16) if s[i] != i and s[i] != 15])))

	return h


def weighted_misplaced(aug: Callable[[int], Union[int, float]]):
	"""
	带权错位数，原理相同
	"""

	def h(s: list[int]) -> int:
		return int(aug(sum([s[i] for i in range(16) if s[i] != i and s[i] != 15])))

	return h

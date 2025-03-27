def manhattan(ratio: float):
	"""
	曼哈顿距离
	"""
	def h(s: list[int]) -> int:
		return int(sum([abs((s[i] >> 2) - (i >> 2)) + abs((s[i] & 3) - (i & 3)) for i in range(16) if s[i] != 15]) * ratio)

	return h


def weighted_manhattan(ratio: float):
	"""
	带权曼哈顿距离
	理论上能刺激 A* 算法按 1,2,3... 的顺序还原 puzzle，从而提高在宽松步数限制下的计算效率
	"""
	def h(s: list[int]) -> int:
		return int(sum([(16 - s[i]) * (abs((s[i] >> 2) - (i >> 2)) + abs((s[i] & 3) - (i & 3))) for i in range(16) if s[i] != 15]) * ratio)

	return h


def misplaced(ratio: float):
	"""
	错位数
	"""
	def h(s: list[int]) -> int:
		return int(sum([1 for i in range(16) if s[i] != i and s[i] != 15]) * ratio)

	return h


def weighted_misplaced(ratio: float):
	"""
	带权错位数，原理相同
	"""
	def h(s: list[int]) -> int:
		return int(sum([s[i] for i in range(16) if s[i] != i and s[i] != 15]) * ratio)

	return h

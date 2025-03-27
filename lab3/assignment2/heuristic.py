def manhattan(ratio: float):
	def h(s: list[int]) -> int:
		return int(sum([abs((s[i] >> 2) - (i >> 2)) + abs((s[i] & 3) - (i & 3)) for i in range(16) if s[i] != 15]) * ratio)

	return h


def weighted_manhattan(ratio: float):
	def h(s: list[int]) -> int:
		return int(sum([(16 - s[i]) * (abs((s[i] >> 2) - (i >> 2)) + abs((s[i] & 3) - (i & 3))) for i in range(16) if s[i] != 15]) * ratio)

	return h


def misplaced(ratio: float):
	def h(s: list[int]) -> int:
		return int(sum([1 for i in range(16) if s[i] != i and s[i] != 15]) * ratio)

	return h


def weighted_misplaced(ratio: float):
	def h(s: list[int]) -> int:
		return int(sum([s[i] for i in range(16) if s[i] != i and s[i] != 15]) * ratio)

	return h

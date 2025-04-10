import numpy as np
from numpy.typing import NDArray


def swap_mutation(gene: NDArray) -> None:
	size = len(gene)
	p1, p2 = np.random.choice(size, size=2, replace=False)
	gene[p1], gene[p2] = gene[p2], gene[p1]


def adjacent_swap_mutation(gene: NDArray) -> None:
	size = len(gene)
	p = np.random.randint(0, size - 1)
	gene[p], gene[p + 1] = gene[p + 1], gene[p]


def range_swap_mutation(gene: NDArray) -> None:
	size = len(gene)
	p1, p2, p3, p4 = sorted(np.random.choice(size, size=4, replace=False))
	rg = min(p2 - p1, p4 - p3)
	p2, p4 = p1 + rg, p3 + rg
	gene[p1:p2], gene[p3:p4] = gene[p3:p4].copy(), gene[p1:p2].copy()


def reverse_mutation(gene: NDArray) -> None:
	size = len(gene)
	p1, p2 = np.random.choice(size, size=2, replace=False)
	gene[p1:p2] = gene[p1:p2][::-1]


def rorate_mutation(gene: NDArray) -> None:
	p1, p2 = sorted(np.random.choice(len(gene), size=2, replace=False))
	gene[p1:p2] = np.roll(gene[p1:p2], shift=1)


def shuffle_mutation(gene: NDArray) -> None:
	size = len(gene)
	p1, p2 = sorted(np.random.choice(size, size=2, replace=False))
	gene[p1:p2] = np.random.permutation(gene[p1:p2])


def random_mutation(gene: NDArray) -> None:
	mut = [swap_mutation, adjacent_swap_mutation, range_swap_mutation, reverse_mutation, rorate_mutation, shuffle_mutation]
	mut[np.random.randint(0, 6)](gene)


def random_mutation_1(gene: NDArray) -> None:
	mut = [swap_mutation, adjacent_swap_mutation, range_swap_mutation, rorate_mutation]
	mut[np.random.randint(0, 4)](gene)


if __name__ == '__main__':
	a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
	range_swap_mutation(a)
	print(a)

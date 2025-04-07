import numpy as np
from numpy.typing import NDArray


def swap_mutation(gene: NDArray) -> None:
	size = len(gene)
	p1, p2 = np.random.choice(size, size=2, replace=False)
	gene[p1], gene[p2] = gene[p2], gene[p1]


def reverse_mutation(gene: NDArray) -> None:
	size = len(gene)
	p1, p2 = np.random.choice(size, size=2, replace=False)
	gene[p1:p2] = gene[p1:p2][::-1]


def shuffle_mutation(gene: NDArray) -> None:
	size = len(gene)
	p1, p2 = np.random.choice(size, size=2, replace=False)
	gene[p1:p2] = np.random.permutation(gene[p1:p2])


def random_mutation(gene: NDArray) -> None:
	mut = [swap_mutation, reverse_mutation, shuffle_mutation]
	mut[np.random.randint(0, 3)](gene)

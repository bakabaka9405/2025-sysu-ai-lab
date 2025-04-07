import numpy as np
from numpy.typing import NDArray


def partial_mapping_crossover(a: NDArray, b: NDArray) -> tuple[NDArray, NDArray]:
	size = len(a)
	cxpoint1, cxpoint2 = sorted(np.random.choice(size, 2, replace=False))

	child1, child2 = np.zeros(size, dtype=int), np.zeros(size, dtype=int)
	child1[cxpoint1:cxpoint2] = b[cxpoint1:cxpoint2]
	child2[cxpoint1:cxpoint2] = a[cxpoint1:cxpoint2]

	mapping1 = {j: i for i, j in zip(a[cxpoint1:cxpoint2], b[cxpoint1:cxpoint2])}
	mapping2 = {i: j for i, j in zip(a[cxpoint1:cxpoint2], b[cxpoint1:cxpoint2])}

	s1 = set(child1[cxpoint1:cxpoint2])
	s2 = set(child2[cxpoint1:cxpoint2])

	for i in range(cxpoint1):
		j = a[i]
		while j in s1:
			j = mapping1[j]
		child1[i] = j

	for i in range(cxpoint2, size):
		j = a[i]
		while j in s1:
			j = mapping1[j]
		child1[i] = j

	for i in range(cxpoint1):
		j = b[i]
		while j in s2:
			j = mapping2[j]
		child2[i] = j

	for i in range(cxpoint2, size):
		j = b[i]
		while j in s2:
			j = mapping2[j]
		child2[i] = j

	return child1, child2


def order_crossover(a: NDArray, b: NDArray) -> tuple[NDArray, NDArray]:
	size = len(a)
	cxpoint1, cxpoint2 = sorted(np.random.choice(size, 2, replace=False))
	child1, child2 = np.zeros(size, dtype=int), np.zeros(size, dtype=int)

	child1[cxpoint1:cxpoint2] = a[cxpoint1:cxpoint2]
	child2[cxpoint1:cxpoint2] = b[cxpoint1:cxpoint2]

	s1 = set(child1[cxpoint1:cxpoint2])
	s2 = set(child2[cxpoint1:cxpoint2])

	ra = np.array([i for i in b if i not in s1])
	rb = np.array([i for i in a if i not in s2])

	child1[:cxpoint1] = ra[:cxpoint1]
	child1[cxpoint2:] = ra[cxpoint1:]
	child2[:cxpoint1] = rb[:cxpoint1]
	child2[cxpoint2:] = rb[cxpoint1:]

	return child1, child2


def position_based_crossover(a: NDArray, b: NDArray) -> tuple[NDArray, NDArray]:
	size = len(a)
	sample = np.random.choice(range(size), np.random.randint(1, size), replace=False)
	child1, child2, rs = np.zeros(size, dtype=int), np.zeros(size, dtype=int), np.full(size, 1)
	child1[sample] = a[sample]
	child2[sample] = b[sample]
	rs[sample] = 0
	rs = np.where(rs == 1)[0]
	
	s1 = set(a[i] for i in sample)
	s2 = set(b[i] for i in sample)

	ra = np.array([i for i in b if i not in s1])
	rb = np.array([i for i in a if i not in s2])

	for i, j, k in zip(rs, ra, rb):
		child1[i] = j
		child2[i] = k

	return child1, child2

if __name__=='__main__':
	a=np.array([1,2,3,4,5])
	b=np.array([6,7,8,9,10])
	print(position_based_crossover(a,b))
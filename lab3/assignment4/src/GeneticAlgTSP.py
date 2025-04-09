from typing import Union
from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import multiprocessing as mp
import time
import config
import mutation
import crossover
import selection
import fitness_transform
from util import euclidean, logger, Gene, Individual
import gc


def crossover_worker(
	population: list[Individual],
	selector: selection.SelectionBase,
	crossover: Callable[[Gene, Gene], tuple[Gene, Gene]],
	distance: Callable[[Gene], float],
	size: int,
) -> list[Individual]:
	new_population: list[Individual] = []
	dis_threshold = population[-1][1]
	mutation_func = mutation.__dict__[config.mutation_policy]
	for _ in range(size):
		p1, p2 = selector()
		parent1, parent2 = population[p1][0], population[p2][0]
		if np.random.rand() < config.mutation_prob:
			mutation_func(parent1)
		if np.random.rand() < config.mutation_prob:
			mutation_func(parent2)
		if (parent1 == parent2).all() and np.random.rand() < config.homozygous_lethality:
			continue
		child1, child2 = crossover(parent1, parent2)
		if np.random.rand() < config.mutation_prob:
			mutation_func(child1)
		if np.random.rand() < config.mutation_prob:
			mutation_func(child2)
		dis1, dis2 = distance(child1), distance(child2)
		if dis1 < dis_threshold:
			new_population.append((child1, dis1))
		if dis2 < dis_threshold:
			new_population.append((child2, dis2))
	return new_population


class GeneticAlgTSP:
	cities: NDArray[np.float64]
	population: list[Individual]
	crossover_fn: Callable
	fitness_transform_fn: Callable
	selection_fn: Callable

	def __init__(self, filename: str):
		# 读取城市坐标
		lines = open(filename).readlines()
		start_line = lines.index('NODE_COORD_SECTION\n') + 1

		try:
			end_line = lines.index('EOF\n')
		except ValueError:
			end_line = len(lines)

		logger.debug(f'城市点数 = {end_line - start_line}')

		self.cities = np.array([list(map(float, reversed(line.split()[1:]))) for line in lines[start_line:end_line]])

		# 对基因进行优化：观察到 TSP 的最优解与起点无关，因此可以将起点固定为最后一个城市。
		# 使得基因长度 -1，当然后续的计算代码也要相应修改。
		# 生成初始种群
		genes = [np.random.permutation(len(self.cities) - 1) for _ in range(config.initial_population_size)]
		self.population = [(g, self.distance(g)) for g in genes]
		self.crossover_fn = crossover.__dict__[config.crossover_policy]
		self.fitness_transform_fn = fitness_transform.__dict__[config.fitness_transform_policy]
		self.selection_fn = selection.__dict__[config.selection_policy]

	def iterate(self, epochs: int) -> list[int]:
		start = time.time()
		for i in range(epochs):
			logger.debug(f'epoch {i+1}/{epochs} 开始')
			self.next_generation()
			if config.output_path_dir:
				self.plot(self.population[0][0], i + 1)
			logger.info(f'epoch {i+1}/{epochs} 结束。最短距离 = {self.population[0][1]:.2f}')
			gc.collect()
		logger.info(f'总耗时 = {time.time()-start:.2f}秒')
		return self.population[0][0].tolist() + [(len(self.cities) - 1)]

	def distance(self, path: Union[list[int], NDArray]) -> float:
		return (
			sum(
				euclidean(
					self.cities[path[i]],
					self.cities[path[i + 1]],
				)
				for i in range(len(self.cities) - 2)
			)
			+ euclidean(self.cities[path[-1]], self.cities[-1])
			+ euclidean(self.cities[-1], self.cities[path[0]])
		)

	def next_generation(self) -> None:
		new_population = self.population[:]
		size = (config.cross_per_epoch + config.num_worker - 1) // config.num_worker
		logger.debug(f'每个进程处理 {size} 个交叉操作')
		pool = mp.Pool(config.num_worker)
		logger.debug('正在创建进程...')

		fitness = np.array([p[1] for p in self.population])
		transform_fn = fitness_transform.__dict__[config.fitness_transform_policy]
		selector: Callable[[], tuple[int, int]] = selection.__dict__[config.selection_policy](fitness, transform_fn)
		results = [
			pool.apply_async(
				crossover_worker,
				(self.population, selector, self.crossover_fn, self.distance, size),
			)
			for _ in range(config.num_worker)
		]
		logger.debug('等待结果中...')
		pool.close()
		pool.join()
		try:
			for result in results:
				new_population += result.get()
		except Exception as e:
			print('Error:', e)
			exit(-1)
		logger.debug('crossover 结束')
		logger.debug(f'新种群大小（不算上一代） = {len(new_population) - len(self.population)}')
		self.population = sorted(
			new_population,
			key=lambda x: x[1],
		)[: config.maximum_population_size]

	def plot(self, path: Union[list[int], NDArray], epoch: int):
		"""
		绘制路径图像并保存到指定目录。
		:param path: 城市路径
		:param epoch: 当前迭代次数，用于命名保存的图像文件
		"""

		fig, ax = plt.subplots(figsize=(10, 8))

		# 绘制所有城市点
		x = self.cities[:, 0]
		y = self.cities[:, 1]
		ax.scatter(x, y, c='blue', marker='o', alpha=0.7)

		# 绘制路径连线
		for i in range(len(path) - 1):
			ax.plot([self.cities[path[i]][0], self.cities[path[i + 1]][0]], [self.cities[path[i]][1], self.cities[path[i + 1]][1]], 'r-', alpha=0.5)

		# 补全最后两根连线
		ax.plot([self.cities[path[-1]][0], self.cities[-1][0]], [self.cities[path[-1]][1], self.cities[-1][1]], 'r-', alpha=0.5)
		ax.plot([self.cities[path[0]][0], self.cities[-1][0]], [self.cities[path[0]][1], self.cities[-1][1]], 'r-', alpha=0.5)

		# 添加标题和标签
		ax.set_title(f'TSP路径 (总距离: {self.distance(path):.2f})')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.grid(True, linestyle='--', alpha=0.7)

		fig.tight_layout()
		fig.savefig(f'{config.output_path_dir}/epoch_{epoch}.png')
		plt.close(fig)

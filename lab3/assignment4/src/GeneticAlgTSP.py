from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray
import multiprocessing as mp
import time
import config
import crossover
import selection
from selection import Selector
import fitness_transform
from util import logger, Individual, path_distance, plot_path, log_basic_info
from worker import parallel_crossover_worker


class GeneticAlgTSP:
	cities: NDArray[np.float64]
	population: list[Individual]
	crossover_fn: Callable
	fitness_transform_fn: Callable
	selection_fn: Callable
	task_cross_count: int
	task_queues: list[mp.Queue]
	result_queue: mp.Queue
	# shared_population: ListProxy[Individual]
	workers: list[mp.Process]

	def __init__(self, filename: str):
		log_basic_info()
		# 读取城市坐标
		lines = open(filename).readlines()
		start_line = lines.index('NODE_COORD_SECTION\n') + 1

		try:
			end_line = lines.index('EOF\n')
		except ValueError:
			end_line = len(lines)

		logger.debug(f'城市点数 = {end_line - start_line}')

		self.cities = np.array([list(map(float, reversed(line.split()[1:]))) for line in lines[start_line:end_line]])

		# 对基因进行优化：观察到 TSP 的最优解与起点无关，因此可以将固定某一个城市为路径的起点或终点
		# 使得基因长度 -1，当然后续的计算代码也要相应修改。
		# 生成初始种群
		logger.debug('正在生成初始种群...')
		genes = [np.random.permutation(len(self.cities) - 1) for _ in range(config.initial_population_size)]
		logger.debug('正在计算初始种群的适应度...')
		self.population = [(g, path_distance(self.cities, g)) for g in genes]
		self.crossover_fn = crossover.__dict__[config.crossover_policy]
		self.fitness_transform_fn = fitness_transform.__dict__[config.fitness_transform_policy]
		self.selection_fn = selection.__dict__[config.selection_policy]

		# 初始化多进程变量
		self.task_queues = []
		self.result_queue = mp.Queue()
		self.workers = []
		self.task_cross_count = (config.cross_per_epoch + config.num_worker - 1) // config.num_worker
		self.manager = mp.Manager()
		self.shared_population = self.manager.list(self.population)

		logger.debug(f'每个进程处理 {self.task_cross_count} 个交叉操作')

	def start_workers(self):
		logger.debug(f'正在创建 {config.num_worker} 个工作进程...')

		for i in range(config.num_worker):
			task_queue = mp.Queue()
			self.task_queues.append(task_queue)

			worker = mp.Process(
				target=parallel_crossover_worker,
				args=(
					i,
					np.random.randint(0, (1 << 31) - 1),
					task_queue,
					self.result_queue,
					self.shared_population,
					self.cities,
					self.task_cross_count,
				),
				daemon=True,
			)

			worker.start()
			self.workers.append(worker)

		logger.debug('所有工作进程已启动')

	def stop_workers(self):
		logger.debug('正在停止工作进程...')

		for task_queue in self.task_queues:
			task_queue.put(None)

		for worker in self.workers:
			worker.join(timeout=1.0)

		logger.debug('所有工作进程已停止')

	def iterate(self, epochs: int) -> list[int]:
		start = time.time()
		last_best_distance = float('inf')
		parent_mutation_prob = config.base_mutation_prob
		self.start_workers()
		for i in range(epochs):
			epoch_start = time.time()
			logger.debug(f'epoch {i+1}/{epochs} 开始')

			self.next_generation(parent_mutation_prob)

			if config.output_path_dir:
				plot_path(self.cities, self.population[0][0], i + 1)
			delta = 0 if last_best_distance == float('inf') else last_best_distance - self.population[0][1]
			if delta == 0:
				parent_mutation_prob += config.mutation_punishment
			else:
				parent_mutation_prob = max(min(parent_mutation_prob, 1.0) / config.mutation_recovery, config.base_mutation_prob)
			logger.info(f'epoch {i+1}/{epochs} 结束，用时 {time.time()-epoch_start:.2f} 秒，最短距离 = {self.population[0][1]:.2f} (-{delta:.2f})')
			last_best_distance = self.population[0][1]

		logger.info(f'总耗时 = {time.time()-start:.2f} 秒')
		self.stop_workers()

		return self.population[0][0].tolist() + [len(self.cities) - 1]

	def next_generation(self, parent_mutation_prob: float) -> None:
		new_population = self.population[:]

		# 准备选择器
		fitness = np.array([p[1] for p in self.population])
		selector: Selector = self.selection_fn(
			len(fitness),
			fitness=fitness,
			transform=self.fitness_transform_fn,
			tournament_size=config.tournament_size,
		)

		# 更新共享内存中的种群数据
		self.shared_population[:] = self.population

		# 分发任务给所有工作进程
		should_copy = len(fitness) * (len(self.cities) - 1) * len(self.workers) <= config.worker_data_copy_threshold
		if should_copy:
			logger.debug('每个进程将复制数据到独立内存')
		else:
			logger.debug('每个进程将使用共享内存')
		logger.debug(f'父辈变异概率 = {parent_mutation_prob:.2f}')
		logger.debug('分发任务给工作进程...')
		for task_queue in self.task_queues:
			task_queue.put(
				(
					selector,
					parent_mutation_prob,
					should_copy,
				)
			)

		# 收集结果
		logger.debug('等待结果中...')
		results_count = 0
		try:
			while results_count < config.num_worker:
				result = self.result_queue.get(timeout=3600)
				new_population += result
				results_count += 1
		except Exception as e:
			logger.error(f'收集结果出错: {e}')

		logger.debug('交叉操作结束')
		logger.debug(f'新种群大小（不算上一代） = {len(new_population) - len(self.population)}')

		# 更新种群
		self.population = sorted(
			new_population,
			key=lambda x: x[1],
		)[: config.maximum_population_size]

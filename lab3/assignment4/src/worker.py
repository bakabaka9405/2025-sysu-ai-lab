from typing import Optional
import numpy as np
from numpy.typing import NDArray
import multiprocessing as mp

import config
import mutation
import crossover
from util import logger, path_distance, Individual
from selection import Selector


def parallel_crossover_worker(
	pid: int, task_queue: mp.Queue, result_queue: mp.Queue, population_array, cities: NDArray[np.float64], task_cross_count: int
):
	"""
	持续运行的工作进程，通过队列接收任务并返回结果

	Args:
		id: 进程 ID
	    task_queue: 任务队列，接收开始信号和参数
	    result_queue: 结果队列，发送生成的新个体
	    population_array: 共享内存中的种群数据
	    cities: 城市坐标数据（直接传递，非共享内存）
	    crossover_fn: 使用的交叉函数名
	    mutation_policy: 变异策略
	"""
	crossover_function = crossover.__dict__[config.crossover_policy]
	mutation_function = mutation.__dict__[config.mutation_policy]

	buffer: list[Optional[Individual]] = [None] * (task_cross_count * 2)
	while True:
		try:
			# 等待任务
			selector: Selector
			mutation_prob: float
			task = task_queue.get()
			if task is None:  # 退出信号
				break

			selector, mutation_prob, should_copy = task
			# logger.debug(f'Worker {pid} 接收到任务，复制策略为 {should_copy}')
			buffer_size = 0

			population_data = population_array[:] if should_copy else population_array
			dis_threshold = population_data[-1][1]
			sel = selector(task_cross_count)
			for p1, p2 in sel:
				# 复制父代个体
				parent1, parent2 = np.copy(population_data[p1][0]), np.copy(population_data[p2][0])

				# 父代变异
				if np.random.rand() < mutation_prob:
					mutation_function(parent1)
				if np.random.rand() < mutation_prob:
					mutation_function(parent2)

				# 纯合子致死检查
				if (parent1 == parent2).all() and np.random.rand() < config.homozygous_lethality:
					continue

				# 交叉产生子代
				child1, child2 = crossover_function(parent1, parent2)

				# 子代变异
				if np.random.rand() < mutation_prob:
					mutation_function(child1)
				if np.random.rand() < mutation_prob:
					mutation_function(child2)

				# 计算适应度并添加到新种群
				dis1, dis2 = path_distance(cities, child1), path_distance(cities, child2)

				if dis1 < dis_threshold:
					buffer[buffer_size] = (child1, dis1)
					buffer_size += 1
				if dis2 < dis_threshold:
					buffer[buffer_size] = (child1, dis1)
					buffer_size += 1

			# 将结果放入结果队列
			result_queue.put(buffer[:buffer_size])

		except Exception as e:
			logger.error(f'Worker进程出错: {e}')
			result_queue.put([])

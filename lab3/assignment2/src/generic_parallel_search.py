from typing import Optional, Union, cast
from collections.abc import Callable
from queue import Queue
import multiprocessing
from threading import Event
import time
import config
from util import State, FlatState, state_solvable, logger
import heuristic

WorkerReturnType = Optional[tuple[list[State], list[int]]]

WorkerFunc = Callable[[FlatState, Callable[[FlatState], int], Event, str], WorkerReturnType]

HandlerReturnType = tuple[str, float, tuple[list[State], list[int]], float]
"""
工作函数的管理器返回值类型
包含启发式函数名称、增强比例、结果和耗时
"""


def run_handler(
	s: FlatState,
	worker: WorkerFunc,
	h_name: str,
	h_aug_type: str,
	h_aug_ratio: float,
	result_queue: Queue[HandlerReturnType],
	stop_event: Event,
) -> None:
	"""
	运行工作函数，执行 worker 进行 A* 或 IDA* 算法的搜索

	:param s: 初始状态
	:param worker: 工作函数，可以是 A* 或 IDA* 算法的工作函数
	:param h_name: 启发式函数名称
	:param h_aug_type: 启发式函数增强类型
	:param h_aug_ratio: 启发式函数增强比例
	:param result_queue: 输出队列
	:param stop_event: 停止事件
	"""

	task_name = f'{h_name}_{h_aug_type}_{h_aug_ratio}'
	logger.debug(f'进程 {task_name} 已启动')

	h_func = heuristic.__dict__[h_name]((heuristic.multiple_aug if h_aug_type == 'mul' else heuristic.power_aug)(h_aug_ratio))

	start_time = time.time()
	result = worker(s, h_func, stop_event, task_name)
	end_time = time.time()

	if result is not None:
		elapsed = end_time - start_time
		logger.info(f'[{task_name}] 找到解决方案，用时 {elapsed:.2f} 秒')

		if not stop_event.is_set():
			result_queue.put((h_name, h_aug_ratio, result, elapsed))
		stop_event.set()
	else:
		if stop_event.is_set():
			logger.debug(f'[{task_name}] 搜索被终止')
		else:
			logger.debug(f'[{task_name}] 未找到解决方案')

	logger.debug(f'[{task_name}] 进程结束')


def generic_parallel_search(
	state: Union[State, FlatState], worker: WorkerFunc, name: str = 'MAIN'
) -> Union[list[State], list[int], tuple[list[State], list[int]], None]:
	"""
	并行搜索的入口函数，根据 `config.h_list` 中的启发式函数列表创建多个进程并运行 A*/IDA* 算法
	:param state: 初始状态，可以是 4x4 矩阵或扁平化的列表
	:param worker: 工作函数，可以是 A* 或 IDA* 算法的工作函数
	:return: 如果找到解决方案，返回状态列表或操作列表（根据 `config.a_star_return_type`），否则返回 None
	"""
	s: FlatState

	if len(state) != 16:
		s = cast(FlatState, [(i - 1) & 0xF for j in state for i in cast(list[int], j)])
	else:
		s = cast(FlatState, state)

	if not state_solvable(s):
		logger.error(f'[{name}] 初始状态不可解')
		return None

	# 准备进程通信组件
	manager = multiprocessing.Manager()
	result_queue: Queue[HandlerReturnType] = manager.Queue()
	stop_event = manager.Event()

	# 创建进程
	logger.info(f'[{name}] 开始创建进程')
	processes = []
	for h_name, h_type, h_aug in config.h_list:
		p = multiprocessing.Process(target=run_handler, args=(s, worker, h_name, h_type, h_aug, result_queue, stop_event))
		processes.append(p)
		p.start()

	# 等待任意一个进程找到结果
	logger.info(f'[{name}] 进程创建完成，等待结果...')
	solution = None
	try:
		first_result = result_queue.get()
		h_name, h_aug, solution, elapsed = first_result
		logger.info(f'[{name}] 最先找到解决方案的是: {h_name}_{h_aug}')
		logger.info(f'[{name}] 耗时: {elapsed:.2f} 秒')
		logger.info(f'[{name}] 解的长度: {len(solution[1])} 步')
	except Exception as e:
		logger.error(f'[{name}] 获取结果时发生错误: {str(e)}')
		logger.error(f'[{name}] 没有找到解决方案')

	# 通知所有进程停止
	stop_event.set()

	# 等待所有进程结束
	for p in processes:
		p.join(timeout=5.0)
		if p.is_alive():
			p.terminate()

	logger.info(f'[{name}] 所有进程已结束')

	assert solution is not None
	if config.search_return_type == 'state_list':
		return solution[0]
	elif config.search_return_type == 'operation_list':
		return solution[1]
	else:
		return solution


def create_searcher(
	worker: WorkerFunc,
	name: str = 'MAIN',
) -> Callable[[Union[State, FlatState]], Union[list[State], list[int], tuple[list[State], list[int]], None]]:
	"""
	创建搜索器函数，便于在其他模块中调用
	:param worker: 工作函数，可以是 A* 或 IDA* 算法的工作函数
	:return: 搜索器函数
	"""

	def searcher(state: Union[State, FlatState]):
		return generic_parallel_search(state, worker, name)

	return searcher

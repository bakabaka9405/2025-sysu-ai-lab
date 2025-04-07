from typing import Union, Optional, cast
from collections.abc import Callable
from queue import PriorityQueue, Queue
import multiprocessing
from threading import Event
import time

import heuristic
from util import logger
import config

adj: list[list[int]] = [
	[1, 4],
	[0, 2, 5],
	[1, 3, 6],
	[2, 7],
	[0, 5, 8],
	[1, 4, 6, 9],
	[2, 5, 7, 10],
	[3, 6, 11],
	[4, 9, 12],
	[5, 8, 10, 13],
	[6, 9, 11, 14],
	[7, 10, 15],
	[8, 13],
	[9, 12, 14],
	[10, 13, 15],
	[11, 14],
]

State = list[list[int]]

FlatState = list[int]

CompressedState = int


class StateNode:
	state: CompressedState
	g: int
	h: int
	pos: int

	def __init__(self, state: CompressedState, g: int, h: int, pos: int):
		self.state = state
		self.g = g
		self.h = h
		self.pos = pos

	def __lt__(self, other: 'StateNode') -> bool:
		return self.g + self.h < other.g + other.h

	def wrap(self) -> tuple[CompressedState, int, int]:
		return self.state, self.g, self.pos


def compress_state(state: FlatState) -> CompressedState:
	"""
	将 1x16 列表转换为压缩状态

	:param state: 1x16 列表
	:return: 压缩状态
	"""
	return sum([state[i] << (i << 2) for i in range(16)])


def decompress_state(state: CompressedState) -> FlatState:
	"""
	将压缩状态转换为 1x16 列表

	:param state: 压缩状态
	:return: 1x16 列表
	"""
	return [(state >> (i << 2)) & 0b1111 for i in range(16)]


def a_star_worker(
	state: FlatState,
	h_func: Callable[[FlatState], int],
	stop_event: Optional[Event] = None,
	task_name: str = 'UNKNOWN',
) -> Optional[tuple[list[State], list[int]]]:
	"""
	A* 算法主函数，由 run_a_star_worker 调用

	:param state: 初始状态
	:param h_func: 启发式函数
	:param stop_event: 停止事件
	:param task_name: 任务名称
	:return: 状态路径和操作列表
	"""
	# 开闭列表
	pq: PriorityQueue[StateNode] = PriorityQueue()
	pq.put(StateNode(compress_state(state), 0, h_func(state), state.index(15)))
	dis: dict[int, int] = {compress_state(state): 0}

	# 路径追踪
	state_from: dict[int, tuple[int, int]] = {}

	# 计算目标状态
	target_state = compress_state(list(range(16)))

	cnt = 0  # 探索节点计数

	while not pq.empty():
		if stop_event is not None and stop_event.is_set():
			logger.debug(f'[{task_name}] 收到停止信号，终止搜索')
			return None

		cnt += 1
		if cnt % config.a_star_log_interval == 0:
			logger.debug(f'[{task_name}] 已探索 {cnt} 个节点，队列大小: {pq.qsize()}')

		if cnt % config.a_star_check_stop_interval == 0 and stop_event is not None and stop_event.is_set():
			return None

		u_state = pq.get()
		u, g, pos = u_state.wrap()

		u_flat = decompress_state(u)

		if u == target_state:
			logger.debug(f'[{task_name}] 找到目标状态，共探索 {cnt} 个节点，步骤数: {g}')
			return reconstruct_solution(u, state_from)

		if g >= config.a_star_max_steps != 0 and config.a_star_max_steps:
			continue

		for i in adj[pos]:
			u_flat[i], u_flat[pos] = u_flat[pos], u_flat[i]
			v = compress_state(u_flat)
			if g + 1 < dis.get(v, 1000):
				state_from[v] = u, u_flat[pos]
				pq.put(StateNode(v, g + 1, h_func(decompress_state(v)), i))
				dis[v] = g + 1
			u_flat[i], u_flat[pos] = u_flat[pos], u_flat[i]

	logger.debug(f'[{task_name}] 搜索完毕，未找到解决方案')
	return None


def reconstruct_solution(
	final_state: CompressedState,
	state_from: dict[int, tuple[int, int]],
) -> tuple[list[State], list[int]]:
	"""
	重建从初始状态到目标状态的路径

	:param final_state_id: 最终状态 ID
	:param state_from: 路径追踪表
	:return: 状态路径和操作列表
	"""

	state_path = [final_state]
	operation = []

	while state_path[-1] in state_from:
		operation.append(state_from[state_path[-1]][1])
		state_path.append(state_from[state_path[-1]][0])

	state_path.reverse()
	operation.reverse()

	sol_state: list[State] = []

	for state in state_path:
		s = list(map(lambda x: (x + 1) & 0xF, decompress_state(state)))
		matrix = [s[j * 4 : j * 4 + 4] for j in range(4)]
		sol_state.append(matrix)

	return sol_state, operation


WorkerReturnType = tuple[str, float, tuple[list[State], list[int]], float]


def run_a_star_worker(
	s: FlatState,
	h_name: str,
	h_aug_type: str,
	h_aug_ratio: float,
	result_queue: Queue[WorkerReturnType],
	stop_event: Event,
) -> None:
	"""
	进程工作函数

	:param s: 初始状态
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
	result = a_star_worker(s, h_func, stop_event, task_name)
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


def state_solvable(s: FlatState) -> bool:
	"""
	判断初始状态是否有解
	结论：将状态矩阵拍平后计算逆序对个数（空格不算在内），如果奇偶性不同于空格所在行数（从 0 算起）的奇偶性，则有解
	"""
	return (sum([1 for i in range(16) for j in range(i + 1, 16) if s[i] > s[j] and s[i] != 15]) + s.index(15) // 4) % 2 != 0


def A_star(state: Union[State, FlatState]) -> Union[list[State], list[int], tuple[list[State], list[int]], None]:
	"""
	A* 入口函数，根据 `config.h_list` 中的启发式函数列表创建多个进程并运行 A* 算法

	:param state: 初始状态，可以是 4x4 矩阵或扁平化的列表
	:return: 如果找到解决方案，返回状态列表或操作列表（根据 `config.a_star_return_type`），否则返回 None
	"""
	s: FlatState

	if len(state) != 16:
		s = cast(FlatState, [(i - 1) & 0xF for j in state for i in cast(list[int], j)])
	else:
		s = cast(FlatState, state)

	if not state_solvable(s):
		logger.error('[MAIN] 初始状态不可解')
		return None

	# 准备进程通信组件
	manager = multiprocessing.Manager()
	result_queue: Queue[WorkerReturnType] = manager.Queue()
	stop_event = manager.Event()

	# 创建进程
	logger.info('[MAIN] 开始创建进程')
	processes = []
	for h_name, h_type, h_aug in config.h_list:
		p = multiprocessing.Process(target=run_a_star_worker, args=(s, h_name, h_type, h_aug, result_queue, stop_event))
		processes.append(p)
		p.start()

	# 等待任意一个进程找到结果
	logger.info('[MAIN] 进程创建完成，等待结果...')
	solution = None
	try:
		first_result = result_queue.get()
		h_name, h_aug, solution, elapsed = first_result
		logger.info(f'[MAIN] 最先找到解决方案的是: {h_name}_{h_aug}')
		logger.info(f'[MAIN] 耗时: {elapsed:.2f} 秒')
		logger.info(f'[MAIN] 解的长度: {len(solution[1])} 步')
	except Exception as e:
		logger.error(f'[MAIN] 获取结果时发生错误: {str(e)}')
		logger.error('[MAIN] 没有找到解决方案')

	# 通知所有进程停止
	stop_event.set()

	# 等待所有进程结束
	for p in processes:
		p.join(timeout=5.0)
		if p.is_alive():
			p.terminate()

	logger.info('[MAIN] 所有进程已结束')

	assert solution is not None
	if config.a_star_return_type == 'state_list':
		return solution[0]
	elif config.a_star_return_type == 'operation_list':
		return solution[1]
	else:
		return solution

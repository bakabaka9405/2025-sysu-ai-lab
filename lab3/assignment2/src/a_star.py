from typing import Union, Optional, Any, cast
from collections.abc import Callable
from queue import PriorityQueue, Queue
import multiprocessing
from threading import Event
import time

import heuristic
from util import cantor_expansion, inverse_cantor_expansion, logger
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


class StateNode:
	state: FlatState
	state_id: int
	g: int
	h: int
	pos: int

	def __init__(self, state: FlatState, state_id: int, g: int, h: int, pos: int):
		self.state = state.copy()
		self.state_id = state_id
		self.g = g
		self.h = h
		self.pos = pos

	def __lt__(self, other: 'StateNode') -> bool:
		return self.g + self.h < other.g + other.h

	def wrap(self) -> tuple[FlatState, int, int, int]:
		return self.state, self.state_id, self.g, self.pos


def a_star_worker(
	state: FlatState,
	h_func: Callable[[FlatState], int],
	stop_event: Optional[Event] = None,
	task_name: str = 'UNKNOWN',
) -> Optional[tuple[list[State], list[int]]]:
	# 开闭列表
	pq: PriorityQueue[StateNode] = PriorityQueue()
	pq.put(StateNode(state, cantor_expansion(state), 0, h_func(state), state.index(15)))
	dis: dict[int, int] = {cantor_expansion(state): 0}
	# 路径追踪
	state_from: dict[int, tuple[int, int]] = {}

	cnt = 0
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
		u, uid, g, pos = u_state.wrap()

		if uid == 0:
			logger.debug(f'[{task_name}] 找到目标状态，共探索 {cnt} 个节点，步骤数: {g}')
			return reconstruct_solution(uid, state_from)

		if g >= config.a_star_max_steps != 0 and config.a_star_max_steps:
			continue

		for i in adj[pos]:
			u[i], u[pos] = u[pos], u[i]
			vid = cantor_expansion(u)
			if g + 1 < dis.get(vid, 1000):
				state_from[vid] = uid, i
				pq.put(StateNode(u, vid, g + 1, h_func(u), i))
				dis[vid] = g + 1
			u[i], u[pos] = u[pos], u[i]

	logger.debug(f'[{task_name}] 搜索完毕，未找到解决方案')
	return None


def reconstruct_solution(final_state_id: int, state_from: dict[int, tuple[int, int]]) -> tuple[list[State], list[int]]:
	"""重建从初始状态到目标状态的路径"""
	state_path = [final_state_id]
	operation = []
	while state_path[-1] in state_from:
		operation.append(state_from[state_path[-1]][1])
		state_path.append(state_from[state_path[-1]][0])
	state_path.reverse()
	operation.reverse()

	sol_state: list[State] = []
	for state_id in state_path:
		s = list(map(lambda x: (x + 1) & 0xF, inverse_cantor_expansion(state_id)))
		matrix = [s[j * 4 : j * 4 + 4] for j in range(4)]
		sol_state.append(matrix)

	return sol_state, operation


WorkerReturnType = tuple[str, Any, tuple[list[State], list[int]], float]


def run_a_star_worker(
	s: FlatState,
	h_name: str,
	h_ratio: float,
	result_queue: Queue[WorkerReturnType],
	stop_event: Event,
):
	task_name = f'{h_name}_{h_ratio}'
	logger.debug(f'[{task_name}] 开始使用 {h_name} 启发式函数 (比例 {h_ratio})...')

	h_func = heuristic.__dict__[h_name](h_ratio)

	start_time = time.time()
	result = a_star_worker(s, h_func, stop_event, task_name)
	end_time = time.time()

	if result is not None:
		elapsed = end_time - start_time
		logger.info(f'[{task_name}] 找到解决方案，用时 {elapsed:.2f} 秒')

		if not stop_event.is_set():
			result_queue.put((h_name, h_ratio, result, elapsed))
		stop_event.set()
	else:
		if stop_event.is_set():
			logger.debug(f'[{task_name}] 搜索被终止')
		else:
			logger.debug(f'[{task_name}] 未找到解决方案')

	logger.debug(f'[{task_name}] 进程结束')


def state_solvable(s: list[int]) -> bool:
	"""
	判断初始状态是否有解
	结论：将状态矩阵拍平后计算逆序对个数（空格不算在内），如果奇偶性不同于空格所在行数（从 0 算起）的奇偶性，则有解
	"""
	return (sum([1 for i in range(16) for j in range(i + 1, 16) if s[i] > s[j] and s[i] != 15]) + s.index(15) // 4) % 2 != 0


def A_star(state: Union[State, FlatState]) -> Union[list[State], list[int], tuple[list[State], list[int]], None]:
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
	for h_name, h_ratio in config.h_list:
		p = multiprocessing.Process(target=run_a_star_worker, args=(s, h_name, h_ratio, result_queue, stop_event))
		processes.append(p)
		p.start()

	# 等待任意一个进程找到结果
	logger.info('[MAIN] 进程创建完成，等待结果...')
	solution = None
	try:
		first_result = result_queue.get()
		h_name, h_ratio, solution, elapsed = first_result
		logger.info(f'[MAIN] 最先找到解决方案的是: {h_name}_{h_ratio}')
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

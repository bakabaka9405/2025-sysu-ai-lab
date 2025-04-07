from threading import Event
from queue import PriorityQueue
from generic_parallel_search import create_searcher, WorkerReturnType
from util import logger, state_adj, State, FlatState, CompressedState, HeuristicFunc
import config


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
	h_func: HeuristicFunc,
	stop_event: Event,
	task_name: str = 'UNKNOWN',
) -> WorkerReturnType:
	"""
	A* 算法主函数，由 run_worker 调用

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
		if stop_event.is_set():
			logger.debug(f'[{task_name}] 收到停止信号，终止搜索')
			return None

		cnt += 1
		if cnt % config.a_star_log_interval == 0:
			logger.debug(f'[{task_name}] 已探索 {cnt} 个节点，队列大小: {pq.qsize()}')

		if cnt % config.a_star_check_stop_interval == 0 and stop_event.is_set():
			return None

		u_state = pq.get()
		u, g, pos = u_state.wrap()

		u_flat = decompress_state(u)

		if u == target_state:
			logger.debug(f'[{task_name}] 找到目标状态，共探索 {cnt} 个节点，步骤数: {g}')
			return reconstruct_solution(u, state_from)

		if 0 != config.max_solution_length <= g:
			continue

		for i in state_adj[pos]:
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


A_star = create_searcher(a_star_worker, 'A*')

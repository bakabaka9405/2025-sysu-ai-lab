from threading import Event
from generic_parallel_search import create_searcher, WorkerReturnType
from util import FlatState, HeuristicFunc, state_adj, restore_state, logger
import config


def ida_star_worker(
	state: FlatState,
	h_func: HeuristicFunc,
	stop_event: Event,
	task_name: str = 'UNKNOWN',
) -> WorkerReturnType:
	"""
	IDA* 算法主函数，由 run_worker 调用

	:param state: 初始状态
	:param h_func: 启发式函数
	:param stop_event: 停止事件
	:param task_name: 任务名称
	:return: 状态路径和操作列表
	"""
	found = False
	stack: list[tuple[FlatState, int]] = []  # (state,last_op)
	max_depth = config.ida_star_max_depth_increment
	target_state = list(range(16))

	dfs_cnt = 0

	def ida_star(state: FlatState, g: int, pos: int, last_op: int) -> None:
		nonlocal found
		nonlocal dfs_cnt
		dfs_cnt += 1
		if stop_event.is_set() or found:
			if last_op == 0:
				logger.debug(f'[{task_name}] 搜索被终止：{stop_event.is_set()}, {found}')
			return
		stack.append((state, last_op))
		if state == target_state:
			found = True
			logger.info(f'[{task_name}] 找到解决方案')
			return
		for v in state_adj[pos]:
			if state[v] == last_op:
				continue
			next_state = state[:]
			next_state[pos], next_state[v] = next_state[v], next_state[pos]
			if g + 1 + h_func(next_state) > max_depth or (config.max_solution_length > 0 and g + 1 > config.max_solution_length):
				continue
			ida_star(next_state, g + 1, v, state[v])
			if found:
				return
		stack.pop()

	while True:
		if stop_event.is_set():
			return None
		logger.info(f'[{task_name}] 当前最大搜索步数：{max_depth}')
		dfs_cnt = 0
		ida_star(state, 0, state.index(15), -1)
		logger.debug(f'[{task_name}] 本轮搜索的节点计数：{dfs_cnt}')
		if stop_event.is_set():
			return None
		if found:
			sol = [restore_state(i[0]) for i in stack]
			sol_op = [i[1] for i in stack][1:]
			return sol, sol_op
		max_depth += config.ida_star_max_depth_increment


IDA_star = create_searcher(ida_star_worker, 'IDA*')

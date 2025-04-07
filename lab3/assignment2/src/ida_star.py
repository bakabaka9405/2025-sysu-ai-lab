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
	target_state = list(range(16))
	if state == target_state:
		return [restore_state(state)], []

	max_depth = config.ida_star_max_depth_increment
	found = False
	while True:
		stack: list[tuple[FlatState, int, int]] = [(state, 0, state.index(15))]  # (state,last_op,pos)
		adj_idxes = [len(state_adj[state.index(15)]) - 1]
		cnt = 0
		g = 0
		# 手写栈递归
		while stack:
			u_state, lst_op, pos = stack[-1]
			adj_idx = adj_idxes[-1]
			if adj_idx < 0 or 0 != config.max_solution_length <= g:
				stack.pop()
				adj_idxes.pop()
				g -= 1
				continue
			cnt += 1
			if cnt % config.search_log_interval == 0:
				logger.debug(f'[{task_name}] 在最大深度 {max_depth} 下已探索 {cnt} 个节点')
			if cnt % config.search_check_stop_interval == 0 and stop_event.is_set():
				return None
			adj_idxes[-1] -= 1
			v = state_adj[pos][adj_idx]
			if u_state[v] == lst_op:
				continue
			v_state = u_state.copy()
			v_state[pos], v_state[v] = v_state[v], 15
			if g + h_func(v_state) > max_depth:
				continue
			stack.append((v_state, u_state[v], v))
			adj_idxes.append(len(state_adj[v]) - 1)
			g += 1
			if v_state == target_state:
				found = True
				break
		if stop_event.is_set():
			return None
		if found:
			logger.debug(f'[{task_name}] 找到目标状态，步骤数: {g}')
			sol = [restore_state(i[0]) for i in stack]
			sol_op = [i[1] for i in stack[1:]]
			return sol, sol_op
		max_depth += config.ida_star_max_depth_increment


IDA_star = create_searcher(ida_star_worker, 'IDA*')

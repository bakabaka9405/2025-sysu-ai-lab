from typing import Literal, Union

h_list: list[tuple[Literal['manhattan', 'weighted_manhattan', 'misplaced', 'weighted_misplaced'], float]] = [
	('manhattan', 1.0),
	('manhattan', 1.5),
	('manhattan', 2.0),
	('manhattan', 2.5),
	('manhattan', 3.0),
	('manhattan', 3.5),
	('manhattan', 4.0),
	('weighted_manhattan', 2.0),
	('weighted_manhattan', 2.5),
	('weighted_manhattan', 3.0),
]
"""
启发式函数列表及其参数
会为每个启发式函数创建一个进程，并在进程中运行独立的 A* 算法
返回最快算出的结果
"""

logger_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'
"""
logger 的输出级别
设置为 'DEBUG' 输出最多信息
设置为 'WARNING' 及以上等级只输出错误信息
"""

a_star_return_type: Literal['state_list', 'operation_list', 'both'] = 'operation_list'
"""
A* 算法的返回值类型
可设定为 'state_list'（返回状态列表），'operation_list'（返回操作列表），或 'both'（同时返回状态列表和操作列表）
默认为课程要求的 'operation_list'
"""

a_star_log_interval: int = 10000
"""
每拓展多少个节点打印一次日志
"""

a_star_check_stop_interval: int = 1000
"""
每拓展多少个节点检查一次停止信号
"""

a_star_max_steps: int = 80
"""
A* 的最大搜索步数，不搜索超过这个步数的状态
如果设置为 0，则不限制搜索步数
如果设置为非 0 值，建议不要小于 15-puzzle 的最优解步数上限 80
"""

a_star_initial_state: Union[list[list[int]], list[int], int, None] = 1234567890127
"""
初始状态，可以是一个 4x4 的二维列表（标准格式，0 表示空），一个一维列表（0~15 的排列，15 表示空），或者一个整数（将通过逆康托展开还原为一维列表）
如果为 None，则从标准输入读取初始状态
"""

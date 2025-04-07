from collections.abc import Callable
import logging
import config

logging.basicConfig(
	level=config.logger_level,
	format='[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s',
	datefmt='%H:%M:%S',
)
logger = logging.getLogger()

state_adj: list[list[int]] = [
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
"""
状态转移表
"""

State = list[list[int]]

FlatState = list[int]

CompressedState = int

HeuristicFunc = Callable[[FlatState], int]

fact: list[int] = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000]
"""
预处理阶乘，用于康托展开
"""


def state_solvable(s: FlatState) -> bool:
	"""
	判断初始状态是否有解
	结论：将状态矩阵拍平后计算逆序对个数（空格不算在内），如果奇偶性不同于空格所在行数（从 0 算起）的奇偶性，则有解
	"""
	return (sum([1 for i in range(16) for j in range(i + 1, 16) if s[i] > s[j] and s[i] != 15]) + s.index(15) // 4) % 2 != 0


def inverse_cantor_expansion(code: int) -> list[int]:
	"""
	逆康托展开
	"""
	res = [0] * 16
	num = list(range(16))
	for i in range(16):
		index = code // fact[15 - i]
		res[i] = num.pop(index)
		code %= fact[15 - i]
	return res


def restore_state(state: FlatState) -> State:
	"""
	将 1x16 列表转换为 4x4 矩阵
	"""
	s = list(map(lambda x: (x + 1) & 0xF, state))
	return [s[j * 4 : j * 4 + 4] for j in range(4)]


def pretty_print_state(state: list[list[int]]) -> None:
	"""
	输出一个状态的矩阵
	"""
	print('---------------------')
	for row in state:
		print('|', end=' ')
		for col in row:
			if col == 0:
				print('  ', end=' | ')
			else:
				print(f'{col:2}', end=' | ')
		print()
	print('---------------------')


def pretty_print_state_list(states: list[list[list[int]]]) -> None:
	for state in states:
		pretty_print_state(state)

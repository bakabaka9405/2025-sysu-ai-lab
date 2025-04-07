import logging
import config

logging.basicConfig(
	level=config.logger_level,
	format='[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s',
	datefmt='%H:%M:%S',
)
logger = logging.getLogger()

fact: list[int] = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000]
"""
预处理阶乘，用于康托展开
"""

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

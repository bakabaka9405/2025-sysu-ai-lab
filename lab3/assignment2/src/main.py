from typing import Union, cast
from a_star import A_star
from ida_star import IDA_star
from util import inverse_cantor_expansion, pretty_print_state_list, State
import config

if __name__ == '__main__':
	# 初始状态
	if config.initial_state is None:
		s = [list(map(int, input().split())) for _ in range(4)] 
	elif isinstance(config.initial_state, int):
		s = inverse_cantor_expansion(config.initial_state)
	else:
		s = config.initial_state

	if config.algorithm == 'a_star':
		res = A_star(s)
	elif config.algorithm == 'ida_star':
		res = IDA_star(s)
	else:
		print('Invalid algorithm name')
		exit(0)

	if res is None:
		print('No solution')
		exit(0)

	sol_states: Union[list[State], None] = None
	sol_operations: Union[list[int], None] = None
	if config.search_return_type == 'state_list':
		sol_states = cast(list[State], res)
	elif config.search_return_type == 'operation_list':
		sol_operations = cast(list[int], res)
	elif config.search_return_type == 'both':
		sol_states, sol_operations = cast(tuple[list[State], list[int]], res)

	if sol_states is not None:
		print('Solution states:')
		pretty_print_state_list(sol_states)

	if sol_operations is not None:
		print('Solution operations:')
		print([(i + 1) & 0xF for i in sol_operations])

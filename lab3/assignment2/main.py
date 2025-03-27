from typing import Union, cast
from a_star import A_star, State
from util import inverse_cantor_expansion, pretty_print_state_list
import config

if __name__ == '__main__':
	# 初始状态
	if config.a_star_initial_state is None:
		s = [list(map(int, input().split())) for _ in range(4)]
	elif isinstance(config.a_star_initial_state, int):
		s = inverse_cantor_expansion(config.a_star_initial_state)
	else:
		s = config.a_star_initial_state

	res = A_star(s)
	if res is None:
		print('No solution')
		exit(0)

	sol_states: Union[list[State], None] = None
	sol_operations: Union[list[int], None] = None
	if config.a_star_return_type == 'state_list':
		sol_states = cast(list[State], res)
	elif config.a_star_return_type == 'operation_list':
		sol_operations = cast(list[int], res)
	elif config.a_star_return_type == 'both':
		sol_states, sol_operations = cast(tuple[list[State], list[int]], res)

	if sol_states is not None:
		print('Solution states:')
		pretty_print_state_list(sol_states)

	if sol_operations is not None:
		print('Solution operations:')
		print(sol_operations)

from knowledge_base import parse_knowledge_base, Term
from resolution import resolution
import config
import time


def main():
	kb = parse_knowledge_base(input())

	if config.verbose:
		print(kb)

	start = time.time()
	res = resolution(kb)
	end = time.time()

	if config.bench_info:
		print(f'Elapsed time: {end - start:.4f}s')
		print(f'Number of resolution steps: {len(res)}')

	def drop_unused_steps() -> None:
		nonlocal res
		vis = [False] * len(res)
		vis[-1] = True

		"""
		不需要递归找相关归结步骤

		新子句的两个源子句的编号必然小于自身，所以倒着扫一遍就可以了
		"""
		for i in range(len(res) - 1, -1, -1):
			if config.reserve_unused_original_clauses and i < len(kb.clauses):
				vis[i] = True
				continue

			if not vis[i]:
				continue

			(i, _), (j, _), _, _ = res[i]
			if i != -1:
				vis[i] = True
			if j != -1:
				vis[j] = True

		# 对新的步骤进行重编号
		lst_idx = -1
		new_idx = []
		for i in range(len(res)):
			if vis[i]:
				lst_idx += 1
			new_idx.append(lst_idx)
		res = [
			(
				(-1 if i == -1 else new_idx[i], x),
				(-1 if j == -1 else new_idx[j], y),
				mapping,
				clause,
			)
			for t, ((i, x), (j, y), mapping, clause) in enumerate(res)
			if vis[t]
		]

	if res == []:
		print('No resolution found.')
		return

	if config.ignore_unused_clauses:
		drop_unused_steps()

	for idx, ((i, x), (j, y), mapping, clause) in enumerate(res, 1):
		print(idx, end=' ')
		if i == -1:
			# 原始子句
			print(clause)
		else:
			# 归结子句
			# i, j 是用到的子句的编号；x, y 是用到的子句的参数编号（如果只有一个参数，则为 -1）

			def sub_id(x: int) -> str:
				return chr(x + ord('a')) if x != -1 else ''

			def mapping_str(mapping: dict[str, Term]) -> str:
				return '' if len(mapping) == 0 else f'{{{", ".join(f"{k}={v}" for k, v in mapping.items())}}}'

			print(f'R[{i+1}{sub_id(x)},{j+1}{sub_id(y)}]{mapping_str(mapping)} = {clause}')


if __name__ == '__main__':
	main()

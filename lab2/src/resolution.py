from knowledge_base import KnowledgeBase, Predicate, Clause, Term, Function, Literal
from typing import Union, NamedTuple
import config
import itertools


class ResolutionStep(NamedTuple):
	src1_id: tuple[int, int]
	src2_id: tuple[int, int]
	mapping: dict[str, Term]
	new_clause: Clause


class ResolutionResult(list[ResolutionStep]):
	pass


def make_nth_vaiable(n: int) -> str:
	"""
	将数字 n 作为标号，返回从 a 开始的第 n 个变量名
	"""
	if n < 0:
		raise ValueError('Negative variable number')
	if n > 25:
		raise ValueError('Too many variables')
	if config.variable_name_type == 'single':
		return chr(n + ord('a'))
	elif config.variable_name_type == 'double':
		return chr(n + ord('a')) * 2
	else:
		raise ValueError('Invalid variable name type')


def alpha_conversion(cl: Clause, next_alpha_id: int = 0) -> tuple[dict[str, str], dict[str, str]]:
	"""
	约束变量改名

	两个子句出现同名变量时，不应看作同一个变量

	为了方便直接把出现的变量统一全部改掉，从 a 开始编号

	这是原地操作！！

	返回两个表，第一个是反向表，将新的变量名映射为原来的变量名

	第二个是正向表，将原来的变量名映射为新的变量名

	返回正向表的作用是快速查询某个原变量是否同时在两个子句都出现过
	"""
	mapping_from: dict[str, str] = {}
	mapping_to: dict[str, str] = {}

	def alpha_conversion_term(term: Term) -> None:
		nonlocal next_alpha_id
		nonlocal mapping_from
		nonlocal mapping_to
		if term.is_literal():
			if term.get_literal().is_variable():
				key = term.get_literal().name
				if key not in mapping_to:
					if next_alpha_id > 25:
						raise ValueError('Too many variables')
					next_alpha = make_nth_vaiable(next_alpha_id)
					mapping_from.update({next_alpha: key})
					mapping_to.update({key: next_alpha})
					next_alpha_id += 1
				term.get_literal().name = mapping_to[key]
		else:
			for i in term.get_function().args:
				alpha_conversion_term(i)

	for i in cl.predicates:
		for j in i.args:
			alpha_conversion_term(j)
	return mapping_from, mapping_to


def resolution(kb: KnowledgeBase) -> ResolutionResult:
	"""
	使用归结原理判断 goal 是否可以从 kb 推导出来。

	不修改传入的对象，返回归结过程。

	如果可以推导出，则返回一个包含归结步骤的列表，否则返回空列表。
	"""
	clauses = kb.clauses[:]
	result = ResolutionResult()
	for clause in clauses:
		result.append(ResolutionStep((-1, -1), (-1, -1), {}, clause))

	"""
	先移动右指针再移动左指针

	在不使用优先队列的情况下，尽可能让先归结出的子句的步骤是较少的
	"""

	vis_str: set[str] = set(map(str, clauses))

	j = 0
	while j < len(clauses):
		clause2 = clauses[j].copy()
		conv_from2, conv_to2 = alpha_conversion(clause2)
		for i in range(j):
			clause1 = clauses[i].copy()
			conv_from1, conv_to1 = alpha_conversion(clause1, len(conv_from2))
			conv_from = {**conv_from1, **conv_from2}

			if config.verbose:
				print('conv:', clause1, clause2)

			resolve_result = resolve_clause(clause1, clause2)
			if resolve_result is None:
				if config.verbose:
					print('cannot resolve:', clause1, clause2)
				continue

			x, y, mgu_mapping, new_clause = resolve_result

			"""
			恢复被改名的约束变量，更新 mgu_mapping 为 resolve_mapping，用以输出归结步骤

			new_clause 中的变量也要恢复
			"""

			def restore_key(key: str, is_new_clause: bool = False) -> str:
				assert key in conv_from
				src = conv_from[key]
				src_id = '₁₂'[key in conv_from2]
				if src in conv_to1 and src in conv_to2:
					if not is_new_clause:
						src += src_id
				else:
					if config.always_show_mapping_source and not is_new_clause:
						src += src_id
				return src

			def restore_term(term: Term, is_new_clause: bool = False) -> Term:
				if term.is_literal():
					name = term.get_literal().name
					if name in conv_from:
						name = restore_key(name, is_new_clause)
					return Term(Literal(name))
				else:
					return Term(
						Function(
							term.get_function().name,
							[restore_term(i, is_new_clause) for i in term.get_function().args],
						)
					)

			resolve_mapping = {restore_key(i): restore_term(j) for i, j in mgu_mapping.items()}

			for pred in new_clause.predicates:
				for t, term in enumerate(pred.args):
					pred.args[t] = restore_term(term, is_new_clause=True)

			if len(clause1.predicates) == 1:
				x = -1
			if len(clause2.predicates) == 1:
				y = -1
			
			if config.drop_duplicate_clauses:
				if str(new_clause) not in vis_str:
					vis_str.add(str(new_clause))
					clauses.append(new_clause)
					result.append(ResolutionStep((i, x), (j, y), resolve_mapping, new_clause))
			else:
				clauses.append(new_clause)
				result.append(ResolutionStep((i, x), (j, y), resolve_mapping, new_clause))

			if new_clause.predicates == []:
				return result
		j += 1
	return ResolutionResult()


def resolve_clause(clause1: Clause, clause2: Clause) -> Union[tuple[int, int, dict[str, Term], Clause], None]:
	"""
	对两个子句进行归结操作。

	这两个子句应当是经过约束变量改名的、没有重复变量的子句。

	不改变传入的对象，返回新的归结后的子句。
	"""
	for i, pred1 in enumerate(clause1.predicates):
		for j, pred2 in enumerate(clause2.predicates):
			# 前提检查
			if pred1.name != pred2.name or pred1.neg == pred2.neg:
				continue

			mgu_mapping = mgu(pred1, pred2)
			if mgu_mapping is None:
				continue

			# 根据 mgu 的结果，对两个子句进行变量替换
			clause1_mgu, clause2_mgu = clause1.copy(), clause2.copy()
			replace_keys(clause1_mgu, mgu_mapping)
			replace_keys(clause2_mgu, mgu_mapping)

			# mgu 不改变传入对象的值，所以这里需要手动替换一遍
			# 以便在后续找到需要删除的谓词
			pred1 = pred1.copy()
			pred2 = pred2.copy()
			replace_keys(pred1, mgu_mapping)
			replace_keys(pred2, mgu_mapping)

			new_clause = Clause()
			pd_set1 = set(clause1_mgu.predicates)
			pd_set2 = set(clause2_mgu.predicates)
			pd_set1.discard(pred1)
			pd_set2.discard(pred2)

			new_clause.predicates = list(pd_set1.union(pd_set2))

			if config.verbose:
				print('resolve:', clause1, clause2, '->', new_clause)
			return i, j, mgu_mapping, new_clause
	return None


def replace_keys(obj: Union[Clause, Predicate, Term], mapping: dict[str, Term]) -> None:
	"""
	对对象进行变量替换。

	传入的对象可以是子句、谓词或项。

	无论传入哪种参数都是原地操作
	"""
	if isinstance(obj, Clause):
		for pred in obj.predicates:
			replace_keys(pred, mapping)
	elif isinstance(obj, Predicate):
		for arg in obj.args:
			replace_keys(arg, mapping)
	elif isinstance(obj, Term):
		if obj.is_literal():
			if obj.get_literal().name in mapping:
				obj.assign(mapping[obj.get_literal().name])
		else:
			for arg in obj.get_function().args:
				replace_keys(arg, mapping)


def mgu(pred1: Predicate, pred2: Predicate) -> Union[dict[str, Term], None]:
	"""
	对两个谓词进行最小一般化统一操作。

	成功则返回新对象和一个映射表，原有的对象不变。

	同样要求两个谓词不能有名字相同的变量
	"""

	"""
	沟槽的 py 没有所有权机制，都不知道对象哪来的

	写少一个 copy 估计要调成傻逼，索性全 copy 了
	"""
	if config.verbose:
		print('mgu:', pred1, pred2)

	if pred1.name != pred2.name or len(pred1.args) != len(pred2.args):
		return None

	mapping: dict[str, Term] = {}

	"""
	使用队列实现，但这不代表复杂度线性，至少我不会证明，这说明它大概率不是线性

	用课件的 P(a,x,h(g(z))) 和 P(z,h(y),h(y)) 作为合一的示例：

	0	init			add: a = z
	0	init			add: h = h(y)
	0	init			add: h(g(z)) = h(y)
	1	a = z			export: z = a
	2	x = h(y)		add: x = h(y)
	3	h(g(z)) = h(y)	add: g(z) = y
	4	x = h(y)		add: x = h(y)
	5	g(z) = y		add: g(a) = y
	6	x = h(y)		add: x = h(y)
	7	g(a) = y		export: y = g(a)
	8	x = h(y)		add: x = h(g(a))
	9	x = h(g(a))		export: x = h(g(a))

	export values(by id):
	1. z=a
	2. y=g(a)
	3. x=h(g(a))

	更新：这个过程是假的，因为两个 z 根本就是不同的约束变量

	但是方法其实没问题，懒得删了
	"""

	"""
	队列元素格式：标记和两个 Term
	标记不生效时为 -1，表示这个 Term 还没有被处理过
	标记生效时表示同样的 Term 上次出现的位置，作用见下方 x = y 或 x = f(y) 的部分
	"""
	pending: list[tuple[int, Term, Term]] = [(-1, i.copy(), j.copy()) for i, j in zip(pred1.args, pred2.args)]
	last_update = -1
	i = 0
	while i < len(pending):
		src, term_a, term_b = pending[i]
		term_a, term_b = term_a.copy(), term_b.copy()
		replace_keys(term_a, mapping)
		replace_keys(term_b, mapping)
		if term_a.is_constant() and term_b.is_constant():
			if term_a != term_b:
				return None
		elif term_a.is_function() and term_b.is_function():
			if term_a.get_function().header_equal_to(term_b.get_function()):
				pending.extend(zip(itertools.repeat(-1), term_a.get_function().args, term_b.get_function().args))
				last_update = i
			else:
				return None
		elif (term_a.is_constant() and term_b.is_function()) or (term_a.is_function() and term_b.is_constant()):
			# 不能出现 f(x) = a 的情况
			return None
		else:
			"""
			还剩几种情况
			1. x = y
			2. x = f(a)
			3. f(a) = x
			4. x = f(y)
			5. f(y) = x
			"""
			if term_b.is_literal() and term_b.get_literal().is_variable():
				term_a, term_b = term_b, term_a
			assert term_a.is_literal() and term_a.get_literal().is_variable()

			if term_b.is_constant() or ((term_b.is_literal() or term_b.is_function()) and src > last_update):
				"""
				x = a 或 x = f(a)，优先让变量映射到常量

				或 x = y 或 x = f(y)，当且仅当 src > last_update
				即这个待处理的 Term 对从入队到出队的整个过程中，mgu 没有任何新的更新时，
				不得不尝试将一个变量映射到另一个变量
				"""
				if term_a.get_literal().name in mapping:
					if term_b != mapping[term_a.get_literal().name]:
						return None
				else:
					# 所有权在这里发生转交
					mapping[term_a.get_literal().name] = term_b
					last_update = i
					if config.verbose:
						print('export:', term_a.get_literal().name, '=', term_b)
			else:
				"""
				x = y 或 x = f(y)
				此时还无法判断映射关系，塞回队列以后再来
				"""
				pending.append((i, term_a, term_b))
				if config.verbose:
					print('add:', term_a, term_b)
		i += 1
		if i > config.mgu_queue_threshold:
			return None

	"""
	当确实发生了一个变量映射到另一个变量的情况时，可能发生：
	1. x->y
	2. y->z

	此时需要进行修补，即让 x->z

	也可能发生
	1. x->f(y)
	2. y->f(x)

	甚至发生
	1. x->f(x)

	这样就爆了，需要返回 None

	写了下面两个函数来处理这些问题
	"""

	vis: set[str] = set()
	conflict = False

	def fix_function_dependency(func: Function):
		nonlocal conflict
		for i, arg in enumerate(func.args):
			if arg.is_literal():
				if arg.is_constant():
					continue
				name = arg.get_literal().name
				if name not in mapping:
					continue
				fix_key_dependency(name)
				if conflict:
					return
				func.args[i] = mapping[name]
			else:
				fix_function_dependency(arg.get_function())
				if conflict:
					return

	def fix_key_dependency(key: str):
		nonlocal conflict
		term = mapping[key]
		if key in vis:
			conflict = True
		if conflict:
			return
		vis.add(key)
		if term.is_literal():
			name = term.get_literal().name
			if name in mapping:
				fix_key_dependency(name)
				mapping[key] = mapping[name]
		else:
			fix_function_dependency(term.get_function())
		vis.discard(key)

	for key in mapping:
		fix_key_dependency(key)
		if conflict:
			return None

	return mapping

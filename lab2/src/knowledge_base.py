from typing import Union
import config


class Literal:
	"""
	常量或变量
	"""

	name: str

	def __init__(self, name: str = ''):
		self.name = name

	def is_variable(self) -> bool:
		"""
		根据变量名判断是否为变量
		"""
		if config.variable_name_type == 'single':
			return len(self.name) == 1
		elif config.variable_name_type == 'double':
			return len(self.name) == 2 and self.name[0] == self.name[1]
		else:
			raise ValueError('Invalid variable name type')

	def is_constant(self) -> bool:
		return not self.is_variable()

	def __str__(self) -> str:
		return self.name

	def __eq__(self, other: object) -> bool:
		if not isinstance(other, Literal):
			return False
		return self.name == other.name

	def __repr__(self) -> str:
		return self.__str__()

	def copy(self) -> 'Literal':
		return Literal(self.name)


class Function:
	"""
	函数名和参数列表
	"""

	name: str
	args: list['Term']

	def __init__(self, name: str = '', args: Union[list['Term'], None] = None):
		self.name = name
		if args is None:
			self.args = []
		else:
			self.args = args

	def __str__(self) -> str:
		return f'{self.name}({", ".join([str(i) for i in self.args])})'

	def __repr__(self) -> str:
		return self.__str__()

	def __eq__(self, other: object) -> bool:
		if not isinstance(other, Function):
			return False
		return self.name == other.name and len(self.args) == len(other.args) and all([i == j for i, j in zip(self.args, other.args)])

	def copy(self) -> 'Function':
		return Function(self.name, [i.copy() for i in self.args])

	def header_equal_to(self, other: 'Function') -> bool:
		"""
		判断函数名和参数数量是否相同
		"""
		return self.name == other.name and len(self.args) == len(other.args)


class Term:
	value: Union[Literal, Function]

	def is_literal(self):
		return isinstance(self.value, Literal)

	def is_function(self):
		return isinstance(self.value, Function)

	"""
	to let the linter happy, define 2 getters
	"""

	def get_literal(self) -> Literal:
		assert isinstance(self.value, Literal)
		return self.value

	def get_function(self) -> Function:
		assert isinstance(self.value, Function)
		return self.value

	def is_constant(self) -> bool:
		"""
		常量，或者只含常量参数的函数
		"""
		if isinstance(self.value, Literal):
			return self.value.is_constant()
		else:
			return all([i.is_constant() for i in self.value.args])

	def __init__(self, value: Union[None, Literal, Function] = None):
		if value is not None:
			self.value = value
		else:
			self.value = Literal()

	def add_function_arg(self, arg: 'Term') -> None:
		assert isinstance(self.value, Function)
		self.value.args.append(arg)

	def __str__(self):
		if isinstance(self.value, Literal):
			return str(self.value)
		else:
			name, args = self.value.name, self.value.args
			return f'{name}({", ".join([str(i) for i in args])})'

	def __repr__(self):
		return self.__str__()

	def copy(self) -> 'Term':
		return Term(self.value.copy())

	def __eq__(self, other: object) -> bool:
		if not isinstance(other, Term):
			return False
		if type(self.value) is type(other.value):
			return self.value == other.value
		else:
			return False

	def assign(self, other: 'Term') -> None:
		"""
		将一个 Term 的值赋值给另一个 Term
		"""
		self.value = other.value.copy()


class Predicate:
	"""
	一个谓词及其参数。
	"""

	name: str
	neg: bool
	args: list[Term]

	def __init__(self):
		self.name = ''
		self.neg = False
		self.args = []

	def __str__(self) -> str:
		neg_s = '~' if self.neg else ''
		if len(self.args) == 0:
			return f'{neg_s}{self.name}'
		else:
			return f'{neg_s}{self.name}({", ".join([str(i) for i in self.args])})'

	def __hash__(self) -> int:
		return hash(self.__str__())

	def __repr__(self) -> str:
		return self.__str__()

	def __getitem__(self, index: int) -> Term:
		return self.args[index]

	def __eq__(self, other: object) -> bool:
		if not isinstance(other, Predicate):
			return False
		if self.name != other.name or len(self.args) != len(other.args):
			return False
		for i in range(len(self.args)):
			if self.args[i] != other.args[i]:
				return False
		return True

	def copy(self) -> 'Predicate':
		pd = Predicate()
		pd.name = self.name
		pd.neg = self.neg
		pd.args = [i.copy() for i in self.args]
		return pd


class Clause:
	"""
	一个子句由多个谓词组成。
	"""

	predicates: list[Predicate]

	def __init__(self):
		self.predicates = []

	def __str__(self) -> str:
		return f'({", ".join([str(i) for i in self.predicates])}{"," if len(self.predicates)==1 else ""})'

	def __repr__(self) -> str:
		return self.__str__()

	def __getitem__(self, index: int) -> Predicate:
		return self.predicates[index]

	def copy(self) -> 'Clause':
		cl = Clause()
		cl.predicates = [i.copy() for i in self.predicates]
		return cl


class KnowledgeBase:
	"""
	一个知识库由多个子句组成。
	"""

	clauses: list[Clause]

	def __init__(self):
		self.clauses = []

	def __str__(self) -> str:
		return f'{{{", ".join([str(i) for i in self.clauses])}}}'

	def __repr__(self) -> str:
		return self.__str__()

	def __getitem__(self, index: int) -> Clause:
		return self.clauses[index]


def parse_term(tm_raw: str) -> Term:
	"""
	解析一个函数或常量。
	"""
	tm_lst: list[Term] = []
	current_name = ''

	def try_push_name(is_function: bool) -> None:
		"""
		将当前的名称（如果不为空）压入栈中
		"""
		nonlocal current_name
		nonlocal tm_lst
		if current_name == '':
			return
		tm_lst.append(Term())
		tm_lst[-1].value = (Function if is_function else Literal)(current_name)
		current_name = ''

	def push_term() -> None:
		"""
		取出栈顶的 Term，将其作为栈次顶元素（应当是一个函数）的参数
		"""
		nonlocal tm_lst
		assert tm_lst[-2].is_function()
		tm_lst[-2].add_function_arg(tm_lst[-1])
		tm_lst.pop()

	for char in tm_raw:
		if char == '(':
			# 禁止无意义的括号
			assert current_name != ''
			try_push_name(True)
		elif char == ')':
			# 假设输入不存在零参数函数，即类似 f() 的情况
			# 所以右括号前要么是变量名，要么是另一个函数的右括号，如 f(g(a))
			# 所以右括号的作用和逗号其实是一致的
			try_push_name(False)
			push_term()
		elif char == ',':
			try_push_name(False)
			push_term()
		else:
			current_name += char

	# 如果整个 Term 只是一个 Literal
	try_push_name(False)

	assert len(tm_lst) == 1
	return tm_lst[0]


def parse_predicate(pd_raw: str) -> Predicate:
	"""
	解析一个谓词（及其参数）

	考虑到内部参数可能有函数嵌套的情况，所以不能单纯用逗号分隔

	用一个栈处理
	"""
	pd = Predicate()
	pd.neg = pd_raw[0] == '~'
	if pd.neg:
		pd_raw = pd_raw[1:]
	if '(' not in pd_raw:
		pd.name = pd_raw
		pd.args = []
	else:
		pd.name = pd_raw[: pd_raw.index('(')]
		args_raw = []
		depth = 0
		last_pos = pd_raw.index('(') + 1
		for i in range(pd_raw.index('(') + 1, len(pd_raw) - 1):
			if pd_raw[i] == '(':
				depth += 1
			elif pd_raw[i] == ')':
				depth -= 1
			elif pd_raw[i] == ',' and depth == 0:
				args_raw.append(pd_raw[last_pos:i])
				last_pos = i + 1
		if last_pos < len(pd_raw) - 1:
			args_raw.append(pd_raw[last_pos:-1])
		pd.args = [parse_term(i) for i in args_raw]

	return pd


def parse_clause(cl_raw: str) -> Clause:
	"""
	解析一个子句

	过程和 parse_predicate 几乎完全相同
	"""
	cl = Clause()
	pd_raws: list[str] = []
	last_pos = 0
	depth = 0
	for i, char in enumerate(cl_raw):
		if char == '(':
			depth += 1
		elif char == ')':
			depth -= 1
		elif char == ',' and depth == 0:
			pd_raws.append(cl_raw[last_pos:i])
			last_pos = i + 1
	if last_pos < len(cl_raw):
		pd_raws.append(cl_raw[last_pos:])
	cl.predicates = [parse_predicate(i) for i in pd_raws]
	return cl


def parse_knowledge_base(kb_raw: str) -> KnowledgeBase:
	"""
	解析知识库

	这里偷一下懒，因为子句集都是逗号分隔的、没有前置名称的括号，所以可以用 "),(" 来分割
	"""
	kb = KnowledgeBase()
	kb_raw = kb_raw.replace(' ', '')
	fm_raws = kb_raw[kb_raw.find('{') + 2 : -2].split('),(')
	kb.clauses = [parse_clause(i) for i in fm_raws]
	return kb

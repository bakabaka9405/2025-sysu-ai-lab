ignore_unused_clauses: bool = True
"""
输出归结过程时，忽略未使用的子句

建议打开，否则证明过程极长
"""

reserve_unused_original_clauses: bool = True
"""
输出归结过程时，保留所有原始子句（无论有没有用到）
"""

mgu_queue_threshold: int = 10000
"""
进行 mgu 操作时队列长度的阈值

如果队列长度超过该值，则认为 mgu 操作失败，返回 None
"""

always_show_mapping_source: bool = False
"""
输出归结过程时，mgu 映射的源变量总是标明来自哪个子句（1 或 2）

否则，只在该变量名同时在两条子句中出现时才标明
"""

variable_name_type: str = 'single'
"""
为了兼容不同的输入格式

设为 'single' 时，将所有单长度名视为变量名

设为 'double' 时，将所有双长度且两个字母相同的名字视为变量名
"""

drop_duplicate_clauses: bool = True
"""
归结过程中是否通过使用 set 去除重复的子句
"""

verbose: bool = False
"""
输出归结过程的详细信息（其实没多详细而且也没什么用）
"""

bench_info: bool = False
"""
输出性能信息（包括计算时间和归结次数）
"""

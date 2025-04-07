from typing import Optional, Literal

seed: int = 0
"""
随机种子。
如果为0，则表示每次运行都使用不同的随机种子。
如果要保证运行结果一致，在设置为非 0 的同时还应该避免使用多进程。
"""

initial_population_size: int = 10000
"""
最初的种群大小。
使用 numpy 的 permutation 函数生成初始种群。
"""

maximum_population_size: int = 20000
"""
最大种群大小。
如果某个 epoch 过后种群大小超过这个值，则保留最优的前 maximum_population_size 个个体。
"""

cross_per_epoch: int = 30000
"""
每个 epoch 进行交叉的次数。
最多产生 2*cross_per_epoch 个子代。
"""

mutation_prob = 0.10
"""
变异概率。
每个新个体有 mutation_prob 的概率进行变异。
"""

crossover: Literal['partial_mapping_crossover', 'order_crossover', 'position_based_crossover'] = 'order_crossover'
"""
交叉方式。
- partial_mapping_crossover: 部分映射交叉
- order_crossover: 次序交叉
- position_based_crossover: 基于位置的交叉
"""

mutation: Literal['swap_mutation','reverse_mutation','shuffle_mutation','random_mutation'] = 'swap_mutation'
"""
变异方式。
- swap_mutation: 随机选取两个顺序编码交换
- reverse_mutation: 随机选取一段编码反转
- shuffle_mutation: 随机选取一段编码随机打乱
- random_mutation: 每次随机选取以上三种变异的一种
"""

num_worker: int = 20
"""
进行交叉操作创建的进程数
每个进程进行 ceil(cross_per_epoch/num_worker) 次交叉操作
"""

input_path: str = 'lab3/assignment4/data/dj38.tsp'
"""
输入文件路径
"""

output_path_dir: Optional[str] = 'C:/Temp/lab3'
"""
输出文件路径
如果为 None，则表示不输出任何文件。
"""

logger_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'DEBUG'
"""
logger 的输出级别
设置为 'DEBUG' 输出最多信息
设置为 'WARNING' 及以上等级只输出错误信息
"""
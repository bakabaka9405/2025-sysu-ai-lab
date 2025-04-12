from typing import Optional, Literal

seed: Optional[int] = None
"""
随机种子。
如果为 None，则表示每次运行都使用不同的随机种子。
如果要保证运行结果一致，在设置为非 None 的同时还应保证进程数一致。
"""

initial_population_size: int = 20000
"""
最初的种群大小。
使用 numpy 的 permutation 函数生成初始种群。
"""

maximum_population_size: int = 20000
"""
最大种群大小。
如果某个 epoch 过后种群大小超过这个值，则保留最优的前 maximum_population_size 个个体。
"""

cross_per_epoch: int = 50000
"""
每个 epoch 进行交叉的次数。
最多产生 2*cross_per_epoch 个子代。
"""

fitness_transform_policy: Literal[
	'linear_baseline',
	'sqrt_linear_baseline',
	'exponential_baseline',
	'inverse_sqrt',
] = 'inverse_sqrt'
"""
将适应度进行变换（如果使用轮盘赌选择）
"""

selection_policy: Literal['RouletteWheel', 'Tournament'] = 'Tournament'
"""
选择策略，实现了轮盘赌和锦标赛两种策略
"""

tournament_size: int = 2
"""
锦标赛选择的大小（如果使用锦标赛选择）
"""

base_mutation_prob = 0.02
"""
基础变异概率。
每个个体有 mutation_prob 的概率进行变异。
当长时间没有找到更优解时，变异概率会逐渐增加。
"""

mutation_punishment: float = 0.3
"""
一个 epoch 内没有找到更优解时，变异概率增加的惩罚因子。
仅对父辈生效
parent_mutation_prob = base_mutation_prob + mutation_punishment * (epoch - last_epoch_no_improve)
"""

mutation_recovery: float = 8.0
"""
一个 epoch 内找到更优解时，变异概率恢复因子。
parent_mutation_prob /= mutation_reward
"""

homozygous_lethality: float = 0.95
"""
纯合子致死率。
如果两个父代的基因相同，则后代有一定概率的概率直接死亡。
较大时能够提升种群多样性，较小时能够提升收敛速度。
"""

crossover_policy: Literal['partial_mapping_crossover', 'order_crossover', 'position_based_crossover'] = 'order_crossover'
"""
交叉方式。
- partial_mapping_crossover: 部分映射交叉
- order_crossover: 次序交叉
- position_based_crossover: 基于位置的交叉
"""

mutation_policy: Literal[
	'swap_mutation',
	'adjacent_swap_mutation',
	'range_swap_mutation',
	'reverse_mutation',
	'rorate_mutation',
	'shuffle_mutation',
	'random_mutation',
	'random_mutation_1',
] = 'random_mutation'
"""
变异方式。
- swap_mutation: 随机选取两个顺序编码交换
- adjacent_swap_mutation: 随机选取两个相邻顺序编码交换
- range_swap_mutation: 随机选取两段顺序编码交换
- reverse_mutation: 随机选取一段编码反转
- rorate_mutation: 随机选取一段编码循环移动
- shuffle_mutation: 随机选取一段编码随机打乱
- random_mutation: 每次随机选取以上几种变异的一种
- random_mutation_1: 不知道扔掉一点垃圾变异能不能提高性能
"""

num_worker: int = 6
"""
进行交叉操作创建的进程数
每个进程进行 ceil(cross_per_epoch/num_worker) 次交叉操作
"""

worker_data_copy_threshold: int = 1000 * 10000 * 8
"""
虽然 population_data 是通过共享内存传递的，但直接读共享内存的延迟有亿点大，所以在数据小的时候一般将其复制到进程独立内存
数据量大且设置进程太多时会爆内存
设置一个上界（单条基因点数 * 基因数 * 进程数）决定是否复制
设为 0 即总使用共享内存
理论上这块设置多少不会影响最终结果
"""

input_path: str = 'lab3/assignment4/data/uy734.tsp'
"""
输入文件路径
"""

output_path_dir: Optional[str] = 'C:/Temp/lab3/uy734/3'
"""
输出文件路径
如果为 None，则表示不输出任何文件。
"""

console_logger_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'DEBUG'
"""
控制台日志的输出级别
设置为 'DEBUG' 输出最多信息
设置为 'WARNING' 及以上等级只输出错误信息
"""

log_to_file: bool = True
"""
日志是否输出到文件
如果是，将输到控制台的同时输出到 output_path_dir/log.txt
输出到文件的内容不包含 DEBUG 级别的信息
否则只在控制台输出
"""

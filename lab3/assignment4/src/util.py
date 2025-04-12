import random
import numpy as np
from numpy.typing import NDArray
import os
import logging
import config
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('TSP')

logging.basicConfig(
	level=config.console_logger_level,
	format='[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s',
	datefmt='%H:%M:%S',
)

mat_logger = logging.getLogger('matplotlib')
mat_logger.setLevel(logging.ERROR)

Gene = NDArray[np.int64]

Individual = tuple[Gene, float]

plt.rc('font', family='SimSun')


def set_random_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)


def log_basic_info() -> None:
	"""
	输出 config.py 中的基本参数配置
	"""
	logger.info(f'输入文件路径: {config.input_path}')
	if config.output_path_dir:
		logger.info(f'输出文件路径: {config.output_path_dir}')
	logger.info(f'随机种子: {config.seed}')
	logger.info(f'工作进程数: {config.num_worker}')
	logger.info(f'初始种群大小: {config.initial_population_size}')
	logger.info(f'最大种群大小: {config.maximum_population_size}')
	logger.info(f'每轮交叉次数： {config.cross_per_epoch}')
	logger.info(f'交叉方式: {config.crossover_policy}')
	logger.info(f'变异方式: {config.mutation_policy}')
	logger.info(f'基础变异概率: {config.base_mutation_prob}')
	logger.info(f'变异乘法因子: {config.mutation_punishment}')
	logger.info(f'变异恢复因子： {config.mutation_recovery}')
	logger.info(f'纯合子致死率: {config.homozygous_lethality}')
	logger.info(f'距离映射方式: {config.fitness_transform_policy}')
	logger.info(f'选择策略： {config.selection_policy}')


def prepare_output_dir(path: str) -> None:
	if not os.path.exists(path):
		os.makedirs(path)
	if config.output_path_dir and config.log_to_file:
		fh = logging.FileHandler(f'{config.output_path_dir}/log.txt', mode='w', encoding='utf-8')
		fh.setLevel(logging.INFO)
		formatter = logging.Formatter('[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
		fh.setFormatter(formatter)
		logger.addHandler(fh)


def path_distance(cities: NDArray, path: NDArray) -> float:
	return np.sum(np.sqrt(np.sum((cities[path] - cities[np.roll(path, -1)]) ** 2, axis=1)))


def plot_path(cities: NDArray, path: NDArray, epoch: int):
	"""
	绘制路径图像并保存到指定目录。
	:param path: 城市路径
	:param epoch: 当前迭代次数，用于命名保存的图像文件
	"""

	fig, ax = plt.subplots(figsize=(10, 8))

	# 绘制所有城市点
	x = cities[:, 0]
	y = cities[:, 1]
	ax.scatter(x, y, c='blue', marker='o', alpha=0.7)

	# 绘制起点和终点
	ax.scatter(cities[path[0]][0], cities[path[0]][1], c='green', marker='o')
	ax.scatter(cities[path[-1]][0], cities[path[-1]][1], c='cyan', marker='o')

	# 绘制路径连线
	for i in range(len(path) - 1):
		ax.plot([cities[path[i]][0], cities[path[i + 1]][0]], [cities[path[i]][1], cities[path[i + 1]][1]], 'r-', alpha=0.5)

	ax.plot([cities[path[-1]][0], cities[path[0]][0]], [cities[path[-1]][1], cities[path[0]][1]], 'r-', alpha=0.5)

	# 添加标题和标签
	ax.set_title(f'TSP路径 (总距离: {path_distance(cities,path):.2f})')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.grid(True, linestyle='--', alpha=0.7)

	fig.tight_layout()
	fig.savefig(f'{config.output_path_dir}/epoch_{epoch}.png')
	fig.savefig(f'{config.output_path_dir}/best.png')
	plt.close(fig)


def plot_best_record(best_record: list[float]) -> None:
	fig, ax = plt.subplots(figsize=(10, 8))
	ax.ticklabel_format(axis='y', style='plain')
	ax.plot(best_record, color='blue', label='最短距离')
	ax.set_title('每轮最短距离变化')
	ax.set_xlabel('迭代次数')
	ax.set_ylabel('最短距离')
	ax.legend()
	ax.grid(True)
	fig.savefig(f'{config.output_path_dir}/best_record.png')
	plt.close(fig)

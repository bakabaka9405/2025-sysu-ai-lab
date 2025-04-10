from typing import Union
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
logger.setLevel(config.logger_level)

logging.basicConfig(
	level=config.logger_level,
	format='[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s',
	datefmt='%H:%M:%S',
)

if config.output_path_dir and config.log_to_file:
	fh = logging.FileHandler(f'{config.output_path_dir}/log.txt', mode='w', encoding='utf-8')
	fh.setLevel(config.logger_level)
	formatter = logging.Formatter('[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
	fh.setFormatter(formatter)
	logger.addHandler(fh)


mat_logger = logging.getLogger('matplotlib')
mat_logger.setLevel(logging.ERROR)

Gene = NDArray[np.int64]

Individual = tuple[Gene, float]

plt.rc('font', family='SimSun')


def set_random_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)


def create_output_dir(path: str) -> None:
	if not os.path.exists(path):
		os.makedirs(path)


def path_distance(cities: NDArray, path: NDArray) -> float:
	tail = len(cities) - 1
	a = np.append(path, [tail], axis=-1)
	b = np.append(path[1:], [tail, path[0]], axis=-1)
	return np.sum(np.sqrt(np.sum((cities[a] - cities[b]) ** 2, axis=1)))


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

	# 绘制路径连线
	for i in range(len(path) - 1):
		ax.plot([cities[path[i]][0], cities[path[i + 1]][0]], [cities[path[i]][1], cities[path[i + 1]][1]], 'r-', alpha=0.5)

	# 补全最后两根连线
	ax.plot([cities[path[-1]][0], cities[-1][0]], [cities[path[-1]][1], cities[-1][1]], 'r-', alpha=0.5)
	ax.plot([cities[path[0]][0], cities[-1][0]], [cities[path[0]][1], cities[-1][1]], 'r-', alpha=0.5)

	# 添加标题和标签
	ax.set_title(f'TSP路径 (总距离: {path_distance(cities,path):.2f})')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.grid(True, linestyle='--', alpha=0.7)

	fig.tight_layout()
	fig.savefig(f'{config.output_path_dir}/epoch_{epoch}.png')
	fig.savefig(f'{config.output_path_dir}/best.png')
	plt.close(fig)

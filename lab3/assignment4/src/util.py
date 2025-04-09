import random
import numpy as np
from numpy.typing import NDArray
import os
import logging
import config
import matplotlib.pyplot as plt

logging.basicConfig(
	level=config.logger_level,
	format='[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s',
	datefmt='%H:%M:%S',
)
logger = logging.getLogger('TSP')

mat_logger = logging.getLogger('matplotlib')
mat_logger.setLevel(logging.ERROR)

Gene = NDArray[np.int64]

Individual = tuple[Gene, float]

plt.rc('font', family='SimSun')


def set_random_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)


def euclidean(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
	return np.sqrt(np.sum(np.square(x - y)))


def create_output_dir(path: str) -> None:
	if not os.path.exists(path):
		os.makedirs(path)

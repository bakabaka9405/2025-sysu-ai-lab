import random
import numpy as np
from numpy.typing import NDArray
import os
import logging
import config

logging.basicConfig(
	level=config.logger_level,
	format='[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s',
	datefmt='%H:%M:%S',
)
logger = logging.getLogger('TSP')

mat_logger = logging.getLogger('matplotlib')
mat_logger.setLevel(logging.ERROR)

gen = np.random.Generator(np.random.PCG64())


def set_random_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	global gen
	gen = np.random.Generator(np.random.PCG64(seed))


def hypot(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
	return np.sqrt(np.sum(np.square(x - y), axis=0))


def create_output_dir(path: str) -> None:
	if not os.path.exists(path):
		os.makedirs(path)

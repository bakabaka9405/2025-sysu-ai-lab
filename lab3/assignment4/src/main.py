import config
import util
import GeneticAlgTSP
import random


def main():
	if not config.seed:
		config.seed = random.randint(0, 2**31 - 1)
	util.set_random_seed(config.seed)

	if config.output_path_dir:
		util.prepare_output_dir(config.output_path_dir)

	ga = GeneticAlgTSP.GeneticAlgTSP(config.input_path)

	res = ga.iterate(400)

	print(res)


if __name__ == '__main__':
	main()

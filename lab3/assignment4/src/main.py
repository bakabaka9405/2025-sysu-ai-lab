import config
import util
import crossover
import GeneticAlgTSP


def main():
	if config.seed != 0:
		util.set_random_seed(config.seed)

	if config.output_path_dir is not None:
		util.create_output_dir(config.output_path_dir)

	ga = GeneticAlgTSP.GeneticAlgTSP(config.input_path, crossover.__dict__[config.crossover])

	res = ga.iterate(10)

	print(res)


if __name__ == '__main__':
	main()

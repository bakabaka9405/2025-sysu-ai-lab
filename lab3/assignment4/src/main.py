import config
import util
import GeneticAlgTSP


def main():
	if config.seed != 0:
		util.set_random_seed(config.seed)

	if config.output_path_dir:
		util.create_output_dir(config.output_path_dir)

	ga = GeneticAlgTSP.GeneticAlgTSP(config.input_path)

	res = ga.iterate(1000)

	print(res)


if __name__ == '__main__':
	main()

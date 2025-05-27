import argparse
import gymnasium
from argument import dqn_arguments
from agent_dir.agent_dqn import AgentDQN
import random
import numpy
import torch


def parse():
	parser = argparse.ArgumentParser()
	parser = dqn_arguments(parser)
	args = parser.parse_args()
	return args


def run(args: argparse.Namespace):
	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	random.seed(args.seed)
	numpy.random.seed(args.seed)
	env = gymnasium.make('CartPole-v1', render_mode='rgb_array')
	agent = AgentDQN(env, args)
	agent.run()


if __name__ == '__main__':
	args = parse()
	run(args)

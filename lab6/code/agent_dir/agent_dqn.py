import random
from collections import deque
import numpy as np
import torch
from torch import nn, optim
from agent_dir.agent import Agent
from gymnasium import Env
from argparse import Namespace
from numpy.typing import NDArray
import math
import os
from .util import img2video, make_frame, moving_average
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super().__init__()

		self.feature_layer = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
		)

		# self.fc = nn.Linear(hidden_size, output_size)  # no dueling DQN

		self.value_layer = nn.Sequential(
			nn.Linear(hidden_size, hidden_size // 2),
			nn.ReLU(),
			nn.Linear(hidden_size // 2, 1),
		)

		self.advantage_layer = nn.Sequential(
			nn.Linear(hidden_size, hidden_size // 2),
			nn.ReLU(),
			nn.Linear(hidden_size // 2, output_size),
		)

	def forward(self, inputs):
		x = self.feature_layer(inputs)

		# return self.fc(x) # no dueling DQN

		value = self.value_layer(x)

		advantage = self.advantage_layer(x)

		q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

		return q_values


class ReplayBuffer:
	buffer: deque

	def __init__(self, buffer_size: int):
		self.buffer = deque(maxlen=buffer_size)

	def __len__(self):
		return len(self.buffer)

	def push(self, *transition):
		self.buffer.append(transition)

	def sample(self, batch_size: int):
		states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
		return np.array(states), actions, rewards, np.array(next_states), dones

	def clean(self):
		self.buffer.clear()


class AgentDQN(Agent):
	device: torch.device
	env: Env
	episodes: int
	state_dim: int
	action_dim: int
	hidden_size: int
	buffer_size: int
	batch_size: int
	gamma: float
	grad_norm_clip: float
	lr: float
	eps: float
	eps_start: float
	eps_end: float
	eps_decay: int
	update_freq: int
	eval_net: QNetwork
	target_net: QNetwork
	optimizer: optim.Optimizer
	memory: ReplayBuffer
	criterion: nn.Module
	steps_done: int
	eps: float

	record_rewards: list[float] = []
	record_steps: list[int] = []
	record_frames: list
	output_dir: str

	save_model: bool
	test: bool

	def __init__(self, env: Env, args: Namespace):
		super().__init__(env)
		self.device = torch.device(args.device)
		self.env = env
		self.episodes = args.episodes

		self.state_dim = env.observation_space.shape[0]  # type:ignore
		self.action_dim = env.action_space.n  # type:ignore

		self.hidden_size = args.hidden_size
		self.buffer_size = args.buffer_size
		self.batch_size = args.batch_size
		self.gamma = args.gamma
		self.grad_norm_clip = args.grad_norm_clip
		self.lr = args.lr
		self.eps_start = args.eps_start
		self.eps_end = args.eps_end
		self.eps_decay = args.eps_decay
		self.eps = self.eps_start
		self.update_freq = args.update_freq

		self.eval_net = QNetwork(self.state_dim, self.hidden_size, self.action_dim).to(self.device)

		if args.test:
			weight = torch.load(args.model_path, map_location=self.device)
			self.eval_net.load_state_dict(weight)

		self.target_net = QNetwork(self.state_dim, self.hidden_size, self.action_dim).to(self.device)
		self.target_net.load_state_dict(self.eval_net.state_dict())
		self.target_net.eval()

		self.optimizer = optim.AdamW(self.eval_net.parameters(), lr=self.lr, weight_decay=1e-4)
		self.memory = ReplayBuffer(self.buffer_size)
		self.criterion = nn.MSELoss()

		self.record_frames = []
		self.record_rewards = []
		self.record_steps = []
		self.output_dir = args.output_dir

		self.save_model = args.save_model
		self.test = args.test

		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

		def lr_func(episode: int):
			return 0.49 * (math.cos(episode / self.episodes * math.pi) + 1)

		self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_func)
		self.steps_done = 0
		self.test = args.test

	def init_game_setting(self):
		pass

	def optimize_model(self):
		assert len(self.memory) >= self.batch_size

		states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

		states = torch.tensor(states, dtype=torch.float32).to(self.device)
		actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
		rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
		next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
		dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

		q_eval = self.eval_net(states).gather(1, actions)

		with torch.no_grad():
			# use ddqn
			next_actions = self.eval_net(next_states).argmax(dim=1, keepdim=True)
			q_target = self.target_net(next_states).gather(1, next_actions)
			q_target = rewards + self.gamma * q_target * (1 - dones)

			# use dqn
			# q_next = self.target_net(next_states).max(dim=1).values.unsqueeze(1)
			# q_target = rewards + self.gamma * q_next * (1 - dones)

		loss = self.criterion(q_eval, q_target)

		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_value_(self.eval_net.parameters(), self.grad_norm_clip)
		self.optimizer.step()

		if self.steps_done % self.update_freq == 0:
			self.target_net.load_state_dict(self.eval_net.state_dict())

		self.eps = self.eps_end + math.exp(-(self.steps_done / self.eps_decay)) * (self.eps_start - self.eps_end)

	def select_action(self, observation: NDArray[np.float32], test=True) -> int:
		if not test and random.random() < self.eps:
			return self.env.action_space.sample()
		else:
			self.eval_net.eval()
			state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
			with torch.no_grad():
				q_eval = self.eval_net(state)
			self.eval_net.train()
			return q_eval.argmax(dim=1).item()

	def run(self):
		if self.test:
			total_rewards = 0
			state, _ = self.env.reset()
			done = False
			episode_steps = 0
			while not done:
				action = self.select_action(state, test=True)
				next_state, reward, terminated, truncated, _ = self.env.step(action)
				done = terminated or truncated
				state = next_state
				total_rewards += float(reward)
				episode_steps += 1
				self.record_frames.append(make_frame(self.env.render(), 1, episode_steps))  # type:ignore
			print(f'Test Episode: Total Reward: {total_rewards}, Steps: {episode_steps}')
			img2video(self.record_frames, f'{self.output_dir}/test.mp4', fps=60)

		else:
			for episodes in range(self.episodes):
				state, _ = self.env.reset(seed=random.randint(0, 2**31 - 1))
				self.env.action_space.seed(random.randint(0, 2**31 - 1))
				episode_steps = 0
				pos = 0
				ang = 0
				buf = []
				while True:
					action = self.select_action(state, test=False)
					next_state, reward, terminated, truncated, _ = self.env.step(action)
					pos = next_state[0]
					ang = next_state[2]
					done = terminated or truncated
					reward = math.exp(-(0 if abs(pos) < 0.25 else abs(abs(pos) - 2.4)) / 2) * math.cos(0 if abs(ang) < 0.05 else ang) ** 2 / 2

					buf.append([state, action, reward, next_state, done])
					state = next_state
					self.steps_done += 1
					episode_steps += 1

					if len(self.memory) > self.batch_size:
						self.optimize_model()

					if (episodes + 1) % 10 == 0:
						self.record_frames.append(make_frame(self.env.render(), episodes + 1, episode_steps))  # type:ignore

					if done:
						break
				if len(buf) != 500:
					for i in range(min(10, len(buf))):
						buf[-1 - i][2] = -0.5
				elif abs(pos) < 0.25:
					for i in buf:
						if abs(i[2] - 0.5) < 1e-6:
							i[2] *= 2
				episode_reward = sum(transition[2] for transition in buf)
				for transition in buf:
					self.memory.push(*transition)
				self.optimizer.step()
				self.scheduler.step()
				print(
					f'Episode {episodes + 1}: Total Reward: {episode_reward:.2f}, Steps: {episode_steps}, Pos: {pos:.2f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}, Epsilon: {self.eps:.2e}, Steps Done: {self.steps_done}, Buffer: {len(self.memory)}'
				)
				self.record_rewards.append(episode_reward)
				self.record_steps.append(episode_steps)

			img2video(self.record_frames, f'{self.output_dir}/train.mp4', fps=60)

			if self.save_model:
				torch.save(self.eval_net.state_dict(), f'{self.output_dir}/model.pth')
				print(f'Model saved to {self.output_dir}/model.pth')

			self.plot()

	def plot(self):
		plt.figure(figsize=(10, 6))
		plt.plot(self.record_rewards, label='Rewards')
		plt.plot(self.record_steps, label='Steps')
		plt.plot(moving_average(self.record_rewards, 50), label='Rewards (MA)', linestyle='--')
		plt.plot(moving_average(self.record_steps, 50), label='Steps (MA)', linestyle='--')
		plt.xlabel('Episodes')
		plt.ylabel('Value')
		plt.title('Training Progress')
		plt.legend()
		plt.grid()
		plt.savefig(f'{self.output_dir}/training_progress.png')
		plt.show()

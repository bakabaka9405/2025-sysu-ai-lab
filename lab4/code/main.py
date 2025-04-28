import random
import numpy as np
from numpy.typing import NDArray
from typing import cast
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)


class MLP:
	weights: list[NDArray[np.float64]]  # 权重
	biases: list[NDArray[np.float64]]  # 偏置
	z: list[NDArray[np.float64]]
	a: list[NDArray[np.float64]]

	def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
		self.layers_dim = [input_size] + hidden_sizes + [output_size]
		self.num_layers = len(self.layers_dim) - 1

		self.weights = []
		self.biases = []

		for i in range(self.num_layers):
			self.weights.append(np.random.randn(self.layers_dim[i], self.layers_dim[i + 1]) * np.sqrt(2.0 / self.layers_dim[i]))
			self.biases.append(np.zeros((1, self.layers_dim[i + 1])))

		self.z = [np.zeros(1)] * self.num_layers
		self.a = [np.zeros(1)] * (self.num_layers + 1)

	def relu(self, x: NDArray[np.float64]):
		return np.maximum(0, x)

	def relu_derivative(self, x: NDArray[np.float64]):
		return np.where(x > 0, 1, 0)

	def forward(self, x: NDArray[np.float64]):
		self.a[0] = x

		for i in range(self.num_layers - 1):
			self.z[i] = np.dot(self.a[i], self.weights[i]) + self.biases[i]
			self.a[i + 1] = self.relu(self.z[i])

		self.z[-1] = np.dot(self.a[-2], self.weights[-1]) + self.biases[-1]
		self.a[-1] = self.z[-1]

		return self.a[-1]

	def backward(self, x: NDArray[np.float64], y: NDArray[np.float64], lr: float):
		m = x.shape[0]  # 样本数量

		# 输出层的梯度
		dz = self.a[-1] - y

		dW: list = [None] * self.num_layers
		db: list = [None] * self.num_layers

		for i in range(self.num_layers - 1, -1, -1):
			if i == self.num_layers - 1:
				dW[i] = (1 / m) * np.dot(self.a[i].T, dz)
				db[i] = (1 / m) * np.sum(dz, axis=0, keepdims=True)
			else:
				dz = np.dot(dz, self.weights[i + 1].T) * self.relu_derivative(self.z[i])
				dW[i] = (1 / m) * np.dot(self.a[i].T, dz)
				db[i] = (1 / m) * np.sum(dz, axis=0, keepdims=True)

		for i in range(self.num_layers):
			# 更新权重和偏置
			self.weights[i] -= lr * dW[i]
			self.biases[i] -= lr * db[i]

	def cosine_annealing(self, epoch: int, warmup_epochs: int, min_lr: float, base_lr: float) -> float:
		# 余弦退火
		if epoch < warmup_epochs:
			return max(min_lr, base_lr * (epoch / warmup_epochs))
		else:
			return max(
				min_lr,
				base_lr * (1 + np.cos((epoch - warmup_epochs) / (num_epochs - warmup_epochs) * np.pi)) / 2,
			)

	def train(
		self,
		x_train: NDArray[np.float64],
		y_train: NDArray[np.float64],
		x_val: NDArray[np.float64],
		y_val: NDArray[np.float64],
		base_lr: float,
		min_lr: float,
		num_epochs: int,
		warmup_epochs: int = 0,
	) -> tuple[list[float], list[float]]:
		losses = []
		val_losses = []

		for epoch in range(num_epochs):
			# 前向传播
			y_pred = self.forward(x_train)
			# 计算 L_{MSE}
			loss = np.mean(np.square(y_pred - y_train))
			losses.append(loss)

			# 反向传播
			self.backward(x_train, y_train, self.cosine_annealing(epoch, warmup_epochs, min_lr, base_lr))

			y_pred_val = self.forward(x_val)
			val_loss = np.mean(np.square(y_pred_val - y_val))
			val_losses.append(val_loss)
			# val_loss = 0
			if (epoch + 1) % 10 == 0:
				print(
					f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, lr: {self.cosine_annealing(epoch, warmup_epochs, min_lr, base_lr):.4f}'
				)

		return losses, val_losses


class AutoEncoder(MLP):
	encode_output_layer: int

	def __init__(
		self,
		input_size: int,
		encoder_hidden_sizes: list[int],
		decoder_hidden_sizes: list[int],
		output_size: int,
	):
		super().__init__(
			input_size,
			encoder_hidden_sizes + [output_size] + decoder_hidden_sizes,
			input_size,
		)

		self.encode_output_layer = len(encoder_hidden_sizes)  # 编码器的输出层索引

	def forward(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
		super().forward(x)

		return self.z[self.encode_output_layer], self.a[-1]  # 返回编码器输出和解码器输出

	def train(  # type:ignore
		self,
		x_train: NDArray[np.float64],
		x_val: NDArray[np.float64],
		base_lr: float,
		min_lr: float,
		num_epochs: int,
		warmup_epochs: int = 0,
	) -> tuple[list[float], list[float]]:
		losses = []
		val_losses = []

		for epoch in range(num_epochs):
			# 前向传播
			encoded, decoded = self.forward(x_train)
			# 计算 L_{MSE}
			loss = np.mean(np.square(decoded - x_train))
			losses.append(loss)

			# 反向传播
			self.backward(x_train, x_train, self.cosine_annealing(epoch, warmup_epochs, min_lr, base_lr))

			encoded, decoded = self.forward(x_val)
			val_loss = np.mean(np.square(decoded - x_val))
			val_losses.append(val_loss)
			if (epoch + 1) % 10 == 0:
				print(
					f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, lr: {self.cosine_annealing(epoch, warmup_epochs, min_lr, base_lr):.4f}'
				)

		return losses, val_losses


# 1. 定义模型参数
input_size = 4
autoencoder_encoder_hidden_sizes = [16]
autoencoder_decoder_hidden_sizes = [16]
autoencoder_output_size = 2
hidden_sizes = [16, 64, 16]
output_size = 1
autoencoder_base_learning_rate = 0.1
regression_base_learning_rate = 0.5
autoencoder_warmup_epochs = 300
regression_warmup_epochs = 1500
num_epochs = 10000

# 2. 创建模型
autoencoder = AutoEncoder(
	input_size,
	autoencoder_encoder_hidden_sizes,
	autoencoder_decoder_hidden_sizes,
	autoencoder_output_size,
)
regression = MLP(input_size, hidden_sizes, output_size)

# 3. 读取训练数据
mlp_data = pd.read_csv(r'c:\Users\WanYa\Desktop\9-MLP&DT-机器学习基础课件及作业\MLP_data_1.csv')
X_train_raw = mlp_data.iloc[:, :-1].values
y_train_raw = cast(NDArray, mlp_data.iloc[:, -1].values).reshape(-1, 1)

# 4. 数据标准化
x_mean, x_std = np.mean(X_train_raw, axis=0), np.std(X_train_raw, axis=0)
x_train = (X_train_raw - x_mean) / x_std
y_mean, y_std = np.mean(y_train_raw), np.std(y_train_raw)
y_train = (y_train_raw - y_mean) / y_std

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# 5. 训练模型
losses1 = autoencoder.train(
	x_train,
	x_val,
	autoencoder_base_learning_rate,
	0.01,
	num_epochs,
	autoencoder_warmup_epochs,
)

losses2 = regression.train(
	x_train,
	y_train,
	x_val,
	y_val,
	regression_base_learning_rate,
	0.05,
	num_epochs,
	regression_warmup_epochs,
)

# 6. 画图
plt.autoscale()
outputs = regression.forward(x_train) * y_std + y_mean

# 画一条 y=x 的线
plt.plot(
	[np.min(y_train) * y_std + y_mean, np.max(y_train) * y_std + y_mean],
	[np.min(y_train) * y_std + y_mean, np.max(y_train) * y_std + y_mean],
	'r--',
	lw=2,
)

# 散点图
plt.scatter(outputs, y_train * y_std + y_mean, s=1, color='blue', label='Predicted vs True')

# 标签
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.title('MLP Model Predictions vs True Values')
plt.legend()
plt.grid()
plt.show()

# 输出降维和预测结果到 csv，用于 matlab 可视化
# pyplot 画 3d 图会卡死，费解
encoded = autoencoder.forward(x_train)[0]
df = pd.DataFrame(np.hstack((encoded, outputs)), columns=['feature1', 'feature2', 'predicted'])
df.to_csv('C:/Temp/MLP_output.csv', index=False)

# 降维器损失曲线
plt.figure()
plt.plot(losses1[0], label='Training Loss')
plt.plot(losses1[1], label='Validation Loss')
plt.title('AutoEncoder Training Loss')
plt.ylim(0,1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# 回归器损失曲线
plt.figure()
plt.plot(losses2[0], label='Training Loss')
plt.plot(losses2[1], label='Validation Loss')
plt.title('Regression Training Loss')
plt.ylim(0,1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

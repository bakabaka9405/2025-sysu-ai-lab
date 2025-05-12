import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from torchvision.transforms import v2 as transforms
from torchvision.io import decode_image, ImageReadMode
import math
from pathlib import Path
from matplotlib import pyplot as plt

torch.random.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Block(nn.Module): # 残差块
	def __init__(
		self,
		in_planes: int,
		out_planes: int,
		stride: int = 1,
		need_downsample: bool = False,
	) -> None:
		super().__init__()
		self.conv1 = nn.Conv2d(
			in_planes,
			out_planes,
			kernel_size=3,
			stride=stride,
			padding=1,
			bias=False,
		)
		self.bn1 = nn.BatchNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(
			out_planes,
			out_planes,
			kernel_size=3,
			padding=1,
			bias=False,
		)
		self.bn2 = nn.BatchNorm2d(out_planes)

		# 输入和输出维度不同的情况下需要对捷径下取样
		self.downsample = (
			nn.Sequential(
				nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_planes),
			)
			if need_downsample
			else None
		)

	def forward(self, x: Tensor) -> Tensor:
		identity = x  # shortcut

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	def __init__(
		self,
		num_classes: int = 1000,
	) -> None:
		super().__init__()
		self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 64, 112, 112
		self.bn = nn.BatchNorm2d(64)  # 卷积层后全部接归一化层，可以关闭 bias
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layers = nn.Sequential(
			Block(64, 64),  # 64, 56, 56
			Block(64, 64),
			Block(64, 128, stride=2, need_downsample=True),  # 128, 28, 28
			Block(128, 128),
			Block(128, 256, stride=2, need_downsample=True),  # 256, 14, 14
			Block(256, 256),
			Block(256, 512, stride=2, need_downsample=True),  # 512, 7, 7
			Block(512, 512),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 512, 1, 1
		self.fc = nn.Linear(512, num_classes)

	def forward(self, x: Tensor) -> Tensor:
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layers(x)

		x = self.avgpool(x)
		x = torch.flatten(x, 1)  # 512,1,1 -> 512
		x = self.fc(x)

		return x


class ZeroOneNormalize:
	"""
	decode_image 默认读取为 uint8 类型，范围为 [0, 255]
	需要手动转到 [0,1] 的范围
	"""

	def __call__(self, tensor: torch.Tensor):
		return tensor.float().div(255)


class_dict: dict[str, int] = {}  # {class_name: class_index}


class CostomDataset(torch.utils.data.Dataset):
	data: list[Tensor]
	labels: list[int]
	transform: transforms.Compose

	def __init__(self, device: torch.device, path: Path, transform: transforms.Compose):
		global class_dict
		class_dict = {}

		self.data = []
		self.labels = []
		for i, folder in enumerate(path.iterdir()):
			class_dict[folder.name] = i
			for file in folder.iterdir():
				# decode_image 读进来就是 Tensor 不需要经历从 PIL 转 Tensor 的过程
				# 空间充足的情况下直接放进显存加速后续读取
				self.data.append(decode_image(str(file), ImageReadMode.RGB).to(device))
				self.labels.append(i)
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		x = self.data[idx]
		y = self.labels[idx]

		x = self.transform(x)

		return x, y


def main():
	root = Path(r'cnn图片')

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Using device: {device}')

	model = ResNet(num_classes=5).to(device)

	warmup_epochs = 10
	epochs = 100
	lr = 1e-4

	def lr_func(epoch: int):
		return min((epoch + 1) / (warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / epochs * math.pi) + 1))

	criterion = CrossEntropyLoss()

	train_transform = transforms.Compose(
		[
			transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
			transforms.RandomResizedCrop(224, scale=(0.8, 1)),
			ZeroOneNormalize(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		]
	)

	test_transform = transforms.Compose(
		[
			transforms.Resize((224, 224)),
			ZeroOneNormalize(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		]
	)

	train_dataset = CostomDataset(device, root / 'train', transform=train_transform)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

	lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

	scaler = torch.GradScaler()  # 混合精度训练的 loss 缩放器

	train_losses = []
	train_accs = []

	# 训练模型
	for epoch in range(epochs):
		train_loss = 0.0
		train_acc = 0.0
		model.train()
		for inputs, labels in train_loader:
			labels = labels.to(device)

			optimizer.zero_grad()

			with torch.autocast('cuda'):  # 混合精度训练
				outputs = model(inputs)
				loss = criterion(outputs, labels)
			scaler.scale(loss).backward()  # 反向传播
			scaler.step(optimizer)
			scaler.update()

			pred = outputs.argmax(dim=1)
			correct = (pred == labels).sum().item()
			acc = correct / len(labels)

			train_loss += loss.item()
			train_acc += acc

		lr_scheduler.step()

		print(
			f'Epoch {epoch + 1}/{epochs}'
			f' LR: {optimizer.param_groups[0]["lr"]:.6f}'
			f' Train Loss: {train_loss / len(train_loader):.4f}'
			f' Train Acc: {train_acc / len(train_loader):.4f}'
		)

		train_losses.append(train_loss / len(train_loader))
		train_accs.append(train_acc / len(train_loader))

	# 准备测试数据
	test_img: list[Tensor] = []

	test_label_list: list[int] = []

	for img_path in (root / 'test').iterdir():
		test_img.append(test_transform(decode_image(str(img_path), ImageReadMode.RGB)))
		test_label_list.append(class_dict[img_path.name.split('.')[0][:-2]])  # 提取文件名中的中药拼音并转换为索引

	test_labels: Tensor = torch.tensor(test_label_list).to(device)

	test_inputs = torch.stack(test_img, dim=0).to(device)

	# 测试模型
	with torch.no_grad():
		model.eval()
		outputs = model(test_inputs)
		loss = criterion(outputs, test_labels)

		pred = outputs.argmax(dim=1)
		correct = (pred == test_labels).sum().item()
		acc = correct / len(test_inputs)

		print(f'Test Loss: {loss.item():.4f}' f' Test Acc: {acc:.4f}')

	# 生成两张图，分别表示训练集和验证集的损失和准确率
	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.plot(train_losses, label='Train Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Loss')
	plt.legend()
	plt.subplot(1, 2, 2)
	plt.plot(train_accs, label='Train Acc')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title('Accuracy')
	plt.legend()
	plt.savefig('loss_acc.png')


if __name__ == '__main__':
	main()

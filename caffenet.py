import torch.nn as nn
import torch
from torch.autograd import Variable

# Parameter output_dim allows for finetuning on different datasets, default is UCF-101
# Input_spatial_size = (224, 224)
class CaffeNet_BN(nn.Module):
	def __init__(self, output_layer='fc8', num_classes = 101):
		super(CaffeNet_BN, self).__init__()
		self.output_layer = output_layer
		self.num_classes = num_classes
		self.features = nn.Sequential(
			nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(96),
			# 55x55
			nn.MaxPool2d(kernel_size=3, stride=2),
			# 27x27
			nn.Conv2d(96, 256, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(256),
			nn.MaxPool2d(kernel_size=3, stride=2),
			# 13x13
			nn.Conv2d(256, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(384),
			nn.Conv2d(384, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(384),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(256),
			nn.MaxPool2d(kernel_size=3, stride=2)
			# 6x6
		)

		self.fc6 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True))
		self.fc7 = nn.Sequential(#nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096))
		if self.output_layer == 'fc6':
			for p in self.fc7.parameters():
				p.requires_grad_(False)
		self.fc8 = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes))
		if self.output_layer == 'fc7':
			for p in self.fc8.parameters():
				p.requires_grad_(False)


	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.fc6(x)
		if self.output_layer == 'fc6':
			return x
		x = self.fc7(x)
		if self.output_layer == 'fc7':
			return x
		x = self.fc8(x)
		if self.output_layer == 'fc8':
			return x


	def reinit_fc(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.reset_parameters()            


	# Loads the conv layers after pre-training from net_features.pkl
	# Input tensors should be globally normalized to have zero mean and unit variance
	def get_conv_weights(self):
		state_dict_features = torch.load('net_features.pkl')
		self.features.load_state_dict(state_dict_features)






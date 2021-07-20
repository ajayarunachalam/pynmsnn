__author__ = 'Ajay Arunachalam'
__version__ = '0.0.1'
__date__ = '17.07.2021'

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class NonSpikingNeuralNetwork(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(NonSpikingNeuralNetwork, self).__init__()
		self.layer1 = nn.Linear(input_dim, 100)
		self.layer2 = nn.Linear(100, output_dim)

	def forward(self, x, is_2D=True):
		x = x.view(x.size(0), -1)  # 1D for FC
		x = F.relu(self.layer1(x))
		x = self.layer2(x)
		return F.log_softmax(x, dim=-1)

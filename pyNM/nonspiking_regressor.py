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
	def __init__(self, input_dim, hidden_dim_l1, hidden_dim_l2, hidden_dim_l3, output_dim=1):
		super(NonSpikingNeuralNetwork, self).__init__()
		self.layer1 = nn.Linear(input_dim, hidden_dim_l1)
		self.layer2 = nn.Linear(hidden_dim_l1, hidden_dim_l2)
		self.layer3 = nn.Linear(hidden_dim_l2, hidden_dim_l3)
		self.layerout = nn.Linear(hidden_dim_l3, output_dim)
		self.relu = nn.ReLU()
		#self.dropout = nn.Dropout(p=0.1)
        #self.batchnorm1 = nn.BatchNorm1d(64)
        #self.batchnorm2 = nn.BatchNorm1d(64)

	def forward(self, inputs):
		x = self.relu(self.layer1(inputs))
		x = self.relu(self.layer2(x))
		x = self.relu(self.layer3(x))
		#x = self.dropout(x)
		x = self.layerout(x)
		return (x)

	def predict(self, test_inputs):
		x = self.relu(self.layer1(test_inputs))
		x = self.relu(self.layer2(x))
		x = self.relu(self.layer3(x))
		#x = self.dropout(x)
		x = self.layerout(x)
		return (x)



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
    def __init__(self, input_dim, output_dim=1):
        super(NonSpikingNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64) 
        self.layer2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, output_dim) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x
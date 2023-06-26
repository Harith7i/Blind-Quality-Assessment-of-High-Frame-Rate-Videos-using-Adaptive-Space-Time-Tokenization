import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
import os 
import argparse
import random
from collections import OrderedDict
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from random import randint

#Defining Spatioal Pooling unit
class lstm(nn.Module):
  def __init__(self, n_input, n_outputs):
    super().__init__()
    self.lstm = nn.LSTM(n_input, n_outputs, 1, batch_first =True)
    self.n_input =n_input
    self.n_outputs = n_outputs
  def forward(self, x):
    h0 = torch.randn(1, x.shape[0], self.n_outputs).to(device)
    c0 = torch.randn(1, x.shape[0], self.n_outputs).to(device)
    output, hn = self.lstm(x, (h0, c0))
    return(output[:,-1,:])
    
class gru(nn.Module):
  def __init__(self, n_input, n_outputs):
    super().__init__()
    self.gru = nn.GRU(n_input, n_outputs, 1, batch_first =True)
    self.n_input =n_input
    self.n_outputs = n_outputs

  def forward(self, x):
    h0 = torch.randn(1, x.shape[0], self.n_outputs).to(device)
    output, hn = self.gru(x, h0)
    return(output[:,-1,:])


class rnn(nn.Module):
  def __init__(self, n_input, n_outputs):
    super().__init__()
    self.rnn = nn.RNN(n_input, n_outputs, 1, batch_first =True)
    self.n_input =n_input
    self.n_outputs = n_outputs
  def forward(self, x):
    h0 = torch.randn(1,  x.shape[0], self.n_outputs).cuda()
    output, hn = self.rnn(x, h0)
    return(output[:,-1,:].cuda())

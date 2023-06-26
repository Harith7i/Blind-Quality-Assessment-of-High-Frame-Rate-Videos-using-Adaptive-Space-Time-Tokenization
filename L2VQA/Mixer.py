import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
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


class MiX(nn.Module):
  def __init__(self, FE, model1, model2):
    super().__init__()
    self.m0 = FE
    self.m1 = model1
    self.m2 = model2
  def forward(self, input_video_path):
    frames = TemporalCrop(input_video_path)
    x = self.m0(frames[0])
    for i in l[1:]:
      x = torch.cat((x,self.m0(i)), dim =0)
    x = self.m1(x)
    x = x.unsqueeze(axis=0)
    x = self.m2(x)
    return x
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

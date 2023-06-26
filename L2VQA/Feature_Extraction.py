import torch
from torch import nn
import torchvision.models as models
import cv2
import numpy as np
import os 
import csv
import argparse
from random import randint
from einops import rearrange


#initialising ResNet50
pretrained_model = models.resnet50(pretrained=True)
for param in pretrained_model.parameters():
    param.requires_grad = False
Backbone = nn.Sequential(*list(pretrained_model.children())[:-2])

#Defining the SC function for patch extaction
def SpatialCrop(img):
  a1,b1,_=img.shape
  a1 = a1 - (a1%224)-226
  b1 = b1 - (b1%224)-226
  s1, s2=(a1)//5, 20+(b1)//10 #additional step
  
  if a1<1500:
    s1, s2 = 224,224
  a = a1//224
  b = b1//224
  t = torch.zeros(1,224,224,3)
  for x in range(0,a1,s1):
    for y in range(0,b1,s2):
      crop = img[x:x+224, y:y+224,:]
      aux = torch.from_numpy(crop).float().unsqueeze(0)
      t = torch.cat((t,aux), axis =0)
  t=t[1:,:,:,:]
  d = t.shape[0]
  while d<50:
    x = randint(0, a1-225)
    y = randint(0, b1-225)
    crop = img[x:224+x, y:224+y,:]
    t = torch.cat((t,torch.from_numpy(crop).float().unsqueeze(0)), axis =0)
    d = t.shape[0]
  return t

#Defining frame-wise feature extraction unit
class FT(nn.Module):
  def __init__(self):
    super().__init__()
    self.resnet = nn.Sequential(*list(pretrained_model.children())[:-2])
    self.pool = nn.AvgPool2d(( 7,7))
  def forward(self, x):
    x = SpatialCrop(x)
    x = rearrange(x, 'a b c d -> a d b c')
    x = self.resnet(x)
    x = self.pool(x)
    x = x.view(50,2048)
    return x.unsqueeze(0)
    
    
#Defining the TC function for patch extaction
def TemporalCrop(input_video_path):
	out = []
	final = []
	cap = cv2.VideoCapture(input_video_path)
	print(input_video_path)
	N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))	
	while(cap.isOpened()):
		
			ret, frame = cap.read()
			if ret:
				out.append(frame)
			else:
				break
	step = int(N/(fps+15)) # we add 15 to total fps so we don't exceed the total number of frames per video

	i = 0
	j = 0
	while i < fps :

		img = out[j]
		final.append(img)
		j = j +step
		i = i +1
	return(final)



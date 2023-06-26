import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np

class HFR_LIVE_Dataset_normalid(Dataset):
  def __init__(self, path):
    data1 = pd.read_csv(path)
    self.vid_names = data1.columns.tolist()[1:]
    self.MOS = data1.values.tolist()[0][1:]
    l = max(self.MOS)
    self.MOS = [i/l for i in self.MOS]
    self.std = data1.values.tolist()[1][1:]
    self.n_samples = len(data1.values.tolist()[1][1:])
  def __getitem__(self, index):
    return self.vid_names[index], self.MOS[index], self.std[index]
  def __len__(self):
    return self.n_samples
def dataloader(l):
  output =[]
  for i in range(len(l[0])):
    output.append((l[0][i], l[1][i], l[2][i]))
  return output

def loaddata_LIVE_Dataset(dataset_path):
  dataset = HFR_LIVE_Dataset_normalid(dataset_path)
  dataset = dataloader(dataset[:])
  indx = [i for i in range(16)]
  random.shuffle(indx)
  train_set, valid_set, test_set = [], [], []
  for i in indx[:-4]:
    train_set.extend(dataset[i*30:i*30+30])
  for i in indx[-4:-2]:
    valid_set.extend(dataset[i*30:i*30+30])
  for i in indx[-2:]:
    test_set.extend(dataset[i*30:i*30+30])
  return train_set, valid_set, test_set

data_set_path = '/content/gdrive/MyDrive/BVI/Participant Scores.csv'
class HFR_BVI_Dataset_normalid(Dataset):
  def __init__(self, path):
    data1 = pd.read_csv(path)
    self.vid_names = data1.columns.tolist()[1:]
    self.element = np.asarray(data1.values.tolist()[:])
    l=max([ np.mean(self.element[:,i]) for i in range(88)])
    li =[]
    for i in range(0,len(self.vid_names)-1):
      path = self.vid_names[i][:-2]
      li.append([path, np.mean(self.element[:,i])/l])
    self.element = li
    self.n_samples = len(data1.values.tolist()[1][1:])
  def __getitem__(self, index):
    return  self.element[index]
  def __len__(self):
    return self.n_samples
dataset = HFR_BVI_Dataset_normalid(data_set_path)




def loaddata_BVI_Dataset(dataset_path):
  dataset = HFR_BVI_Dataset_normalid(dataset_path)
  li = []
  for i in dataset[:]:
    li.append(i[0].split("-")[0])
    li = list(set(li))
  indx = [i for i in range(22)]
  random.shuffle(indx)
  train_set, valid_set, test_set = [], [], []
  for j in indx[:-4]:
    for i in dataset[:]:
      if li[j] == i[0].split("-")[0]:
        train_set.append(i)

  for i in indx[-4:-2]:
      for i in dataset[:]:
        if li[j] == i[0].split("-")[0]:
          valid_set.append(i)

  for i in indx[-2:]:
      for i in dataset[:]:
        if li[j] == i[0].split("-")[0]:
          test_set.append(i)
  return train_set, valid_set, test_set



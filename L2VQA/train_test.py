import argparse
import random
from collections import OrderedDict
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from random import randint
from scipy.stats import spearmanr,pearsonr
import scipy.stats
import sklearn.metrics
import math

from ViT import  ViT
from SpatialPooling import lstm, rnn, gru
from Feature_Extraction import FT, TemporalCrop
from Mixer import MiX





def train( model, train_load, valid_load, epochs, path_vid, dataset, weights_path):
  # Put the paths before the training 
  loss_epoch_train = []
  loss_epoch_val = []
  criterion = nn.MSELoss()  #Loss Function
  criterion.to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.0001)   # Optimizer
  # Here we start the train...
  for epoch in tqdm(range(epochs)):
    loss_train=[]
    loss_val=[]
    random.shuffle(train_load)
    model.train()
    for i in train_load:
      if dataset == 'LIVE':
      	path = path_vid+i[0]+'.webm'
      else dataset == 'BVI':
      	path = path_vid+i[0]+'.yuv'
      x = model(path)
      score = torch.Tensor([i[1]]).to(device)
      loss = criterion(score,x[0])
      loss_train.append(np.asarray(loss.cpu().detach().numpy()))
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    loss_epoch_train.append(sum(loss_train)/len(loss_train))

    #This is the validation phase
    random.shuffle(valid_load)
    model.eval()
    for i in valid_load:
      if dataset == 'LIVE':
      	path = path_vid+i[0]+'.webm'
      else dataset == 'BVI':
      	path = path_vid+i[0]+'.yuv'
      x = model(path)
      score = torch.Tensor([i[1]]).to(device)
      loss = criterion(score,x[0])
      loss_val.append(np.asarray(loss.cpu().detach().numpy())) 
    loss_epoch_val.append(sum(loss_val)/len(loss_val))


    torch.save(model.state_dict(), weights_path+spatial_pool+'_'+str(emb_dim)+'_'+str(epoch)+'.pth')
  point= [i+1 for i in range(epochs)]
  plt.plot(point , loss_epoch_train, label='train') 
  plt.plot(point , loss_epoch_val, label='val')
  plt.show()
  return
  
  
def test( model, test_set, epochs, path_vid, dataset, weights_path):
  model.load_state_dict(torch.load(weights_path)).to(device)
  result = []
  pristine =[]
  model.eval()
  for i in  test_set:
        valeur=np.load(os.path.join(path_to_data,i[0]+'.npy'))
        valeurc = torch.from_numpy(valeur).float().to(device)
        x = model(valeurc)
        score = torch.Tensor([i[1]]).to(device)
        result.append(np.asarray(x.cpu().detach().numpy())[0][0])
        pristine.append(i[1])

  l1 = [ i for i in pristine]
  l2 = [ i for i in result]
  points = [ fot i in range(len(l1))]
  plt.plot(point , l1, label='pristine')
  plt.plot(point , l2, label='pristine')
  plt.show()

  print('SROCC = ',spearmanr(result,pristine).correlation,'\n PLCC = ', scipy.stats.pearsonr(np.asarray(result),np.asarray(pristine))[0],
        '\n RMSE = ', math.sqrt(sklearn.metrics.mean_squared_error(l1, l2)))
  return 
  
  
  


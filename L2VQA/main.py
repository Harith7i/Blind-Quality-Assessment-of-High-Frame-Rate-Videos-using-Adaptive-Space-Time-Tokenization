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

from ViT import  ViT
from SpatialPooling import lstm, rnn, gru
from Feature_Extraction import FT, TemporalCrop
from Mixer import MiX
from train_test import train, test
from dataloader import loaddata_LIVE_Dataset, loaddata_BVI_Dataset

import argparse



def model(args.spatial_pooling, args.emb_dim):
	model0 = FT().to(device)
	if spatial_pool == 'rnn':
	    model1 = rnn(n_input=2048, n_outputs=emb_dim-1).to(device)
	elif spatial_pool == 'gru':
	    model1 = gru(n_input=2048, n_outputs=emb_dim-1).to(device)
	elif spatial_pool == 'lstm':
	    model1 = lstm(n_input=2048, n_outputs=emb_dim-1).to(device)
	    
	model1 = lstm(n_input=2048, n_outputs=95).to(device)
	model2 =  ViT(
	  input_size = emb_dim,
	  num_classes = 1,
	  dim = emb_dim,
	  depth = 6,
	  heads = 16,
	  mlp_dim = 2048,
	  nb_frames=120,
	  dropout = 0.1,
	  emb_dropout = 0.1
	  ).to(device)
	model= MiX(model0,model1, model2).to(device)
	return model

def end_to_end( dataset, dataset_path, args., spatial_pooling, emb_dim, path_vid, weights_folder, weights_path):

	if dataset == 'LIVE':
	    train_set, valid_set, test_set = loaddata_LIVE_Dataset(dataset_path)
	else dataset == 'BVI':
	    train_set, valid_set, test_set = loaddata_BVI_Dataset(dataset_path)
	
	train( model, train_set, valid_set, epochs, path_vid, dataset, weights_folder)
	test( model, test_set, epochs, path_vid, dataset, weights_path)



if ___name__ =='__main__':

	# Create the parser and add arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('-epochs', '--e', type=int, help='The total number of training epochs')
	parser.add_argument('-dataset_type', '--type', type=str, help='Set the variable to LIVE or BVI depending on your choice or training dataset')
	parser.add_argument('-dataset_csv_path', '--csv', type=str, help='Set the path to dataset CSV')
	parser.add_argument('-video_path', '--vp', type=str, help='Set the path to the videos directory')
	parser.add_argument('-spatial_pooling', '--sp', type=str, help='Set aptial pooling to lstm, rnn, or gru ')
	parser.add_argument('-embedding_size', '--emb_dim', type=int, help='Set the embedding sitz (the size of the transformer token) ')
	parser.add_argument('-weights_folder', '--wf', type=str, help='Set the path to the directory containing all training weights')
	parser.add_argument('-best_score_weights', '--bsw', type=str, help='Set the path to the .pth containtnig the best weights')

	args = parser.parse_args()
	
	# Call for end to end function
	end_to_end( args.type, args.csv, args.e, args.sp, args.size, args.vp, args.wf, args.bsw)
	return
















    

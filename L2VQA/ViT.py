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


  
#Defining Temporel Pooling unit
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
        
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x     
        
class ViT(nn.Module):
    def __init__(self, *, input_size, num_classes, dim, depth, heads, mlp_dim, nb_frames, pool = 'add', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.dim = dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(input_size-1, dim-1))
        self.pos_embedding = nn.Parameter(torch.randn(1, nb_frames, dim)) #31 if nb frames + 1

        self.add_token = nn.Parameter(torch.ones(1, 120, 1))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(120),
            nn.Linear(120, num_classes)
        )
        

    def forward(self, img ):
      a = img.shape[1]
      img = torch.cat((img, torch.zeros(1,120-a,self.dim-1).to(device)), axis=1)
      mask =torch.cat((torch.zeros((1,a,1),dtype=bool),torch.ones((1,120-a,1), dtype=bool)),axis=1).to(device)

      x = self.to_patch_embedding(img)
      b, n, _ = x.shape
      add_tokens = repeat(self.add_token, '1 n d -> b n d', b = b)
      mask_value = -1e-10
      add_tokens = add_tokens.masked_fill(mask, mask_value)
      x = torch.cat((add_tokens, x), dim=2)
      x += self.pos_embedding[:, :(n + 1)]
      x = self.dropout(x)
      x = self.transformer(x)  
      x = x.mean(dim = 1) if self.pool == 'mean' else x[:,:, 0]
      x = self.to_latent(x)
      x = self.mlp_head(x)
      return x        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

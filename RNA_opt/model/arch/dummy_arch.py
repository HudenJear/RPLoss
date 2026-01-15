import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F


class dummy_arch(nn.Module):
  def __init__(self, rna_mode=126,rna_max_length=2048,layers=6,head_legnth=1):
        super().__init__()
        self.linear_seq=nn.ModuleList()
        for ind in range(layers):
            self.linear_seq.append(nn.Linear(rna_max_length,rna_max_length))
            self.linear_seq.append(nn.LayerNorm(rna_max_length))
        self.head_layer=nn.Linear(rna_max_length,head_legnth)
        self.outpro=nn.Linear(rna_mode,head_legnth)
        
  
  def forward(self,x):
      # in (B,L,C) 128,2048,65 for example
      x=x.transpose(1, 2) # in (B,C,L)
      for layer in self.linear_seq:
        x=layer(x)
      x=self.head_layer(x)
      # return torch.nn.functional.normalize(x.transpose(1, 2),p=1.0,dim=2)# OUT (B,l,C)
      # return x.transpose(1, 2)
      x=self.outpro(x.squeeze())
      return torch.exp(x.squeeze())

class opt_dummy_arch(nn.Module):
  def __init__(self, rna_mode=126,rna_max_length=2048,layers=6,head_legnth=1024):
        super().__init__()
        self.linear_seq=nn.ModuleList()
        for ind in range(layers):
            self.linear_seq.append(nn.Linear(rna_max_length,rna_max_length))
            self.linear_seq.append(nn.LayerNorm(rna_max_length))
        self.head_layer=nn.Linear(rna_max_length,head_legnth)
        
  
  def forward(self,x):
      # in (B,L,C) 128,2048,65 for example
      x=x.transpose(1, 2) # in (B,C,L)
      for layer in self.linear_seq:
        x=layer(x)
      x=self.head_layer(x)
      return torch.nn.functional.normalize(x.transpose(1, 2),p=1.0,dim=2)# OUT (B,l,C)
      # return x.transpose(1, 2)
      # x=self.outpro(x.squeeze())
      # return torch.exp(x.squeeze())


class MFELoss_archBKUP(nn.Module):
  def __init__(self,rna_mode=126,rna_max_length=1024,layers=32,head_legnth=1):
    super().__init__()
    self.linear_seq=nn.ModuleList()
    for ind in range(layers):
      self.linear_seq.append(nn.Linear(rna_max_length,rna_max_length))
      self.linear_seq.append(nn.LayerNorm(rna_max_length))
    self.head_layer=nn.Linear(rna_max_length,head_legnth)
    self.outpro=nn.Linear(rna_mode,head_legnth)
    
  
  def forward(self,x):
    x=x.transpose(1, 2) # in (B,C,L)
    for layer in self.linear_seq:
      x=layer(x)
    x=self.head_layer(x)
    x=self.outpro(x.squeeze())
    return torch.exp(x.squeeze())
    # return x.squeeze()


class MFELoss_arch(nn.Module):
  def __init__(self,rna_mode=126,rna_max_length=1024,layers=8,layer_depth=4,head_legnth=1):
    super().__init__()
    self.linear_seq=nn.ModuleList()
    for ind in range(layers):
      for ind in range(layer_depth):
        self.linear_seq.append(nn.Linear(rna_max_length,rna_max_length))
      self.linear_seq.append(nn.LayerNorm(rna_max_length))
      self.linear_seq.append(nn.LeakyReLU(negative_slope=0.1))
    self.head_layer=nn.Linear(rna_max_length,head_legnth)
    self.outpro=nn.Linear(rna_mode,head_legnth)
    
  
  def forward(self,x):
    x=x.transpose(1, 2) # in (B,C,L)
    for layer in self.linear_seq:
      x=layer(x)
    x=self.head_layer(x)
    x=self.outpro(x.squeeze())
    return torch.exp(x.squeeze())
  
class MLPRNo_net(nn.Module):
  def __init__(self,rna_mode=126, rna_max_length=1024, hidden_size=2048, layers=8,layer_depth=4,drop_out=0.0):
    super().__init__()
    # input projection to hidden size
    self.input_proj=nn.Linear(rna_max_length,hidden_size)

    # MLP body
    self.linear_seq=nn.ModuleList()
    for ind in range(layers):
      for ind in range(layer_depth):
        self.linear_seq.append(nn.Linear(hidden_size,hidden_size))
      self.linear_seq.append(nn.LayerNorm(hidden_size))
      self.linear_seq.append(nn.LeakyReLU(negative_slope=0.1))
      if drop_out != 0:
         self.linear_seq.append(nn.Dropout(p=drop_out))
    self.mlp_end=nn.Linear(hidden_size,rna_max_length)

    # output layers
    # self.head_layer=nn.Linear(rna_max_length,head_legnth)
    
    # self.outpro=nn.Linear(rna_mode,head_legnth)
    
  
  def forward(self,x):
    x=x.transpose(1, 2) # in (B,C,L)
    x=self.input_proj(x)
    for layer in self.linear_seq:
      x=layer(x)
    fea=self.mlp_end(x)
    return fea.transpose(1, 2)
    

from torchvision.ops import MLP

class MFELossMLP_arch(nn.Module):
  def __init__(self,rna_mode=126, rna_max_length=1024, hidden_size=2048, layers=8,layer_depth=4,head_legnth=1,drop_out=0.0):
    super().__init__()
    # input projection to hidden size
    self.input_proj=nn.Linear(rna_max_length,hidden_size)
    

    # MLP body
    self.linear_seq=nn.ModuleList()
    for ind in range(layers):
      for ind in range(layer_depth):
        self.linear_seq.append(nn.Linear(hidden_size,hidden_size))
      self.linear_seq.append(nn.LayerNorm(hidden_size))
      self.linear_seq.append(nn.LeakyReLU(negative_slope=0.1))
      if drop_out != 0:
         self.linear_seq.append(nn.Dropout(p=drop_out))
    self.mlp_end=nn.Linear(hidden_size,rna_max_length)

    # output layers
    self.head_layer=nn.Linear(rna_max_length,head_legnth)
    
    self.outpro=nn.Linear(rna_mode,head_legnth)
    
  
  def forward(self,x):
    x=x.transpose(1, 2) # in (B,C,L)
    x=self.input_proj(x)
    for layer in self.linear_seq:
      x=layer(x)
    x=self.mlp_end(x)
    x=self.head_layer(x)
    x=self.outpro(x.squeeze())
    return torch.exp(x.squeeze())
    # return x.squeeze()
  
import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from torch.nn import L1Loss, MSELoss, SmoothL1Loss
from .utils import matrix_prefetch

class RNAProteinLoss(nn.Module):
    """Define feature matching loss for RNA

    Args:
        criterion (str): Support 'l1', 'l2', 'sl1'.
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.

    Usgae:
        RNA_vec_pred (tensor): Shall be (B,L,65/126). The 65(64+1) logits will be sequences without ambiguoius codon N.
        RNA_vec_gt (tensor): Shall be (B,L,65/126).
        weight(float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, criterion='l1', loss_weight=1.0, reduction='mean',codon_length='65'):
        super(RNAProteinLoss, self).__init__()
        if criterion == 'l1':
            self.loss_op = L1Loss(loss_weight, reduction)
        elif criterion == 'l2':
            self.loss_op = MSELoss(loss_weight, reduction)
        elif criterion == 'sl1':
            self.loss_op = SmoothL1Loss(loss_weight, reduction)
        else:
            raise ValueError(f'Unsupported loss mode: {criterion}. Supported ones are: l1|l2|sl1(SmoothL1Loss)')

        self.loss_weight = loss_weight


        if codon_length=='65':
            self.code_mat=matrix_prefetch(length=64)
            self.val_mat=matrix_prefetch(length=64).transpose_(dim0=0,dim1=1)
        elif codon_length=='126':
            self.code_mat=matrix_prefetch(length=125,mode='extend')
            self.val_mat=matrix_prefetch(length=125).transpose_(dim0=0,dim1=1)
        elif codon_length=='126extend':
            self.code_mat=matrix_prefetch(length=125,mode='extend')
            self.val_mat=matrix_prefetch(length=125,mode='extend').transpose_(dim0=0,dim1=1)
        # self.matrix64=matrix_prefetch(length=64)
        # self.matrix125=matrix_prefetch(length=125)
        if torch.cuda.is_available():
            self.code_mat=self.code_mat.cuda()
            self.val_mat=self.val_mat.cuda()


        # self.matrix64_expand=1
        # self.matrix125_expand=matrix_prefetch(length=125,mode='extend')
        # print(self.matrix64,self.matrix125,self.matrix125_expand)
        

    def forward(self, RNA_vec_pred, RNA_vec_gt):
        num_d = RNA_vec_pred.shape[2]
        if num_d!= RNA_vec_gt.shape[2]:
            raise ValueError(f'The Encoding of two codon do not comply: {RNA_vec_pred.shape}, {RNA_vec_gt.shape}')
        if RNA_vec_pred.shape[1]!= RNA_vec_gt.shape[1]:
            raise ValueError('The Length of two codon do not comply')
        # if mode not in ['precise','extend']:
        #   raise ValueError(f'Unsupported mode: {mode}. Supported ones are: precise/extend')
        
        # begin the loss compuattion
        # detect the correct coding region for gt sequence
        mat_E=RNA_vec_gt @ self.code_mat # (B,L,23)
        # print(mat_E.shape)
        mat_P=mat_E @ self.val_mat # (B,L,length)
        # print(mat_P.shape)
        mat_OP=torch.clamp(mat_P,min=0,max=1)*RNA_vec_pred # (B,L,length)
        vec_L=torch.sum(mat_OP,dim=2) # (B,L)
        loss = self.loss_op(vec_L,torch.ones(vec_L.shape).cuda()) # (B)

        return loss * self.loss_weight
    

if __name__=='__main__':
    
    import random

    loss= RNAProteinLoss(criterion='l1', loss_weight=1.0, reduction='mean',codon_length='65')
    for ind9 in range(10):
      input=torch.rand((4,16,512,65))
      tgt=torch.zeros((4,16,512,65))
      for ind0 in range(4):
        for ind1 in range(16):
          for indx in range(512):
              tgt[ind0,ind1,indx,random.randint(0,64)]=1
      # input=tgt
      
      # using p=1 to normlize the numbers
      input=torch.nn.functional.normalize(input,p=1.0,dim=3)
      # for ind1 in range(8):
      #   print(input[:,ind1,:,:].squeeze().shape)
      #   o_loss=loss(input[:,ind1,:,:].squeeze(),tgt[:,ind1,:,:].squeeze())
      #   print(o_loss)
      o_loss=loss(input,tgt)
      print(o_loss)

    loss= RNAProteinLoss(criterion='l1', loss_weight=1.0, reduction='mean',codon_length='65')
    for ind9 in range(10):
      input=torch.rand((4,512,65))
      tgt=torch.zeros((4,512,65))
      for ind0 in range(4):
        for indx in range(512):
            tgt[ind0,indx,random.randint(0,64)]=1
      
      input=tgt
      input=torch.nn.functional.normalize(input,p=1.0,dim=2)
      # print(input[0,1,:])
      # print(tgt[0,1,:])
      o_loss=loss(input,tgt)
      print(o_loss)

    # loss= RNAProteinLoss(criterion='sl1', loss_weight=1.0, reduction='mean',codon_length='126extend')
    # for ind9 in range(10):
    #   input=torch.rand((4,512,126))
    #   tgt=torch.zeros((4,512,126))
    #   for ind0 in range(4):
    #     for indx in range(512):
    #         # rand_ind=
    #         tgt[ind0,indx,random.randint(0,125)]=1
    #   input=tgt
    #   for ind0 in range(4):
    #     for indx in range(512):
    #         # rand_ind=
    #         input[ind0,indx,random.randint(0,125)]=1
    #   input=torch.nn.functional.normalize(input,p=1.0,dim=2)

    #   o_loss=loss(input,tgt)
    #   print(o_loss)

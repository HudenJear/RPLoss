import math,os
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from torch.nn import L1Loss, MSELoss, SmoothL1Loss

model_dict={
  '65': None,
  '126': '/data/huden/RNAloss/RNA_opt/model/RPLoss/MFEpretrainedModel/0_net_g_100000.pth',
}

class MFELoss_arch(nn.Module):
  def __init__(self, modelpath,rna_mode=126,rna_max_length=1024,layers=32,head_legnth=1):
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

class MFELoss(nn.Module):
  """Define Minimum free energy loss for RNA. The prediction of rna maybe not precise. This function is not recommanded to predict the MFE directly....

  Args:
    mode(str): Use ground truth or compuert the input RNA sequnence again for comparison. Support 'GroundTruth', 'ReCompute'.
    codon_length(str): Support '65', '126'
    criterion (str): Support 'l1', 'l2', 'sl1'.
    loss_weight (float): Loss weight. Default: 1.0.
    reduction (str): Specifies the reduction to apply to the output.
        Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
      

  Usgae:
    target (tensor): Shall be (B,L,65/126) tensor for 'ReCompute' mode, and (B,L) for 'GroundTruth' mode. 
    RNA_vec_gt (tensor): Shall be (B,L,65/126).
      
  """

  def __init__(self, mode='GroundTruth',codon_length='65',criterion='l1', loss_weight=1.0, reduction='mean'):
    super(MFELoss, self).__init__()

    self.mode=mode
    self.weight=loss_weight
    self.loss_model=MFELoss_arch(int(codon_length))

    if torch.cuda.is_available():
      self.loss_model=self.loss_model.cuda()
    
    if criterion == 'l1':
      self.loss_op = L1Loss(loss_weight, reduction)
    elif criterion == 'l2':
      self.loss_op = MSELoss(loss_weight, reduction)
    elif criterion == 'sl1':
      self.loss_op = SmoothL1Loss(loss_weight, reduction)
    else:
      raise ValueError(f'Unsupported loss mode: {criterion}. Supported ones are: l1|l2|sl1(SmoothL1Loss)')
    

    if model_dict[codon_length] is not None:
      if os.path.exists(model_dict[codon_length]):
        print("Pretrained MFE model found, will use its weight!!")
        
        state_dict = torch.load(model_dict[codon_length], map_location=lambda storage, loc: storage)
        # for k in ["total_ops", "total_params", "features.total_ops", "features.total_params", "classifier.total_ops", "classifier.total_params"]:
        #     state_dict['params'].pop(k)
        self.loss_model.load_state_dict(state_dict['params'])
    self.loss_model.eval()
    for param in self.parameters():
      param.requires_grad = False

  def forward(self,pred,gt):
    if self.mode=='GroundTruth':
      if pred.shape== gt.shape:
        raise ValueError(f'The mode and input do not comply: Need MFE gt input')
      gt_mfe=gt
      pred_mfe=self.loss_model(pred)      
      gt_mfe[gt_mfe<pred_mfe]=pred_mfe[gt_mfe<pred_mfe]
      return self.loss_op(pred_mfe,gt_mfe)*self.weight
    elif self.mode=='ReCompute':
      if pred.shape!= gt.shape:
        raise ValueError(f'The mode and input do not comply: Need gt Sequence input')
      # loss_num=self.loss_model(gt)-self.loss_model(pred)
      gt_mfe=self.loss_model(gt)
      pred_mfe=self.loss_model(pred)      
      gt_mfe[gt_mfe<pred_mfe]=pred_mfe[gt_mfe<pred_mfe]
      # print(pred_mfe)
      # print(gt_mfe)
      # if gt_mfe<pred_mfe:
      #   return self.loss_op(pred_mfe,pred_mfe)
      # else:
      return self.loss_op(pred_mfe,gt_mfe)*self.weight

      
    else:
      raise ValueError(f'The mode not recognised!!')
      return None
    
if __name__=='__main__':

  import random

  loss= MFELoss(mode='ReCompute',codon_length='126', criterion='l1', loss_weight=1.0, reduction='mean')
  for ind9 in range(10):
    input=torch.rand((4,1024,126)).cuda()
    tgt=torch.zeros((4,1024,126)).cuda()
    for ind0 in range(4):
        for indx in range(1024):
            tgt[ind0,indx,random.randint(0,64)]=1
    # input=tgt
    

    # using p=1 to normlize the numbers
    input=torch.nn.functional.normalize(input,p=1.0,dim=2)
    # for ind1 in range(8):
    #   print(input[:,ind1,:,:].squeeze().shape)
    #   o_loss=loss(input[:,ind1,:,:].squeeze(),tgt[:,ind1,:,:].squeeze())
    #   print(o_loss)
    o_loss=loss(input,tgt)
    print(o_loss)


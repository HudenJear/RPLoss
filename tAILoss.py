import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import L1Loss, MSELoss, SmoothL1Loss

from .utils import tRNA_gene_count,get_nc_dict

# we note that the tAI loss computation is now limited within several species: ecoli, yeast, human, rat, mouse, dog. More speceis can be accessed if there is need for specified variant in tRNADB. But the computation parameters are not optimized since there are too many species and we can not access the later version of generic or other more accurate tAI computation.

protein_dict={
    'A'	:	['GCT','GCC','GCA','GCG'],
    'R'	:	['CGT','CGC','CGA','CGG','AGA','AGG'],
    'N'	:	['AAT','AAC'],
    'D'	:	['GAT','GAC'],
    'C'	:	['TGT','TGC'],
    'E'	:	['GAA','GAG'],
    'Q'	:	['CAA','CAG'],
    'G'	:	['GGT','GGC','GGA','GGG'],
    'H'	:	['CAT','CAC'],
    'I'	:	['ATT','ATC','ATA'],
    'L'	:	['TTA','TTG','CTT','CTC','CTA','CTG'],
    'K'	:	['AAA','AAG'],
    'M'	:	['ATG'],
    'F'	:	['TTT','TTC'],
    'P'	:	['CCT','CCC','CCA','CCG'],
    'S'	:	['TCT','TCC','TCA','TCG','AGT','AGC'],
    'T'	:	['ACT','ACC','ACA','ACG'],
    'W'	:	['TGG'],
    'Y'	:	['TAT','TAC'],
    'V' :	['GTT','GTC','GTA','GTG'],
    '*': ['TAG','TGA','TAA'],
    'START': ['ATG'],
    'No-Codon': ['EEE'],
}

sstrain={
   'optimized': [0.0, 0.0, 0.0, 0.0, 0.41, 0.28, 0.9999, 0.68, 0.89],
   'standard': [0, 0, 0, 0, 0.5, 0.5, 0.75, 0.5, 0.5]
}


def get_compliment_trna(rna: str)        :
  codon=[]
  for cha in rna:
    if cha=='A':
      codon.insert(0,'T')
    elif cha=='T':
      codon.insert(0,'A')
    elif cha=='C':
      codon.insert(0,'G')
    elif cha=='G':
      codon.insert(0,'C')
    else:
       raise ValueError('Nuclie not support ATCG form of RNA codon')
  # print(codon[0]+codon[1]+codon[2])
  return codon[0]+codon[1]+codon[2]
    
            
def tAI_vector_prefetch(length,ss_list,trna_count_dict,reduction= 'protein'):
    
    if length==64:
        mat=torch.zeros((65,1))
        nc_dict=get_nc_dict(length)
    elif length==125:
        mat=torch.zeros((126,1))
        nc_dict=get_nc_dict(length)
    else:
        raise ValueError(f'Unsupported NC length mode: {length}. Supported ones are: 64/125')
    

    for alf1 in ['A', 'G', 'C', 'T']:
      for alf2 in ['A', 'G', 'C', 'T']:
        pos1=alf1+alf2+'T'
        pos2=alf1+alf2+'C'
        pos3=alf1+alf2+'A'
        pos4=alf1+alf2+'G'
        mat[nc_dict[pos1],0]=(1-ss_list[0])*float(trna_count_dict[get_compliment_trna(pos1)])+(1-ss_list[4])*float(trna_count_dict[get_compliment_trna(pos2)])
        mat[nc_dict[pos2],0]=(1-ss_list[1])*float(trna_count_dict[get_compliment_trna(pos2)])+(1-ss_list[5])*float(trna_count_dict[get_compliment_trna(pos1)])
        mat[nc_dict[pos3],0]=(1-ss_list[2])*float(trna_count_dict[get_compliment_trna(pos3)])+(1-ss_list[6])*float(trna_count_dict[get_compliment_trna(pos1)])
        mat[nc_dict[pos4],0]=(1-ss_list[3])*float(trna_count_dict[get_compliment_trna(pos4)])+(1-ss_list[7])*float(trna_count_dict[get_compliment_trna(pos3)])
    
    
    if reduction == 'protein':
      for pkey in  protein_dict:
        codon_list=protein_dict[pkey]
        sub_sequence=[]
        for cod in codon_list:
           sub_sequence.append(mat[nc_dict[cod],0])
        if max(sub_sequence)!=0:
          for cod in codon_list:
            mat[nc_dict[cod],0]=mat[nc_dict[cod],0]/max(sub_sequence)
        else:
          for cod in codon_list:
            mat[nc_dict[cod],0]=1
    # add other coding 
    elif reduction == 'all':
      mat=mat/max(mat)
      mat[[nc_dict['EEE'],nc_dict['ATG'],nc_dict['TAA'],nc_dict['TGA'],nc_dict['TAG']],0]=1
    # to avoid nan situation, a small numder is added to the mat==0 position
    mat[mat==0]=1e-10
    # print(torch.log(mat))
    return torch.log(mat)


class tAILoss(nn.Module):
    """Define tRNA affinity index loss function for RNA, species shall be specified first in the initialization.

    Args:
        criterion (str): Support 'l1', 'l2', 'sl1'.
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        codon_length: The codon coding sequnce length, if the 126 was chosen, the 'N' codon will be considered. Support '65', '126'. Default: 65.
        specy: Support ecoli, yeast, human, rat, mouse, dog.
        constrain: tAI compuation constarin. Support 'optimized' and 'standard'.

    Usgae:
        RNA_vec_pred (tensor): Shall be (B,L,65/126). The 65(64+1) logits will be sequences without ambiguoius codon N.
        RNA_vec_gt (tensor): Shall be (B,L,65/126).
        weight(float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, criterion='l1', loss_weight=1.0, reduction='mean',codon_length='65',specy='mouse',constrain='optimized'):
        super(tAILoss, self).__init__()
        if criterion == 'l1':
            self.loss_op = L1Loss(loss_weight, reduction)
        elif criterion == 'l2':
            self.loss_op = MSELoss(loss_weight, reduction)
        elif criterion == 'sl1':
            self.loss_op = SmoothL1Loss(loss_weight, reduction)
        else:
            raise ValueError(f'Unsupported loss mode: {criterion}. Supported ones are: l1|l2|sl1(SmoothL1Loss)')

        self.loss_weight = loss_weight

        if specy not in ['ecoli','yeast','human','mouse','rat']:
           raise ValueError(f'Unsupported species: {specy}.')

        # the vector is accually the log of w in the original work, and has 2 different reduction methods but set to be more proper one for Loss computation of single codon
        if codon_length=='65':
            self.val_mat=tAI_vector_prefetch(length=64,ss_list=sstrain[constrain],trna_count_dict=tRNA_gene_count[specy])
        elif codon_length=='126':
            self.val_mat=tAI_vector_prefetch(length=125,ss_list=sstrain[constrain],trna_count_dict=tRNA_gene_count[specy])
        else:
            raise ValueError(f'Unsupported length mode: {codon_length}. Supported ones are: 65/126. The extend mode is not allowed in this Loss function.')
        
        if torch.cuda.is_available():
            # self.code_mat=self.code_mat.cuda()
            self.val_mat=self.val_mat.cuda()
        # print(torch.cuda.is_available())


        

    def forward(self, RNA_vec_pred, RNA_vec_gt):
        
        if RNA_vec_pred.shape[1]!= RNA_vec_gt.shape[1]:
            raise ValueError('The Length of two codon do not comply')

        
        # begin the loss compuattion
        # detect the correct coding region for gt sequence
        vec_L=RNA_vec_gt @ self.val_mat # (B,L,23)
        # print(vec_L.shape)

        # mat_P=mat_E @ self.val_mat # (B,L,length)
        # # print(mat_P.shape)
        # mat_OP=torch.clamp(mat_P,min=0,max=1)*RNA_vec_pred # (B,L,length)
        # vec_L=torch.sum(mat_OP,dim=2) # (B,L)
        loss = self.loss_op(torch.exp(vec_L),torch.ones(vec_L.shape).cuda()) # (B)

        return loss * self.loss_weight
    

if __name__=='__main__':
    
    import random

    loss= tAILoss(criterion='l1', loss_weight=1.0, reduction='mean',codon_length='65',specy='ecoli')
    for ind9 in range(10):
      input=torch.rand((4,8,512,65))
      tgt=torch.zeros((4,8,512,65))
      for ind0 in range(4):
        for ind1 in range(8):
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

    # loss= tAILoss(criterion='l1', loss_weight=1.0, reduction='mean',codon_length='126',specy='mouse')
    # for ind9 in range(10):
    #   input=torch.rand((4,512,126))
    #   tgt=torch.zeros((4,512,126))
    #   for ind0 in range(4):
    #     for indx in range(512):
    #         tgt[ind0,indx,random.randint(0,125)]=1
      
    #   input=tgt
    #   input=torch.nn.functional.normalize(input,p=1.0,dim=2)
    #   # print(input[0,1,:])
    #   # print(tgt[0,1,:])
    #   o_loss=loss(input,tgt)
    #   print(o_loss)





import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import L1Loss, MSELoss, SmoothL1Loss
from .utils import CAI_dict,matrix_prefetch,get_nc_dict


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
    # 'START': ['ATG'],
    # 'No-Codon': ['EEE'],
}
            
            
def cai_vector_prefetch(length,cai_dict):
    
    if length==64:
        mat=torch.zeros((65,22))
        nc_dict=get_nc_dict(length)
        key_list=list(protein_dict.keys())
        for indx in range(len(key_list)):
            codon_list=protein_dict[key_list[indx]]
            for codon in codon_list:
                mat[nc_dict[codon],indx]=cai_dict[codon]
    elif length==125:
        mat=torch.zeros((126,22))
        nc_dict=get_nc_dict(length)
        key_list=list(protein_dict.keys())
        for indx in range(len(key_list)):
            codon_list=protein_dict[key_list[indx]]
            for codon in codon_list:
              mat[nc_dict[codon],indx]=cai_dict[codon]
    else:
        raise ValueError(f'Unsupported NC length mode: {length}. Supported ones are: 64/125')
    # add other coding 
    mat[nc_dict['EEE'],-1]=1 
    # print(mat[:,1])
    return mat

def get_cai_dict(raw_dict: list):
  dict1={}
  for indx in range(len(raw_dict[0])):
    dict1[raw_dict[0][indx]]=raw_dict[2][indx]
  # print(dict1)
  for ky in protein_dict.keys():
    rna_list=[]
    for sky in protein_dict[ky]:
      rna_list.append(float(dict1[sky]))
    ratio=max(rna_list)
    # print(ratio,rna_list)
    for sky in protein_dict[ky]:
      dict1[sky]=float(dict1[sky])/ratio

  return dict1

class CAILoss(nn.Module):
    """Define codon affinity index loss function for RNA

    Args:
        criterion (str): Support 'l1', 'l2', 'sl1'.
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        codon_length: The codon coding sequnce length, if the 126 was chosen, the 'N' codon will be considered. Support '65', '126'. Default: 65.
        specie: Support 'ecoli','yeast','insect','c.elegans','Drosophila','human','mouse','rat','pig','pichia','arabidopsis','streptomyces','zeamays','tabacco'.

    Usgae:
        RNA_vec_pred (tensor): Shall be (B,L,65/126). The 65(64+1) logits will be sequences without ambiguoius codon N.
        RNA_vec_gt (tensor): Shall be (B,L,65/126).
        

    """

    def __init__(self, criterion='l1', loss_weight=1.0, reduction='mean',codon_length='65',species='ecoli'):
        super(CAILoss, self).__init__()
        if criterion == 'l1':
            self.loss_op = L1Loss(loss_weight, reduction)
        elif criterion == 'l2':
            self.loss_op = MSELoss(loss_weight, reduction)
        elif criterion == 'sl1':
            self.loss_op = SmoothL1Loss(loss_weight, reduction)
        else:
            raise ValueError(f'Unsupported loss mode: {criterion}. Supported ones are: l1|l2|sl1(SmoothL1Loss)')

        self.loss_weight = loss_weight

        if species in ['ecoli','yeast','insect','c.elegans','Drosophila','human','mouse','rat','pig','pichia','arabidopsis','streptomyces','zeamays','tabacco']:
          self.cai_dict=get_cai_dict(CAI_dict[species])


        if codon_length=='65':
            self.code_mat=matrix_prefetch(length=64)
            self.val_mat=cai_vector_prefetch(length=64,cai_dict=self.cai_dict).transpose_(dim0=0,dim1=1)
        elif codon_length=='126':
            self.code_mat=matrix_prefetch(length=125,mode='extend')
            self.val_mat=cai_vector_prefetch(length=125,cai_dict=self.cai_dict).transpose_(dim0=0,dim1=1)
        else:
            raise ValueError(f'Unsupported length mode: {codon_length}. Supported ones are: 65/126. The extend mode is not allowed in this Lossfunction.')

        # self.matrix64=matrix_prefetch(length=64)
        # self.matrix125=matrix_prefetch(length=125)
        if torch.cuda.is_available():
            self.code_mat=self.code_mat.cuda()
            self.val_mat=self.val_mat.cuda()

        self.code_mat=nn.Parameter(self.code_mat,requires_grad=False)
        self.val_mat=nn.Parameter(self.val_mat,requires_grad=False)


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

    loss= CAILoss(criterion='l1', loss_weight=1.0, reduction='mean',codon_length='65',species='ecoli')
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

    loss= CAILoss(criterion='l1', loss_weight=1.0, reduction='mean',codon_length='65')
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





# if __name__=='__main__':


#   txtfile='tempfile.txt'
#   with open(txtfile, 'r') as txtf:
#     cnt=-1
#     info_list=[]
#     for i in range(5):
#       info_list.append([])
#     info_ind=0
#     row_ind=0
#     for line in txtf:
#       line = line.strip()
#       cnt+=1
      
#       for fragment in line.split('\t'):
#         info_list[info_ind].append(fragment)
#         # print(info_ind,fragment)
#         # print(fragment)
#         pass
#         row_ind+=1
#         if row_ind//4==1:
#           info_ind+=1
#           info_ind=info_ind%5
#           row_ind=0
#       if cnt/3==10:
#         cnt=-1
      
      
      
#     print(info_list)


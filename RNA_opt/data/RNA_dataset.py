import numpy as np
import os
from torch.utils import data as data
import pandas as pd

enc_dict = {'AAA': 'Ā', 'AAT': 'ā', 'AAG': 'Ă', 'AAC': 'ă', 'AAN': 'Ą', 'ATA': 'ą', 'ATT': 'Ć', 'ATG': 'ć', 'ATC': 'Ĉ',
            'ATN': 'ĉ', 'AGA': 'Ċ', 'AGT': 'ċ', 'AGG': 'Č', 'AGC': 'č', 'AGN': 'Ď', 'ACA': 'ď', 'ACT': 'Đ', 'ACG': 'đ',
            'ACC': 'Ē', 'ACN': 'ē', 'ANA': 'Ĕ', 'ANT': 'ĕ', 'ANG': 'Ė', 'ANC': 'ė', 'ANN': 'Ę', 'TAA': 'ę', 'TAT': 'Ě',
            'TAG': 'ě', 'TAC': 'Ĝ', 'TAN': 'ĝ', 'TTA': 'Ğ', 'TTT': 'ğ', 'TTG': 'Ġ', 'TTC': 'ġ', 'TTN': 'Ģ', 'TGA': 'ģ',
            'TGT': 'Ĥ', 'TGG': 'ĥ', 'TGC': 'Ħ', 'TGN': 'ħ', 'TCA': 'Ĩ', 'TCT': 'ĩ', 'TCG': 'Ī', 'TCC': 'ī', 'TCN': 'Ĭ',
            'TNA': 'ĭ', 'TNT': 'Į', 'TNG': 'į', 'TNC': 'İ', 'TNN': 'ı', 'GAA': 'Ĳ', 'GAT': 'ĳ', 'GAG': 'Ĵ', 'GAC': 'ĵ',
            'GAN': 'Ķ', 'GTA': 'ķ', 'GTT': 'ĸ', 'GTG': 'Ĺ', 'GTC': 'ĺ', 'GTN': 'Ļ', 'GGA': 'ļ', 'GGT': 'Ľ', 'GGG': 'ľ',
            'GGC': 'Ŀ', 'GGN': 'ŀ', 'GCA': 'Ł', 'GCT': 'ł', 'GCG': 'Ń', 'GCC': 'ń', 'GCN': 'Ņ', 'GNA': 'ņ', 'GNT': 'Ň',
            'GNG': 'ň', 'GNC': 'ŉ', 'GNN': 'Ŋ', 'CAA': 'ŋ', 'CAT': 'Ō', 'CAG': 'ō', 'CAC': 'Ŏ', 'CAN': 'ŏ', 'CTA': 'Ő',
            'CTT': 'ő', 'CTG': 'Œ', 'CTC': 'œ', 'CTN': 'Ŕ', 'CGA': 'ŕ', 'CGT': 'Ŗ', 'CGG': 'ŗ', 'CGC': 'Ř', 'CGN': 'ř',
            'CCA': 'Ś', 'CCT': 'ś', 'CCG': 'Ŝ', 'CCC': 'ŝ', 'CCN': 'Ş', 'CNA': 'ş', 'CNT': 'Š', 'CNG': 'š', 'CNC': 'Ţ',
            'CNN': 'ţ', 'NAA': 'Ť', 'NAT': 'ť', 'NAG': 'Ŧ', 'NAC': 'ŧ', 'NAN': 'Ũ', 'NTA': 'ũ', 'NTT': 'Ū', 'NTG': 'ū',
            'NTC': 'Ŭ', 'NTN': 'ŭ', 'NGA': 'Ů', 'NGT': 'ů', 'NGG': 'Ű', 'NGC': 'ű', 'NGN': 'Ų', 'NCA': 'ų', 'NCT': 'Ŵ',
            'NCG': 'ŵ', 'NCC': 'Ŷ', 'NCN': 'ŷ', 'NNA': 'Ÿ', 'NNT': 'Ź', 'NNG': 'ź', 'NNC': 'Ż', 'NNN': 'ż',
            'virus': 'Ž', 'bacteria': 'ž', 'mammalia': 'ſ', 'SEP': 'ƀ', 'CLS': 'Ɓ', 'EEE': 'Ƃ', 'PAD': 'ƃ', '': 'Ƅ'}
dec_dict = {'Ā': 'AAA', 'ā': 'AAT', 'Ă': 'AAG', 'ă': 'AAC', 'Ą': 'AAN', 'ą': 'ATA', 'Ć': 'ATT', 'ć': 'ATG', 'Ĉ': 'ATC',
            'ĉ': 'ATN', 'Ċ': 'AGA', 'ċ': 'AGT', 'Č': 'AGG', 'č': 'AGC', 'Ď': 'AGN', 'ď': 'ACA', 'Đ': 'ACT', 'đ': 'ACG',
            'Ē': 'ACC', 'ē': 'ACN', 'Ĕ': 'ANA', 'ĕ': 'ANT', 'Ė': 'ANG', 'ė': 'ANC', 'Ę': 'ANN', 'ę': 'TAA', 'Ě': 'TAT',
            'ě': 'TAG', 'Ĝ': 'TAC', 'ĝ': 'TAN', 'Ğ': 'TTA', 'ğ': 'TTT', 'Ġ': 'TTG', 'ġ': 'TTC', 'Ģ': 'TTN', 'ģ': 'TGA',
            'Ĥ': 'TGT', 'ĥ': 'TGG', 'Ħ': 'TGC', 'ħ': 'TGN', 'Ĩ': 'TCA', 'ĩ': 'TCT', 'Ī': 'TCG', 'ī': 'TCC', 'Ĭ': 'TCN',
            'ĭ': 'TNA', 'Į': 'TNT', 'į': 'TNG', 'İ': 'TNC', 'ı': 'TNN', 'Ĳ': 'GAA', 'ĳ': 'GAT', 'Ĵ': 'GAG', 'ĵ': 'GAC',
            'Ķ': 'GAN', 'ķ': 'GTA', 'ĸ': 'GTT', 'Ĺ': 'GTG', 'ĺ': 'GTC', 'Ļ': 'GTN', 'ļ': 'GGA', 'Ľ': 'GGT', 'ľ': 'GGG',
            'Ŀ': 'GGC', 'ŀ': 'GGN', 'Ł': 'GCA', 'ł': 'GCT', 'Ń': 'GCG', 'ń': 'GCC', 'Ņ': 'GCN', 'ņ': 'GNA', 'Ň': 'GNT',
            'ň': 'GNG', 'ŉ': 'GNC', 'Ŋ': 'GNN', 'ŋ': 'CAA', 'Ō': 'CAT', 'ō': 'CAG', 'Ŏ': 'CAC', 'ŏ': 'CAN', 'Ő': 'CTA',
            'ő': 'CTT', 'Œ': 'CTG', 'œ': 'CTC', 'Ŕ': 'CTN', 'ŕ': 'CGA', 'Ŗ': 'CGT', 'ŗ': 'CGG', 'Ř': 'CGC', 'ř': 'CGN',
            'Ś': 'CCA', 'ś': 'CCT', 'Ŝ': 'CCG', 'ŝ': 'CCC', 'Ş': 'CCN', 'ş': 'CNA', 'Š': 'CNT', 'š': 'CNG', 'Ţ': 'CNC',
            'ţ': 'CNN', 'Ť': 'NAA', 'ť': 'NAT', 'Ŧ': 'NAG', 'ŧ': 'NAC', 'Ũ': 'NAN', 'ũ': 'NTA', 'Ū': 'NTT', 'ū': 'NTG',
            'Ŭ': 'NTC', 'ŭ': 'NTN', 'Ů': 'NGA', 'ů': 'NGT', 'Ű': 'NGG', 'ű': 'NGC', 'Ų': 'NGN', 'ų': 'NCA', 'Ŵ': 'NCT',
            'ŵ': 'NCG', 'Ŷ': 'NCC', 'ŷ': 'NCN', 'Ÿ': 'NNA', 'Ź': 'NNT', 'ź': 'NNG', 'Ż': 'NNC', 'ż': 'NNN',
            'Ž': 'virus', 'ž': 'bacteria', 'ſ': 'mammalia', 'ƀ': 'SEP', 'Ɓ': 'CLS', 'Ƃ': 'EEE', 'ƃ': 'PAD', 'Ƅ': ''}


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
}


# def get_nc_dict(length=64):
#     nc_dict={}
#     pos=0
#     if length==125:
#       for alf1 in ['A', 'G', 'C', 'T', 'N']:
#           for alf2 in ['A', 'G', 'C', 'T', 'N']:
#             for alf3 in ['A', 'G', 'C', 'T', 'N']:
#               nc_dict[alf1 + alf2 + alf3] = pos
#               pos+=1
#     elif length==64:
#         for alf1 in ['A', 'G', 'C', 'T']:
#           for alf2 in ['A', 'G', 'C', 'T']:
#             for alf3 in ['A', 'G', 'C', 'T']:
#               nc_dict[alf1 + alf2 + alf3] = pos
#               pos+=1
#     nc_dict['EEE']=pos # adding the ending code
#     # print(f'nc_dict length {pos}')
#     return nc_dict


def get_nc_dict(length=64):
    print(f'nc_dict length {length}')
    nc_dict={}
    pos=0
    
    if length==64:
        for alf1 in ['T', 'C', 'A', 'G']:
          for alf2 in ['T', 'C', 'A', 'G']:
            for alf3 in ['T', 'C', 'A', 'G']:
              nc_dict[alf1 + alf2 + alf3] = pos
              pos+=1
    elif length==125:
        for alf1 in ['T', 'C', 'A', 'G', 'N']:
          for alf2 in ['T', 'C', 'A', 'G', 'N']:
            for alf3 in ['T', 'C', 'A', 'G', 'N']:
                nc_dict[alf1 + alf2 + alf3] = pos
                pos+=1
    else:
       raise ValueError('This version only support the 64 and 125 coding')
    nc_dict['EEE']=pos # adding the ending code
    # print(f'nc_dict length {pos}')
    return nc_dict

class RNAdataset(data.Dataset):
    """

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            image_folder (str): the folder containing all the images.
            csv_path (str): the csv file consists of all image names and their class.
            class (int/float): the classification label of the image.
            image_size (tuple): Resize the image into a fin size (should be square).

            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(RNAdataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        nc_dict=get_nc_dict(opt['rna_mode']-1)
        # read csv and build a data list
        # self.dt_folder = opt['image_folder']
        
        self.augment_ratio=opt['augment_ratio'] if 'augment_ratio' in opt else None
        with open(opt['txt_path'],'r') as txtf:
          self.data_list=[]
          self.class_list=[]
          for line in txtf:
            line = line.strip('\n')
            # if len(line)<2:
            #    continue
            line=line.split(' ')
            if len(line)<30:
               continue
            tgt_line=np.zeros((opt['max_rna_len'],opt['rna_mode']))
            self.class_list.append(dec_dict[line[0]])
            # skip the class label
            for ind0 in range(0,len(line)-1):
               tgt_line[ind0,nc_dict[dec_dict[line[ind0+1]]]]=1
            for ind1 in range(len(line)-1,opt['max_rna_len']):
              #  print(nc_dict['EEE'])
               tgt_line[ind1,nc_dict['EEE']]=1
            self.data_list.append(tgt_line)

        # print(self.data_list)


        # directly increase the lenth of list, will effect the validation time and epochs counting
        if self.augment_ratio is not None and self.augment_ratio>1:
          newdl=[]
          newcl=[]
          for ind2 in range(self.augment_ratio):
            newdl.extend(self.data_list)
            newcl.extend(self.class_list)
          self.data_list=newdl
          self.class_list=newcl

        # print(len(self.data_list),len(self.class_list))

    def __getitem__(self, index):

        return {'data': self.data_list[index],'class': self.class_list[index]}

    def __len__(self):
        return len(self.data_list)
    
class RNAcsvData(RNAdataset):
  def __init__(self, opt):
    super(RNAdataset, self).__init__()
    self.opt = opt
    # file client (io backend)
    # read csv and build a data list    
    self.augment_ratio=opt['augment_ratio'] if 'augment_ratio' in opt else None
    pd_data=pd.read_csv(opt['txt_path'])
    # print(self.data_list)
    self.nc_dict=get_nc_dict(opt['rna_mode']-1)
    self.seq_mode=opt['sequence_mode'] if 'sequence_mode' in opt else 'rna'
    


    # directly increase the lenth of list, will effect the validation time and epochs counting
    if self.augment_ratio is not None and self.augment_ratio>1:
      pd_data = pd.DataFrame(np.repeat(pd_data.values, self.augment_ratio,axis=0))

    self.data_list=pd_data

  def __getitem__(self, index):
    tgt_line=np.zeros((self.opt['max_rna_len'],self.opt['rna_mode']))

    
    seq_data=self.data_list.loc[index]['seq'].strip().split(' ')
    
      
    # skip the class label
    for ind0 in range(0,len(seq_data)):
      if self.seq_mode!='pro':
        tgt_line[ind0,self.nc_dict[seq_data[ind0].replace('U','T')]]=1
      else: # if the sequence is protein, the coding codon is selected in the first place.
        tgt_line[ind0,self.nc_dict[protein_dict[seq_data[ind0]][0]]]=1
    # for ind1 in range(len(seq_data),self.opt['max_rna_len']):
    #   #  print(nc_dict['EEE'])
    #     tgt_line[ind1,self.nc_dict['EEE']]=1
    tgt_line[len(seq_data):self.opt['max_rna_len'],self.nc_dict['EEE']]=1


    try: 
      # dummy=self.data_list.loc[index]['ee']
      return {'data': tgt_line,'class': self.data_list.loc[index]['class'],'ensemble_energy': 0-self.data_list.loc[index]['ee']}
    except:
      # print(f'The dataset does not include the ensemble energy in the csv table, loading basic infomation instead')
      return {'data': tgt_line,'class': self.data_list.loc[index]['class']}
    

class RNAcsvTkn(RNAdataset):
  def __init__(self, opt):
    super(RNAdataset, self).__init__()
    self.opt = opt
    # file client (io backend)
    # read csv and build a data list    
    self.augment_ratio=opt['augment_ratio'] if 'augment_ratio' in opt else None
    pd_data=pd.read_csv(opt['txt_path'])
    # print(self.data_list)
    self.nc_dict=get_nc_dict(opt['rna_mode']-1)
    self.pad_atten=opt['pad_atten'] if 'pad_atten' in opt else True


    # directly increase the lenth of list, will effect the validation time and epochs counting
    if self.augment_ratio is not None and self.augment_ratio>1:
      pd_data = pd.DataFrame(np.repeat(pd_data.values, self.augment_ratio,axis=0))

    self.data_list=pd_data

  def __getitem__(self, index):
    seq_data=self.data_list.loc[index]['seq'].strip().split(' ')
    atten_mask=np.ones((self.opt['max_rna_len']))
    # # skip the class label
    if self.pad_atten:
      # for ind0 in range(0,len(seq_data)):
      #   atten_mask[ind0]=1
      for ind1 in range(len(seq_data),self.opt['max_rna_len']):
        atten_mask[ind1]=0

    tgt_line=np.zeros((self.opt['max_rna_len'],self.opt['rna_mode']))
    # skip the class label
    for ind0 in range(0,len(seq_data)):
        tgt_line[ind0,self.nc_dict[seq_data[ind0]]]=1
    tgt_line[len(seq_data):self.opt['max_rna_len'],self.nc_dict['EEE']]=1

    try: 
      # dummy=self.data_list.loc[index]['ee']
      # print()
      return {'data': tgt_line,'class': self.data_list.loc[index]['class'],'atten_mask': atten_mask, 'ensemble_energy': 0-self.data_list.loc[index]['ee']}
    except:
      # print(f'The dataset does not include the ensemble energy in the csv table, loading basic infomation instead')
      return {'data': tgt_line,'class': self.data_list.loc[index]['class']}
    else:
      return {'data': tgt_line,'class': self.data_list.loc[index]['class']}
  


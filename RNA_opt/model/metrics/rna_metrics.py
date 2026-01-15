import numpy as np
from scipy import stats


from ..RPLoss.utils import protein_dict,get_nc_dict,CAI_dict,tAI_vector_prefetch,get_cai_dict

dict_ks=list(protein_dict.keys())


def calculate_difference(seq1,seq2,**kwargs):
    """
    Calculate the L1 loss for 2 imgs.
    Order of input did not effect the result.
    """

    gt_co =seq2.strip().split(' ')
    pre_co =seq1.strip().split(' ')

    if len(gt_co)!=len(pre_co):
      return None
    else:
      diff=0
      wrong=0
      for ind in range(len(pre_co)):
        # print(pre_co[ind],gt_co[ind])
        if pre_co[ind]!=gt_co[ind]:
          diff+=1
        for dkey in dict_ks:
          if pre_co[ind] in protein_dict[dkey] :
            if gt_co[ind] not in protein_dict[dkey] :
              wrong+=1

      if wrong>0:
        return None
      else:
         return diff/len(seq1)
    

def calculate_lengtherror(seq1,seq2,**kwargs):
    """
    Calculate the Spearman rank CC for the score.
    Order of input did not effect the result.
    """
    gt_co =seq2.strip().split(' ')
    pre_co =seq1.strip().split(' ')

    if len(gt_co)!=len(pre_co):
      return 1
    else:
      return 0

def calculate_codonerror(seq1,seq2,**kwargs):
    """
    Calculate the Spearman rank CC for the score.
    Order of input did not effect the result.
    """
    gt_co =seq2.strip().split(' ')
    pre_co =seq1.strip().split(' ')

    if len(gt_co)!=len(pre_co):
      return 0
    else:
      for ind in range(len(pre_co)):        
        for dkey in dict_ks:
          if pre_co[ind] in protein_dict[dkey] :
            if gt_co[ind] not in protein_dict[dkey] :
              return 1
      return 0

def calculate_codonerrorrate(seq1,seq2,**kwargs):
    """
    Calculate the Pearson linear CC for the score.
    Order of input did not effect the result.
    """
    gt_co =seq2.strip().split(' ')
    pre_co =seq1.strip().split(' ')

    if len(gt_co)!=len(pre_co):
      return None
    else:
      wrong=0
      for ind in range(len(pre_co)):
        for dkey in dict_ks:
          if pre_co[ind] in protein_dict[dkey] :
            if gt_co[ind] not in protein_dict[dkey] :
              wrong+=1

      if wrong>0:
        return wrong/len(seq1)
      else:
         return None

def calculate_cai(seq1, seq2, species=None, **kwargs):
    """
    Calculate the Codon Adaptation Index (CAI) for a predicted RNA sequence.
    
    CAI measures the relative adaptiveness of a codon to the synonymous codons
    for a particular amino acid in a reference set of highly expressed genes.
    
    Args:
        seq1 (str): Predicted sequence (space-separated codons)
        seq2 (str): Ground truth sequence (space-separated codons, used for validation)
        species (str): Species identifier for CAI reference data
        
    Returns:
        float: CAI value (ranges from 0 to 1, higher values indicate better adaptation)
        
    Raises:
        ValueError: If species is not provided or sequences are invalid
    """
    if not species:
        raise ValueError("Species must be provided for CAI calculation")
    
    # Get CAI dictionary for the specified species
    try:
        cai_weights = get_cai_dict(CAI_dict[species])
    except KeyError:
        raise ValueError(f"No CAI data available for species: {species}")
    
    # Validate and prepare input sequences
    pre_line = seq1.strip()
    
    # Split into codons and validate equal length
    pre_codons = pre_line.split()

    
    # Calculate CAI as geometric mean of codon weights
    log_weights = []
    valid_codons = 0
    
    for codon in pre_codons:
        if codon in cai_weights:
            weight = cai_weights[codon]
            if weight > 0:  # Only include positive weights in calculation
                log_weights.append(np.log(weight))
                valid_codons += 1
    
    if valid_codons == 0:
        return 0.0  # No valid codons found
    
    # Calculate geometric mean of weights
    cai = np.exp(sum(log_weights) / valid_codons)
    return float(cai)  # Convert from numpy to Python float
      
def calculate_tai(seq1, seq2, species=None, **kwargs):
    """
    Calculate the tRNA Adaptation Index (tAI) for a predicted RNA sequence.
    
    The tAI measures how well the codons in the sequence match the tRNA pool
    of the specified species, which is related to translation efficiency.
    
    Args:
        seq1 (str): Predicted sequence (space-separated codons)
        seq2 (str): Ground truth sequence (space-separated codons, used for validation)
        species (str): Species identifier for tRNA gene count data
        
    Returns:
        float: tAI value (higher values indicate better tRNA adaptation)
        
    Raises:
        ValueError: If species is not provided or sequences are invalid
    """
    if not species:
        raise ValueError("Species must be provided for tAI calculation")
    
    # Get codon to index mapping and precompute tAI vector
    nc_dict = get_nc_dict(64)
    tai_vec = tAI_vector_prefetch(species, nc_dict)
    
    # Validate and prepare input sequences
    gt_line = seq2.strip()
    pre_line = seq1.strip()
    if not (gt_line and pre_line):
        raise ValueError("Both input sequences must be non-empty")
    
    # Split into codons and validate equal length
    gt_codons = gt_line.split()
    pre_codons = pre_line.split()

    
    # Initialize codon count matrix (65 possible codons, 2 sequences)
    codon_counts = np.zeros((65, 2))
    
    # Count codon occurrences in both sequences
    valid_codons = 0
    for gt_codon, pre_codon in zip(gt_codons, pre_codons):
        if gt_codon in nc_dict and pre_codon in nc_dict:
            codon_counts[nc_dict[pre_codon], 1] += 1  # Predicted sequence
            codon_counts[nc_dict[gt_codon], 0] += 1   # Ground truth sequence
            valid_codons += 1
    
    if valid_codons == 0:
        return 0.0  # No valid codons found
    
    # Remove stop codons and other special codons from calculation
    special_codons = ['EEE', 'ATG', 'TAA', 'TGA', 'TAG']
    indices_to_remove = [nc_dict[codon] for codon in special_codons if codon in nc_dict]
    filtered_counts = np.delete(codon_counts[:, 1], indices_to_remove)  # Only use predicted sequence
    
    # Calculate tAI as weighted geometric mean of tRNA adaptation values
    if np.sum(filtered_counts) > 0:
        tai_value = np.exp(np.dot(filtered_counts, tai_vec) / np.sum(filtered_counts))
        return float(tai_value[0])  # Convert from numpy array to Python float
    else:
        return 0.0

from copy import deepcopy
import os ,time ,cv2,torch,math
from torchvision.utils import make_grid
import numpy as np
from .arch.transformerXL_arch import TransXL,TransDown
from .arch.swin_transformer_arch import SwinTransformer
from .arch.dummy_arch import dummy_arch
# Optional architectures - may not be available in test repository
try:
    from .arch.Berna_arch import Berna_Dummy_arch,Berna_Seq_arch,Forna_arch
    from .arch.dummy_arch import opt_dummy_arch,MFELoss_arch,MLPRNo_net
    from .arch.test_transformer_arch import TransForna
except ImportError:
    Berna_Dummy_arch = None
    Berna_Seq_arch = None
    Forna_arch = None
    opt_dummy_arch = None
    MFELoss_arch = None
    MLPRNo_net = None
    TransForna = None
from .logger_utils import get_root_logger
from .loss.losses import L1Loss,MSELoss,SmoothL1Loss
try:
    from .RPLoss.RPLoss import RNAProteinLoss
    from .RPLoss.CAILoss import CAILoss
    from .RPLoss.tAILoss import tAILoss
    from .RPLoss.MFELoss import MFELoss,ForceMFELoss
except ImportError:
    # Placeholder imports for test-only repository
    RNAProteinLoss = None
    CAILoss = None
    tAILoss = None
    MFELoss = None
    ForceMFELoss = None
from .RPLoss.utils import get_dec_dict
from .metrics.rna_metrics import calculate_difference,calculate_codonerror,calculate_codonerrorrate,calculate_lengtherror
# Optional metrics - may not be fully available
try:
    from .metrics.rna_metrics import calculate_cai,calculate_tai
    from .metrics.class_metrics import calculate_acc,calculate_f1,calculate_p,calculate_r
    from .metrics.iqa_metrics import calculate_mae,calculate_plcc,calculate_r2,calculate_rmse,calculate_srcc,calculate_l1
except ImportError:
    # Placeholders for optional metrics
    pass



def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    # print('One real call')
    logger = get_root_logger()
    if network_type in ['TransXL','dummy_arch','SwinTransformer','TransDown'] or (network_type in ['Berna_Dummy_arch','Berna_Seq_arch','MFELoss_arch','Forna_arch','MLPRNo_net','TransForna'] and globals().get(network_type) is not None):
      if network_type=='TransXL':
        net = TransXL(**opt)
      elif network_type=='Berna_Dummy_arch':
        net = Berna_Dummy_arch(**opt)
      elif network_type=='Berna_Seq_arch':
        net = Berna_Seq_arch(**opt)
      elif network_type=='dummy_arch':
        net = dummy_arch(**opt)
      elif network_type=='SwinTransformer':
        net = SwinTransformer(**opt)
      elif network_type=='MFELoss_arch':
        net = MFELoss_arch(**opt)
      elif network_type=='Forna_arch':
        net = Forna_arch(**opt)
      elif network_type=='MLPRNo_net':
        net = MLPRNo_net(**opt)
      elif network_type=='TransDown':
        net = TransDown(**opt)
      elif network_type=='TransForna':
        net = TransForna(**opt)
      logger.info(f'Network [{net.__class__.__name__}] is created.')
    else:
      net = None
      logger.info(f'Network is NOT created. No matched name.')
    
    return net

def build_loss(opt):
  opt = deepcopy(opt)
  loss_type = opt.pop('type')
  logger = get_root_logger()
  if loss_type in ['L1Loss','SmoothL1Loss','MSELoss','RNAProteinLoss','CAILoss','tAILoss','MFELoss','ForceMFELoss']:
    
    if loss_type=='L1Loss':
      new_loss =L1Loss(**opt)
    elif loss_type=='SmoothL1Loss':
      new_loss =SmoothL1Loss(**opt)
    elif loss_type=='MSELoss':
      new_loss =MSELoss(**opt)
    elif loss_type=='RNAProteinLoss':
      if RNAProteinLoss is None:
        logger.warning('RNAProteinLoss not available, using placeholder')
        new_loss = None
      else:
        new_loss = RNAProteinLoss(**opt)
    elif loss_type=='tAILoss':
      if tAILoss is None:
        logger.warning('tAILoss not available, using placeholder')
        new_loss = None
      else:
        new_loss =tAILoss(**opt)
    elif loss_type=='CAILoss':
      if CAILoss is None:
        logger.warning('CAILoss not available, using placeholder')
        new_loss = None
      else:
        new_loss =CAILoss(**opt)
    elif loss_type=='MFELoss':
      if MFELoss is None:
        logger.warning('MFELoss not available, using placeholder')
        new_loss = None
      else:
        new_loss =MFELoss(**opt)
    elif loss_type=='ForceMFELoss':
      if ForceMFELoss is None:
        logger.warning('ForceMFELoss not available, using placeholder')
        new_loss = None
      else:
        new_loss =ForceMFELoss(**opt)
    if new_loss is not None:
      logger.info(f'Loss [{new_loss.__class__.__name__}] is created.')
    else:
      logger.warning(f'Loss Function {loss_type} is NOT created (placeholder only).')
  else:
    new_loss = None
    logger.info(f'Loss Function '+loss_type+' is NOT created. No matched name.')

  return new_loss



def calculate_metric(data, opt):
  """Calculate metric from data and options.

  Args:
      opt (dict): Configuration. It must contain:
          type (str): Model type.
  """
  opt = deepcopy(opt)
  logger = get_root_logger()
  metric_type = opt.pop('type')
  if metric_type in ['calculate_acc','calculate_f1','calculate_p','calculate_r', 'calculate_srcc','calculate_plcc','calculate_rmse','calculate_l1','calculate_difference','calculate_Dice','calculate_mae','calculate_r2','calculate_lengtherror','calculate_codonerrorrate','calculate_codonerror','calculate_cai','calculate_tai']:
    if metric_type=='calculate_acc':
      result = calculate_acc(**data,**opt)
    elif metric_type=='calculate_f1':
      result = calculate_f1(**data,**opt)
    elif metric_type=='calculate_p':
      result =calculate_p(**data,**opt)
    elif metric_type=='calculate_r':
      result =calculate_r(**data,**opt)
    elif metric_type=='calculate_srcc':
      result =calculate_srcc(**data,**opt)
    elif metric_type=='calculate_plcc':
      result =calculate_plcc(**data,**opt)
    elif metric_type=='calculate_rmse':
      result =calculate_rmse(**data,**opt)
    elif metric_type=='calculate_l1':
      result =calculate_l1(**data,**opt)
    elif metric_type=='calculate_difference':
      result =calculate_difference(**data,**opt)
    elif metric_type=='calculate_mae':
      result =calculate_mae(**data,**opt)
    elif metric_type=='calculate_r2':
      result =calculate_r2(**data,**opt)
    elif metric_type=='calculate_lengtherror':
      result =calculate_lengtherror(**data,**opt)
    elif metric_type=='calculate_codonerrorrate':
      result =calculate_codonerrorrate(**data,**opt)
    elif metric_type=='calculate_codonerror':
      result =calculate_codonerror(**data,**opt)
    elif metric_type=='calculate_cai':
      result =calculate_cai(**data,**opt)
    elif metric_type=='calculate_tai':
      result =calculate_tai(**data,**opt)
    
  else:
    result = None
    logger.info(f'Loss Function '+metric_type+' is NOT created. No matched name.')

  return result

def txt_write(str_to_save, file_path, params=None, auto_mkdir=True):
  """Write str to txt file.

  Args:
      str_to_sav (str): trx data.
      file_path (str): saving file path.
      params (None or list): Same as to_csv() interference.
      auto_mkdir (bool): If the parent folder of `file_path` does not exist,
          whether to create it automatically.
  """
  if auto_mkdir:
      dir_name = os.path.abspath(os.path.dirname(file_path))
      os.makedirs(dir_name, exist_ok=True)
  if isinstance(str_to_save,str):
     with open(file_path,'a') as xtxf:
        xtxf.write(str_to_save+'\n')
  else:
     with open(file_path,'a') as xtxf:
        for ind in range(len(str_to_save)):
          xtxf.write(str_to_save[ind]+'\n')

  return 

def csv_write(data_frame, file_path, params=None, auto_mkdir=True):
  """Write csv to file.

  Args:
      data_frame (pd.DataFrame): csv data.
      file_path (str): saving file path.
      params (None or list): Same as to_csv() interference.
      auto_mkdir (bool): If the parent folder of `file_path` does not exist,
          whether to create it automatically.
  """
  if auto_mkdir:
      dir_name = os.path.abspath(os.path.dirname(file_path))
      os.makedirs(dir_name, exist_ok=True)
  sav = data_frame.to_csv(file_path)

  return sav

def check_resume(opt, resume_iter):
    """Check resume states and pretrain_network paths.

    Args:
        opt (dict): Options.
        resume_iter (int): Resume iteration.
    """
    if opt['path']['resume_state']:
        # get all the networks
        networks = [key for key in opt.keys() if key.startswith('network_')]
        flag_pretrain = False
        for network in networks:
            if opt['path'].get(f'pretrain_{network}') is not None:
                flag_pretrain = True
        if flag_pretrain:
            print('pretrain_network path will be ignored during resuming.')
        # set pretrained model paths
        for network in networks:
            name = f'pretrain_{network}'
            basename = network.replace('network_', '')
            if opt['path'].get('ignore_resume_networks') is None or (network
                                                                     not in opt['path']['ignore_resume_networks']):
                opt['path'][name] = os.path.join(opt['path']['models'], f'net_{basename}_{resume_iter}.pth')
                print(f"Set {name} to {opt['path'][name]}")

        # change param_key to params in resume
        param_keys = [key for key in opt['path'].keys() if key.startswith('param_key')]
        for param_key in param_keys:
            if opt['path'][param_key] == 'params_ema':
                opt['path'][param_key] = 'params'
                print(f'Set {param_key} to params')


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if os.path.exists(path):
        new_name = path + '_archived_' + time.strftime('%Y%m%d_%H%M%S', time.localtime())
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


# @master_only
def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt.pop('experiments_root'))
    else:
        mkdir_and_rename(path_opt.pop('results_root'))
    for key, path in path_opt.items():
        if ('strict_load' in key) or ('pretrain_net' in key) or ('resume' in key) or ('param_key' in key):
            continue
        else:
            os.makedirs(path, exist_ok=True)



def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)



def fea2seq(tensor, mode=64, split=' '):
    """Convert torch Tensors into rna sequences.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')
    dec_dict=get_dec_dict(mode)

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        n_dim = _tensor.dim()
        if n_dim == 3:
            img_np = _tensor.numpy()
            max_index=np.argmax(img_np,axis=2)
            # print(max_index)
            # print(img_np)
            for ind0 in range(img_np.shape[0]):
              sub_res=''
              for ind1 in range(img_np.shape[1]):
                if dec_dict[str(max_index[ind0,ind1])]!='EEE':
                  sub_res+=dec_dict[str(max_index[ind0,ind1])]
                  sub_res+=split
              result.append(sub_res)
            
        elif n_dim == 2:
            img_np = _tensor.numpy()
            max_index=np.argmax(img_np,axis=1)
            sub_res=''
            for ind1 in range(img_np.shape[0]):
              if dec_dict[str(max_index[ind0,ind1])]!='EEE':
                sub_res+=dec_dict[str(max_index[ind0,ind1])]
                sub_res+=split
            result.append(sub_res)

        else:
            raise TypeError('Only support 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
    if len(result) == 1:
        result = result[0]
    return result




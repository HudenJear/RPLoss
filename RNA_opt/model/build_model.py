# from .catintell_dehaze_model import DehazeModel
# from .catintell_generate_model import GenerateModel
from .rnaopt_model import RNAoptModel
from .logger_utils import get_root_logger
from copy import deepcopy

def build_model(opt):
  """Build model from options.

  Args:
      opt (dict): Configuration. It must contain:
          model_type (str): Model type.
  """
  model_type = opt.pop('model_type')
  opt = deepcopy(opt)
  if model_type in ['RNAoptModel',]:
    if model_type == 'RNAoptModel':
      model = RNAoptModel(opt)
    
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
  else: 
    model =None
    logger = get_root_logger()
    logger.info('Model '+model_type+' is NOT created. No matched name.')
  
  return model
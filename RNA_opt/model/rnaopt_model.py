import pandas as pd
import torch, time
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from .base_model import BaseModel
from .build_utils import build_network,build_loss,fea2seq,txt_write,calculate_metric
from .logger_utils import get_root_logger

class RNAoptModel(BaseModel):
    """Catintell model for dehaze."""

    def __init__(self, opt):
        super(RNAoptModel, self).__init__(opt)

        # define degrade network
        self.net_g = build_network(opt['network_g'])
        self.fp16= load_path = self.opt.get('fp16', False)
        if self.fp16:
          self.net_g = self.model_to_device(self.net_g).half()
        else:
          self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.species=None


        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()

        train_opt = self.opt['train']

        if train_opt.get('protein_opt'):
            self.cri_pro = build_loss(train_opt['protein_opt']).to(self.device)
            if self.fp16:
                self.cri_pro = self.cri_pro.half()
        else:
            self.cri_pro = None
        if train_opt.get('CAI_opt'):
            self.cri_cai = build_loss(train_opt['CAI_opt']).to(self.device)
            if not self.species:
              self.species=train_opt['CAI_opt']['species']
            if self.fp16:
                self.cri_cai = self.cri_cai.half()
          
        else:
            self.cri_cai = None
        if train_opt.get('tAI_opt'):
            self.cri_tai = build_loss(train_opt['tAI_opt']).to(self.device)
            if not self.species:
              self.species=train_opt['tAI_opt']['species']
            if self.fp16:
                self.cri_tai = self.cri_tai.half()
        else:
            self.cri_tai = None
        if train_opt.get('mfe_opt'):
            self.cri_mfe = build_loss(train_opt['mfe_opt']).to(self.device)
            if self.fp16:
                self.cri_mfe = self.cri_mfe.half()
            self.mfe_mode=train_opt['mfe_opt']['mode']
        else:
            self.cri_mfe = None


        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')

            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            if self.fp16:
              self.net_g_ema = build_network(self.opt['network_g']).to(self.device).half()
            else:
              self.net_g_ema = build_network(self.opt['network_g']).to(self.device)


            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # set optimizer for model A
        

        optim_params = []
        for k, v in self.net_g.named_parameters():  # key ,value
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # # optimizer d
        # optim_type = optimizer_opt['optim_d'].pop('type')
        # self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **optimizer_opt['optim_d'])
        # self.optimizers.append(self.optimizer_d)

        # # set up iter number of g & d
        # self.net_d_iters = optimizer_opt.get('net_d_iters', 1)
        self.net_g_init_iters = train_opt.get('net_g_init_iters', 0)
        self.net_g_end_iters= train_opt.get('net_g_end_iters', 99999999)

        

    def feed_data(self, data):
      # print(data['data'],data['class'])
      datype=torch.float16 if self.fp16 else torch.float32
      self.rna_seq = data['data'].to(self.device).type(datype)
      self.rna_cls = data['class']
      self.rna_ee=data['ensemble_energy'].to(self.device).type(datype) if 'ensemble_energy' in data else None
        

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # prediction
        # while 'paras['target_in_vec']' is the input to target net
        self.output = self.net_g(self.rna_seq)
        # print(self.output.device)
        # print(self.rna_seq.device)


        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter > self.net_g_init_iters and current_iter < self.net_g_end_iters ):
            # tAI loss
            if self.cri_tai:
                l_g_tai = self.cri_tai(self.output,self.rna_seq)
                l_g_total += l_g_tai
                loss_dict['l_tai'] = l_g_tai
            # CAI loss
            if self.cri_cai:
                l_g_cai= self.cri_cai(self.output,self.rna_seq)
                l_g_total += l_g_cai
                loss_dict['l_g_cai'] = l_g_cai
            if self.cri_mfe:
                l_g_mfe='Error!'
                if self.mfe_mode=='ReCompute':
                  l_g_mfe= self.cri_mfe(self.output,self.rna_seq)
                elif self.mfe_mode=='GroundTruth' and self.rna_ee is not None:
                  l_g_mfe= self.cri_mfe(self.output,self.rna_ee)
                # print(self.mfe_mode,self.rna_ee,self.cri_mfe)
                l_g_total += l_g_mfe
                loss_dict['l_g_mfe'] = l_g_mfe
            # protein loss
            l_g_pro = self.cri_pro(self.output,self.rna_seq)
            l_g_total += l_g_pro
            loss_dict['l_pro'] = l_g_pro

            l_g_total.backward()
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g)
        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())
        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)


    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                # prediction
                self.output =  self.net_g_ema(self.rna_seq)
                # self.output =  0.8*self.net_g_ema(self.cataract_image)+0.2*self.cataract_image
        else:
            self.net_g.eval()
            
            with torch.no_grad():
                # prediction
                self.output = self.net_g(self.rna_seq)
                # self.output =  0.8*self.net_g(self.cataract_image)+0.2*self.cataract_image
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']

        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: [] for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)

        if self.opt['val'].get('use_pbar'):
            self.use_pbar = self.opt['val']['use_pbar']
        else:
            self.use_pbar = False
        
        if self.use_pbar: 
            pbar = tqdm(total=len(dataloader), unit='seq',smoothing=0)

        if save_img:
          if self.opt['is_train']:
            save_img_path = osp.join(self.opt['path']['visualization'], 'RNA_sequence')
          else:
            if self.opt['val']['suffix']:
              save_img_path = osp.join(self.opt['path']['visualization'], 
                                          f'RNA_{dataset_name}_{self.opt["val"]["suffix"]}')
            else:
              save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                          f'RNA_{dataset_name}_{self.opt["name"]}')
          open(save_img_path+'-Pre.txt', "w").close()
          open(save_img_path+'-GT.txt', "w").close()
                

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            
            # tentative for out of GPU memory
            rna_original=fea2seq(visuals['rna'],mode=self.opt['network_g']['rna_mode']-1)
            rna_optimal=fea2seq(visuals['opt_rna'],mode=self.opt['network_g']['rna_mode']-1)
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(seq1=rna_optimal, seq2=rna_original)
                    # print(name,opt_,metric_data)
                    metric_res=calculate_metric(metric_data, opt_)
                    if metric_res is not None:
                      self.metric_results[name].append(metric_res)

            if save_img:
                
                txt_write(rna_original,save_img_path+'-GT.txt')
                txt_write(rna_optimal,save_img_path+'-Pre.txt')
            del self.rna_seq
            del self.output
            torch.cuda.empty_cache()

            if self.use_pbar:
                pbar.update(1)
                pbar.set_description(f'Testing....')
        if self.use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] = sum(self.metric_results[metric])/len(self.metric_results[metric]) if len(self.metric_results[metric])>0 else 0
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['rna'] = self.rna_seq.detach().cpu()
        out_dict['opt_rna'] = self.output.detach().cpu()
        
        return out_dict
    
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        # print(self.metric_results.items())
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
      
    def save(self, epoch, current_iter,multi_round=0):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter,round=multi_round, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter,round=multi_round)
        

        
        self.save_training_state(epoch, current_iter)


    # rewrite the saving
    def save_network(self, net, net_label, current_iter,round, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{round}_{net_label}_{current_iter}.pth'
        save_path = osp.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')

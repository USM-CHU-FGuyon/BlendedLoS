import json
import typing
from pathlib import Path
from types import SimpleNamespace
from time import strftime, localtime, time

import numpy as np
import pandas as pd

from database.BlendedICU import BlendedICU


class MultiExperimentConfig:
    def __init__(self,
                 config_file,
                 model,
                 save=True,
                 **kwargs):
        self.save = model == 'tpc'
        self.model = model
        self.config = self._read_config(config_file)
        self.default_config = self._default_config()
        self.model_config = self._model_config(model)
        self.more_configs = {**kwargs}
        self.global_configs = self._global_configs()
        self.experiment_configs, self.run_experiment = self._experiment_configs()
        
        
    def __repr__(self):
        keys = list(self.experiment_configs.keys())
        
        return f'Config for Experiments {keys}'
        
    def __getitem__(self, item):
        return self.experiment_configs[item].run_configs
        
    def __iter__(self):
        return iter(self.experiment_configs)
    
    def _read_config(self, config_file):
        return json.load(open(config_file))
        
    def _check_configs(self):
        expected_config_keys = set(self.default_config.keys())
        given_config_keys = (set(self.config['config'].keys())
                             | set(self.more_configs.keys()))
        
        unexpected_configs = given_config_keys - expected_config_keys
        if unexpected_configs:
            raise ValueError(f'Got Unexpected config entries : '
                             f'{unexpected_configs}')

    def _global_configs(self):
        self._check_configs()
        global_configs = (self.default_config 
                          | self.model_config
                          | self.config['config'] 
                          | self.more_configs)
        return global_configs
        
    def _experiment_configs(self):
        experiment_config_dics = [{
            'Experiment_name': exp_name,
            'config': self.global_configs,
            'runs': exp_params['runs']
            } for exp_name, exp_params in self.config['experiments'].items()]
        
        run_experiment = {exp_name: self.model in exp_params['models']
                          for exp_name, exp_params in self.config['experiments'].items()}
        
        experiment_configs = {d['Experiment_name']: ExperimentConfig(d, save=self.save)
                              for d in experiment_config_dics}
        
        return experiment_configs, run_experiment
        
    @staticmethod
    def _lstm_configs():
        lstm_config =  {
            'exp_name': 'LSTM',
            'n_layers':2,
            'hidden_size':128,
            'learning_rate':0.0001,
            'lstm_dropout_rate':0.2,
            'bidirectional':False,
            'channelwise':False,
            }
        return lstm_config
    
    @staticmethod
    def _transformer_configs():
        return {
            'exp_name': 'Transformer',
            'n_layers': 6,
            'feedforward_size': 256,
            'd_model': 16,
            'n_heads': 2,
            'learning_rate': 0.0001,
            'tf_dropout_rate': 0,
            'positional_encoding': False,
        }
    
    @staticmethod
    def _tpc_configs():
        c =  {'exp_name': 'TPC',
              'n_layers': 9,
              'kernel_size': 4,
              'n_temp_kernels': 12,
              'point_size': 13,
              'learning_rate': 0.0001,
              'temp_dropout_rate': 0.05,
              'share_weights': False
            }
        c['temp_kernels'] = [c['n_temp_kernels']]*c['n_layers']
        c['point_sizes'] = [c['point_size']]*c['n_layers']
        return c
    
    def _model_config(self, model):
        if model=='lstm':
            return self._lstm_configs()
        elif model=='transformer':
            return self._transformer_configs()
        elif model=='tpc':
            return self._tpc_configs()
        else:
            return {}
        
    @staticmethod
    def _default_config():
        default_config = {
            'datasets': ['mimic4', 'eicu', 'amsterdam', 'hirid'],
            'exp_name': 'exp_name',
            'dataset': None,
            'disable_cuda': False,
            'external_validation': None,
            'equal_samples': True,
            'batch_size_test': 32,
            'batch_size_val': 32,
            'batch_size_train': 32,
            "patience": 10,
            'n_epochs': 200,
            'min_los': 0.0208,
            'max_los': 10,
            'shuffle_train': False,
            'percentage_trainval': 100,
            'percentage_test': 100, 
            'percentage_trainval_dic': None,
            'percentage_test_dic': None,
            'task': 'LoS',
            'loss': 'msle',
            'sum_losses': False,
            'ignore_los_dc': True,
            'alpha': 20,
            'max_n_batch': np.inf,
            'no_exp': True,
            'main_dropout_rate': 0.45,
            'L2_regularisation': 0,
            'last_linear_size': 17,
            'batchnorm': False,
            'train_on': None,
            'use_flat': True,
            'use_med': True,
            'use_vitals': True,
            'use_resp': False,
            'use_lab': False,
            'use_US_indicator': False,
            'time_before_pred': 5,
            'time_start': strftime("%y-%m-%d_%H%M%S", localtime(time())),
            'results_dir': 'results/',
            'test_at_each_epoch': False,
            'from_pretrained': False,
            'use_as_pretrained': False,
            'pretrained_model_pth': None,
            'retrain_last_layer_only': True,
            
        }
        return default_config
    

class ExperimentConfig(BlendedICU):
    def __init__(self,
                 experiment_config_dic,
                 save=True):
        super().__init__()
        self.experiment_config_dic = experiment_config_dic
        self.global_config = self.experiment_config_dic['config']
        self.experiment_name = self.experiment_config_dic['Experiment_name']
        self.run_configs = self._get_configs()
        if save:
            self._save()
        
    def __iter__(self):
        return iter(self.run_configs)
        
    def _get_configs(self):
        configs = []
        
        self._check_sanity_use_as_pretrained()
        for run_config in self.experiment_config_dic['runs']:
            config_dic = (self.global_config | run_config)
            config_dic = self._additional_configs(config_dic)
            self._check_sanity_datasets(config_dic)
            config_dic['datareader_pth'] = self.datareader_path(config_dic['train_on'])
            config_dic = self._pretrain_config(config_dic)
            
            configs.append(SimpleNamespace(**config_dic))
        return configs
    
    def _check_sanity_datasets(self, c):
        for dataset in c['train_on']:
            if dataset not in c['datasets']:
                raise ValueError(f'{dataset} not in {c["datasets"]}')
    
    def _additional_configs(self, c):
        
        if ('percentage_trainval_dic' not in c) or (c['percentage_trainval_dic'] is None):
            c['percentage_trainval_dic'] = {
                dataset: c['percentage_trainval'] for dataset in c['train_on']
                }
            
        if ('percentage_test_dic' not in c) or (c['percentage_test_dic'] is None):
            c['percentage_test_dic'] = {
                dataset: c['percentage_test'] for dataset in c['datasets']
                }
        
        if c['train_on'] is None:
            c['train_on'] = list(c['percentage_trainval_dic'].keys())
            
        c['los_task'] = c['task'] in {'LoS', 'multitask'}
        c['mort_task'] = c['task'] in {'mortality', 'multitask'}
        c['jobname'] = self._jobname(c)
        c['basedir'] = f'{c["results_dir"]}/{self.experiment_name}/{c["jobname"]}/{c["task"]}/{c["exp_name"]}'
        c['savedir'] = f'{c["basedir"]}/{c["time_start"]}'
        c['tag'] = '+'.join(c['train_on'])
        return c
    
    def _check_sanity_use_as_pretrained(self):
        def _n_use_as_pretrained():
            n = 0
            for run in self.experiment_config_dic['runs']:
                try:
                    if run['use_as_pretrained']:
                        n +=1
                except KeyError:
                    pass
            return n
        def _any_use_pretrain():
            for run in self.experiment_config_dic['runs']:
                try:
                    if run['from_pretrained']:
                        return True
                except KeyError:
                    pass
            return False
        
        def _any_pretrain_model_pth():
            for run in self.experiment_config_dic['runs']:
                if 'pretrained_model_pth' in run:
                        return True
            return False
        
        N_define_pretrain = _n_use_as_pretrained()
        any_use_pretrain = _any_use_pretrain()
        any_pretrain_model_pth = _any_pretrain_model_pth()
        
        if N_define_pretrain >1:
            raise ValueError(f'Only 1 model per experiment should be used as '
                             f'pretrained, found {N_define_pretrain} in configs of '
                             f'experiment {self.experiment_name}')
        
        if any_use_pretrain & (N_define_pretrain==0) & (not any_pretrain_model_pth):
            raise ValueError(f'Experiment {self.experiment_name} : cannot use'
                             'from_pretrained if no use_as_pretrained was defined.')
        
    
    def _pretrain_config(self, config_dic):
        """!!! use_as_pretrained should be listed in first.
        
        There is no security check for this yet.
        """
        if config_dic['use_as_pretrained']:
            self.pretrained_dataset = '.'.join(config_dic['train_on'])
            self.pretrained_model_pth = config_dic['savedir'] + '/best_model.pth'

        if config_dic['from_pretrained'] and not config_dic['pretrained_model_pth']:
            config_dic['tag'] = self.pretrained_dataset+'->'+config_dic['tag']
            config_dic['pretrained_model_pth'] = self.pretrained_model_pth

        return config_dic
    
    @staticmethod
    def _jobname(c):
        train_txt = "_".join(sorted(c["train_on"]))
        n_txt = c['percentage_trainval']
        retrain_txt = '_retrain' if c['from_pretrained'] else ''
        layers_txt = c['from_pretrained']*('_last_layer' if c['retrain_last_layer_only'] else '_all_layers')
        return  f'{train_txt}_{n_txt}{retrain_txt}{layers_txt}'
    
    
    def _common_configs_runs(self):
        dics = [pd.Series(conf.__dict__) for conf in self.run_configs]
        
        config_table = pd.concat(dics, axis=1)

        loc_table = config_table.copy()
        loc_table = loc_table.map(lambda x: x if isinstance(x, typing.Hashable) else tuple(x))
        
        nunique_entries = loc_table.nunique(1)
        common_configs = config_table.loc[nunique_entries<2, 0].to_dict()
        runs = [*config_table.loc[nunique_entries>1].to_dict().values()]
        return common_configs, runs
    
    def _save(self):
        self.config_savepath = 'results/'+self.experiment_name+'/config.json'
        common_configs, runs = self._common_configs_runs()

        jsonfile = {
            'Experiment_name': self.experiment_name,
            'config': common_configs,
            'runs': runs
            }
        
        Path(self.config_savepath).parent.mkdir(exist_ok=True, parents=True)
        json.dump(jsonfile, open(self.config_savepath, 'w'), indent=2)
        print(f'Saved {self.config_savepath}')

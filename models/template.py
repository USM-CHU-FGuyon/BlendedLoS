from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam

from models.metrics import Metrics
from models.loss import Loss
from database.BlendedICU import BlendedICU
from database.datareader import DataReader
from models.transformer_model import Transformer
from models.tpc_model import TempPointConv
from models.lstm_model import BaseLSTM


class Template(BlendedICU):
    def __init__(self, config, modelconstructor=None):
        super().__init__()
        self.config = config
        self.pretrained = self._get_pretrained()
        self.modelconstructor = modelconstructor
        self.savedir = self.config.savedir
        self.n_epochs = self.config.n_epochs
        self.device = self._get_device()
        self.model_savepath = f'{self.savedir}/best_model.pth'
        self.time_before_pred = config.time_before_pred
        self.task = config.task
        self.los_task = config.los_task
        self.mort_task = config.mort_task
        self.datareaders = self._get_datareaders()
        self.n_train = self.datareaders['train'].n_unique_persons
        self.nfeatures_ts = self.datareaders['train'].F
        self.nfeatures_diag = self.datareaders['train'].D
        self.nfeatures_flat = self.datareaders['train'].n_flat_features
        self.epoch_idx = 0
        self.patience = 5
        self.min_loss = np.inf
        self.loss = self._init_loss()
        self.sets = ['train', 'val', 'test']
        
        self.safetychecker = SafetyChecker(self)
        self.reporter = Reporter(self)
        
        self.model = self._init_model()
        self.n_train_total = self.model.n_train #n_train + pretrained n_train.
        
        self.optimiser = Adam(self.model.parameters(), 
                              lr=self.config.learning_rate, 
                              weight_decay=self.config.L2_regularisation)
        
    def _get_pretrained(self):
        which = ['all', 'last'][self.config.retrain_last_layer_only]
        return which if self.config.from_pretrained else False
        
    def _init_model(self):
        if self.config.pretrained_model_pth:
            print(f'Loading pretrained model {self.config.pretrained_model_pth}')
            model = self._load_pretrained_model()
        else:
            model = self.modelconstructor(config=self.config,
                                      F=self.nfeatures_ts,
                                      D=self.nfeatures_diag,
                                      n_flat_features=self.nfeatures_flat,
                                      device=self.device).to(device=self.device)
        model.n_train += self.n_train
        return model
    
    def _load_pretrained_model(self):
        '''
        Loads a pretrained model. 
        Only point_final_los and point_final_mort will be trainable layers.
        '''
        
        def _set_trainable_layers(model):
            if self.config.retrain_last_layer_only:
                trainable_layers = ['point_final_los.weight',
                                    'point_final_los.bias',]
            else:
                trainable_layers = [name for name, _ in model.named_parameters()]
            print('Trainable :')
            for name, param in model.named_parameters():
                train = name in trainable_layers
                param.requires_grad = train
                if train:
                    print(name)
            print()
                
        model = torch.load(self.config.pretrained_model_pth)
        
        _set_trainable_layers(model)
        
        return model
    
    def _init_loss(self):
        return Loss(self.los_task,
                    self.mort_task,
                    loss_type='msle',
                    sum_losses=self.config.sum_losses,
                    alpha=self.config.alpha,
                    ignore_los_dc=self.config.ignore_los_dc)
        
    def _get_datareaders(self):
        trainvaltest = ['train', 'val', 'test'] 
        datareaders = {
            c: DataReader(c, self.config, self.device) for c in trainvaltest           
            }
        self.datareader_incomplete_init = any(d.incomplete_init for d in datareaders.values())
        return datareaders
        
    def _get_device(self):
        if (not self.config.disable_cuda) and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def _out_to_pandas(self,
                       y_los,
                       y_hat_los,
                       y_mort,
                       y_hat_mort,
                       patientids,
                       mask):

        batch = {
            'y_hat_los': self._remove_padding(y_hat_los, mask),
            'y_los': self._remove_padding(y_los, mask),
            'y_hat_mort': self._remove_padding(y_hat_mort, mask),
            'y_mort': y_mort.cpu().numpy(),
            'patientids': patientids
            }

        df_batch = pd.DataFrame(batch).set_index('patientids')
        self.df_batch = df_batch
        
        if self.df_epoch.empty:
            return df_batch
        return pd.concat([self.df_epoch, df_batch])
    
    def _remove_padding(self, y, mask):
        """
            Filters out padding from tensor of predictions or labels

            Args:
                y: tensor of los predictions or labels
                mask (bool_type): tensor showing which values are padding (0)
                                    and which are data (1)
        """
        
        y = (y.where(mask, torch.tensor(float('nan'))
              .to(device=self.device)).flatten().detach().cpu().numpy())
        return y[~np.isnan(y)]
    
    def train_one_epoch(self):   
        self.trainvaltest = 'train'
        self.model.train()
        
        self._run_batches()

        self.reporter.epoch_end()
    
    def validate_one_epoch(self):
        self.trainvaltest = 'val'
        self.model.eval()
        with torch.no_grad():
            epoch_val_loss = self._run_batches()

        self.reporter.epoch_end()
        self.reporter.plot_loss()

        if epoch_val_loss < self.min_loss:
            self.min_loss = epoch_val_loss
            self.best_epoch = self.epoch_idx
            self.reporter.save_model()
            
        stop_training = self.epoch_idx>self.best_epoch+self.patience
        
        return stop_training
            
    def test(self):
        self.trainvaltest = 'test'
        self.model = torch.load(self.model_savepath)
        self.model.eval()
        with torch.no_grad():
            self._run_batches()

        
        self.reporter.epoch_end()
        

    def run(self):
        if self.datareader_incomplete_init:
            raise ValueError('Datareader initialization was incomplete')
        while self.epoch_idx < self.n_epochs:
            self.epoch_idx += 1
            self.train_one_epoch()
            stop_training = self.validate_one_epoch()
            if (self.config.test_at_each_epoch 
                or (self.epoch_idx==self.n_epochs)
                or stop_training):
                self.test()
                
                if stop_training:
                    break

    def _run_batches(self):
        
        training = self.trainvaltest=='train'
        datareader = self.datareaders[self.trainvaltest]
        
        for batch_idx, batch in enumerate(datareader.batch_gen()):
            (patient_batch,
             repeated_patientids,
             true_los,
             true_mort,
             repeated_true_mort,
             flat, 
             ts_padded,
             msk,
             seq_lengths) = batch

            if training:
                self.optimiser.zero_grad()
            
            pred_los, pred_mort = self.model(ts_padded,
                                             flat,
                                             time_before_pred=self.time_before_pred)

            self.safetychecker.nans_in_input(ts_padded,
                                             flat,
                                             pred_los,
                                             batch,
                                             batch_idx)    

            #print(f'pred_los {pred_los.min():.3f} - {pred_los.max():.3f}')

            mort_loss, LoS_loss, loss = self.loss(pred_los,
                                                  pred_mort,
                                                  true_los,
                                                  true_mort,
                                                  msk,
                                                  seq_lengths)
            self.safetychecker.nan_loss(loss,
                                        pred_los,
                                        true_los,
                                        msk,
                                        true_mort,
                                        pred_mort)
            if training:
                loss.backward()
                self.optimiser.step()
                self.optimiser.zero_grad()

            self.reporter.epoch_loss(loss, LoS_loss, mort_loss)
            
            self.df_epoch = self._out_to_pandas(true_los,
                                                pred_los,
                                                repeated_true_mort,
                                                pred_mort,
                                                repeated_patientids,
                                                msk)

        epoch_loss = pd.DataFrame(self.epoch_loss).mean().replace(0, np.nan)
        idx = (self.n_train_total, self.epoch_idx, self.trainvaltest)
        self.loss_df.loc[idx] = epoch_loss
        return self.loss_df.loc[idx, 'tot']


class SafetyChecker:
    """
    class that countains safety checks for the computation template.
    """
    def __init__(self, template):
        self.template = template
        
    def nans_in_input(self, padded, flat, y_hat_los, batch, batch_idx):
        if torch.isnan(padded).any():
            nacount = padded.isnan().sum()
            self.template.padded, self.template.batch = padded, batch
            raise ValueError(f'Timeseries input contains {nacount} NaN in batch'
                             f'{batch_idx}, see self.padded and self.batch')

        if flat.isnan().any():
            raise ValueError(
                f'Flat input contains NaN in batch {batch_idx}')

        if y_hat_los.isnan().any():
            raise ValueError(f'Model diverged in batch {batch_idx}')
    
    def nan_loss(self, loss, pred_los, true_los, msk, true_mort, pred_mort):
            if loss.isnan():
                self.loss = loss
                self.pred_los = pred_los
                self.true_los = true_los
                self.msk = msk
                self.true_mort = true_mort
                self.pred_mort = pred_mort
                raise ValueError("Loss is NaN")

    def check_mode_sanity(self):
        accepted_tasks = {'LoS', 'mortality', 'multitask'}
        if not self.template.trainvaltest in self.template.sets:
            raise ValueError(
                f"Argument 'traintestval' should be 'train', 'test' or 'val',"
                f" not '{self.template.trainvaltest}'")
        if not self.template.task in accepted_tasks:
            raise ValueError(f"Argument 'task' should be in {accepted_tasks}"
                             f" not '{self.template.task}'")


class Reporter:
    '''
    class for handling all reporting, metric computation and saving of results
    '''
    def __init__(self, template):
        self.template = template
        self.reset_epoch()
        self._init_savedir()
        self.init_loss_df()
        self.init_metrics_df()
        self.los_metrics = Metrics(key_true='y_los',
                                   key_pred='y_hat_los',
                                   ignore_los_dc=self.template.config.ignore_los_dc)
        self.mort_metrics = Metrics(key_true='y_mort', key_pred='y_hat_mort')
        
    def _init_savedir(self):
        Path(self.template.savedir).mkdir(parents=True, exist_ok=True)
        
    def init_loss_df(self):
        loss_df_idx = pd.MultiIndex.from_tuples([], names=('n_train',
                                                           'epoch',
                                                           'step'))
        loss_df = pd.DataFrame(index=loss_df_idx,
                               columns=['mort', 'LoS', 'tot'])
        self.template.loss_df = loss_df
    
    def init_metrics_df(self):     
        mux = pd.MultiIndex.from_tuples([], names=('n_train',
                                                   'task',
                                                   'epoch',
                                                   'set',
                                                   'source_dataset',
                                                   'pretrained'))
        
        self.template.metrics_df = pd.DataFrame(columns=['mad',
                                                         'mse',
                                                         'mape',
                                                         'msle',
                                                         'auc'],
                                                index=mux)
    
    def reset_epoch(self):
        self.template.epoch_loss = {'mort': [], 'LoS': [], 'tot': []}
        self.template.df_epoch = pd.DataFrame(columns=['y_los',
                                                       'y_hat_los',
                                                       'y_mort',
                                                       'y_hat_mort'], 
                                              index=pd.Index([], name='patientid'))

    def epoch_loss(self, loss, LoS_loss, mort_loss):
        self.template.epoch_loss['mort'].append(mort_loss.detach().item())
        self.template.epoch_loss['LoS'].append(LoS_loss.detach().item())
        self.template.epoch_loss['tot'].append(loss.detach().item())

    def epoch_end(self):

        self.template.df_epoch['eval_on'] = self.template.df_epoch.index.map(lambda x: x.split('-')[0])
        self.template.df_epoch = self.template.df_epoch.set_index('eval_on', append=True)
        
        if self.template.trainvaltest=='test':
            self.save_test_pred()
        
        self.save_metrics()
        self.save_loss()
        self.reset_epoch()
        
    def save_metrics(self):
        self.template.safetychecker.check_mode_sanity()
        if self.template.los_task:
            self.compute_epoch_metrics_los()
        if self.template.mort_task:
            self.compute_epoch_metrics_mort()
        metrics = self.template.metrics_df.map(lambda x: f'{x:.3f}')
        self._save_to_csv(metrics, f'metrics_{self.template.task}.csv')
        
    def save_loss(self):
        self._save_to_csv(self.template.loss_df, f'loss_{self.template.task}.csv')
        
    def _save_to_csv(self, data, path):
        savepath = f'{self.template.savedir}/{path}'
        data.to_csv(savepath)
        
    def save_model(self):
        print(f'Best epoch ! saving model @ {self.template.model_savepath}')
        torch.save(self.template.model, self.template.model_savepath)

    def compute_epoch_metrics_mort(self):
        df_epoch = self.template.df_epoch
        groups = df_epoch.groupby('eval_on')
        _metrics_mort = pd.DataFrame({
                                'auc': groups.apply(self.mort_metrics.auc), 
                                'bce': groups.apply(self.mort_metrics.bce), 
                                })
        for dataset, met in _metrics_mort.iterrows():
            idx = (self.template.n_train_total,
                   'mortality',
                   self.template.epoch_idx,
                   self.template.trainvaltest,
                   dataset,
                   self.template.pretrained)
            self.template.metrics_df.loc[idx] = met

    def compute_epoch_metrics_los(self):
        '''Ignore length of stay for deceased patients.'''
        df_epoch = self.template.df_epoch
        df_epoch.loc[df_epoch.y_mort==1, ['y_hat_los', 'y_los']] = np.nan
        groups = df_epoch.groupby('eval_on')
        
        self.groups = groups

        _metrics_los =  pd.DataFrame({
                                'mad': groups.apply(self.los_metrics.mad), 
                                'mse': groups.apply(self.los_metrics.mse), 
                                'mape': groups.apply(self.los_metrics.mape), 
                                'msle': groups.apply(self.los_metrics.msle), 
                                })

        for dataset, met in _metrics_los.iterrows():
            idx = (self.template.n_train_total,
                   'LoS',
                   self.template.epoch_idx,
                   self.template.trainvaltest,
                   dataset,
                   self.template.pretrained)
            self.template.metrics_df.loc[idx] = met

    def save_test_pred(self):

        self._save_to_csv(self.template.df_epoch, 'test_predictions.csv')
        try:
            if self.template.los_task:
                df = (self.template.df_epoch
                          .rename(columns={'y_hat_los': 'pred_los',
                                           'y_los': 'label'})
                          .drop(columns=['y_hat_mort']))
    
                self._save_to_csv(df, 'test_predictions_los.csv')
    
            if self.template.mort_task:
                df = (self.template.df_epoch
                          .rename(columns={'y_hat_mort': 'pred_mort',
                                           'y_mort': 'label'})
                          .drop(columns=['y_los', 'y_hat_los']))
                
                self._save_to_csv(df, 'test_predictions_mort.csv')
        except AttributeError:
            pass

            
    def plot_loss(self):
        cmaps = {name: matplotlib.colormaps[name] for name in {'Purples',
                                                               'Oranges',
                                                               'Greens'}}
        def _plot_sublosses(ax, step, cmaps, level):
            df = self.template.loss_df.xs(step, level='step')
            epochs = df.index.get_level_values('epoch')
            ax.plot(epochs, df['mort'],
                    label=f'{step} mort',
                    c=cmaps['Purples'](level))
            ax.plot(epochs, df['LoS'],
                    label=f'{step} LoS',
                    c=cmaps['Oranges'](level))
            ax.plot(epochs, df['tot'],
                    label=f'{step} total',
                    c=cmaps['Greens'](level))
            return [f'{step} mort', f'{step} LoS', f'{step} total']
        
        fig, ax = plt.subplots()
        labels_train = _plot_sublosses(ax, 'train', cmaps, level=3/4)
        labels_val = _plot_sublosses(ax, 'val', cmaps, level=1/2)


        for line, name in zip(ax.lines, labels_train+labels_val):
            y = line.get_ydata()[-1]
            if not np.isnan(y):
                ax.annotate(name,
                            xy=(1,y),
                            xytext=(6,0),
                            color=line.get_color(), 
                            xycoords=ax.get_yaxis_transform(),
                            textcoords="offset points",
                            size=10,
                            va="center")

        title = self.template.config.exp_name + ' ' + ' '.join(self.template.config.train_on)

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.yaxis.tick_right()
        ax.yaxis.set_ticks_position('both')
        ax.set_ylim((0, None))
        plt.tight_layout()
        plt.savefig(f'{self.template.savedir}/loss.png')
        plt.show()
        
        plt.close()
        
class TransformerTemplate(Template):
    def __init__(self, config):
        super().__init__(config, Transformer)

class TPCTemplate(Template):
    def __init__(self, config):
        super().__init__(config, TempPointConv)
        
class LSTMTemplate(Template):
    def __init__(self, config):
        super().__init__(config, BaseLSTM)
        
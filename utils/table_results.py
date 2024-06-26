from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd

from models.config import ExperimentConfig


class TableResults:
    """Loads the result table from a single run."""
    def __init__(self, experiment_directory, config):
        self.ordered_indices = ['task',
                                'percent_train_data',
                                'n_train',
                                'train_on',
                                'model',
                                'source_dataset',
                                'set',
                                'epoch',
                                'pretrained']
        self.task = config.task
        self.job = config.jobname
        self.train_on = config.train_on
        self.percentage_trainval = config.percentage_trainval
        self.basedir = experiment_directory
        self.metrics_fname = f'metrics_{self.task}.csv'
        self.loss_fname = f'Loss_{self.task}.csv'
        self.pred_fname = 'test_predictions.csv'
        
        self.result_dirs = self._result_dirs()
        self.metrics = self._metrics()
        self.losses = self._losses()

    def _result_dirs(self):
        p = Path(self.basedir+self.job+'/'+self.task+'/')

        if not p.is_dir():
            return {}

        model_dirs = {pth.stem: pth for pth in p.iterdir()}

        for dirpath in model_dirs.values():
            for pth in dirpath.iterdir():
                if len({*pth.rglob('*csv')})==0:
                    inp = input(f'No csv found in {pth}, remove it ? y/[n]')
                    if inp.lower() =='y':
                        shutil.rmtree(pth)

        for p in model_dirs.values():
            if len(list(p.iterdir()))!=1:
                raise ValueError(f'Several or no directory in {p}')

        return {key: list(v.iterdir())[0] for key, v in model_dirs.items()}

    def _losses(self):
        losses = [self._read_loss(pth/self.loss_fname, model)
                  for model, pth in self.result_dirs.items()]
        if losses:
            losses = pd.concat(losses)
        else :
            mux=pd.MultiIndex.from_tuples([], names=('model','n_train', 'epoch', 'step'))
            losses = pd.DataFrame(index=mux)
        losses['train_on'] = '+'.join(self.train_on)
        losses = (losses.set_index('train_on', append=True)
                        .reorder_levels(['n_train',
                                         'train_on',
                                         'model',
                                         'step',
                                         'epoch']))
        return losses
    
    @staticmethod
    def _empty_df(index_names):
        emptyidx = pd.MultiIndex.from_tuples([], names=index_names)
        return  pd.DataFrame(index=emptyidx)
    
    def _read_loss(self, pth, modelname):
        print('  Reading', pth)
        index_names = ['n_train', 'epoch', 'step']
        try:
            loss = pd.read_csv(pth, index_col=index_names)
        except FileNotFoundError:
            loss = self._empty_df(index_names)
        
        loss['model'] = modelname
        loss = loss.set_index('model', append=True)
        return loss

    def _read_metrics(self, pth, modelname):
        print('  Reading', pth)
        index_names = ['n_train', 'epoch', 'set', 'source_dataset', 'task', 'pretrained']
        try:
            metrics = pd.read_csv(pth, index_col=index_names)
        except FileNotFoundError:
            metrics = self._empty_df(index_names)
        metrics['model'] = modelname
        metrics = metrics.set_index('model', append=True)
        return metrics

    def _metrics(self):
        print()
        metrics = [self._read_metrics(pth/self.metrics_fname, model) 
                   for model, pth in self.result_dirs.items()]
        if metrics:
            metrics = pd.concat(metrics)
            metrics['train_on'] = '+'.join(self.train_on)
            metrics['percent_train_data'] = self.percentage_trainval
            metrics = (metrics
                       .set_index(['train_on', 'percent_train_data'], append=True)
                       .reorder_levels(self.ordered_indices))
        else :
            metrics = self._empty_df(self.ordered_indices)
        metrics.index = metrics.index.rename({'source_dataset': 'eval_on'})
        return metrics


class ResultReader:
    """Loads results of a full experiment (that may contain several runs)"""
    def __init__(self,
                 experiment_directory,
                 jobnames=None,
                 los_metric='msle',
                 mort_metric='auc'):
        
        self.experiment_directory = experiment_directory+'/'
        self.los_metric = los_metric
        self.mort_metric = mort_metric
        self.config = self._configs()
        
        self.tableresults = self._load_metrics(jobnames)
        
        self.metrics = self._metrics_table()
        self.losses = self._losses_table()
        
        self.printed_datasets = {
            'eicu': 'eICU',
            'amsterdam': 'AmsterdamUMC',
            'hirid': 'HiRID',
            'mimic4': 'MIMIC-IV',
            'mimic3': 'MIMIC-III'
            }
        
    def _metrics_table(self):
        table = pd.concat([tab.metrics for tab in self.tableresults.values()])
        return table
    
    def _losses_table(self):
        table = pd.concat([tab.losses for tab in self.tableresults.values()])
        return table
    
    def _load_metrics(self, jobnames):
        if jobnames is None:
            return {c.jobname: TableResults(self.experiment_directory, c)
                    for c in self.config}
        else:
            return {c.jobname: TableResults(self.experiment_directory, c)
                    for c in self.config if c.jobname in jobnames}
        
    def __getitem__(self, item):
         return self.tableresults[item]
        
    def _configs(self):
        config_pth = self.experiment_directory+'config.json'
        config = json.load(open(config_pth, 'r'))
        return ExperimentConfig(config)
    

class Latexify:
    """Toolbox for printing tables into Latex code."""
    @staticmethod
    def _format_latex(s):
        return (s.replace('\\cline{1-4}\n\\bottomrule', '\\bottomrule')
                .replace('\\cline{1-5}\n\\bottomrule', '\\bottomrule')
                .replace('\\cline{1-4}', '\\midrule')
                .replace('\\cline{1-5}', '\\midrule')
                .replace('\multirow[t]', '\multirow[l]')
                .replace('eicu', r'\eicu')
                .replace('hirid', r'\hirid')
                .replace('mimic3', r'\mimicthree')
                .replace('mimic4', r'\mimicfour')
                .replace('amsterdam', r'\amsterdam')
                .replace('LoS', r'\rlos')
                .replace('mortality', 'Mortality')
                .replace('composite', 'Composite')
                .replace('internal', 'Internal')
                .replace('external', 'External')
                )
    
    @staticmethod
    def _textbf_values(s, values):
        #Not extremely robust but does the textbf for the minimum value.
        for val in values:
            num = f'{val:.3f}'
            s = s.replace(num, fr'\textbf{{{num}}}')
        return s
    
    def cohort_experiment(self, tab):
        tab = (tab.rename_axis('', axis=1)
               .rename_axis(index={'train_on': 'train on',
                                   'n_train': 'patients'}))
        
        s = (tab.to_latex(float_format="{:.3f}".format,
                          column_format='llccc')
             .replace('\\cline{1-4}\n', '')
             
             .replace(r'\multirow[t]{3}', r'\midrule'+'\n'+r'\multirow[t]{3}'))
        
        s = self._format_latex(s)
        s = self._textbf_values(s, [tab['LoS'].min()])
        s = self._textbf_values(s, [tab['Mortality'].max()])
        return s
        
        
    def dataset_benchmark(self, tab):
        tab = tab.rename_axis('', axis=1)
        tab.index = tab.index.rename({'eval_on': 'Evaluation set'})
        
        tab.loc['LoS'] = tab.xs('LoS', level='task', drop_level=False).round(1)
        tab.loc['mortality'] = tab.xs('mortality', level='task', drop_level=False).round(3)
        s = (tab.to_latex(float_format="{:.3f}".format,
                          column_format='llccc')
             )
        s  = self._format_latex(s)
        
        s = self._textbf_values(s, tab.loc['LoS'].min(1).values)
        s = self._textbf_values(s, tab.loc['mortality'].max(1).values)
        return s
    
    def main_experiment(self, tab):
        tab.columns = tab.columns.map(lambda x: '\makecell{'+ x.replace(',',r'\\')+ '}')
        tab = tab.rename_axis('', axis=1)
        tab.index = tab.index.rename({'eval_on': 'Evaluation set'})
        s = (tab.to_latex(float_format="{:.3f}".format,
                          column_format='llcccc')
             .replace('\midrule', '')
             .replace('\\cline{1-6}\n', '')
             .replace(r'multirow', r'midrule'+'\n'+r'\multirow')
             .replace('->', r'$\rightarrow$')
             .replace('Transfer Learning', r'Transfer Learning\textsuperscript{1}')
             .replace('Data Pooling', r'Data Pooling\textsuperscript{2}')
             .replace(r'{*}{amsterdam}', r'{*}{\makecell[l]{Specialization on\\amsterdam}}')
             .replace(r'{*}{mimic4}', r'{*}{\makecell[l]{Specialization on\\mimic4}}')
             .replace('NaN', '')
             )
        s  = self._format_latex(s)
        s = self._textbf_values(s, tab.xs('LoS', level='metric').min(1).values)
        s = self._textbf_values(s, tab.xs('composite', level='metric').min(1).values)
        s = self._textbf_values(s, tab.xs('mortality', level='metric').max(1).values)
        return s
    
    def model_benchmark(self, tab):
        tab = tab.rename_axis('', axis=1)
        tab.index = tab.index.rename({'eval_on': 'Evaluation set',
                                      'task': 'Task'})
        s = (tab.to_latex(float_format="{:.3f}".format,
                          column_format='llccc'))
        s  = self._format_latex(s)
        
        s = self._textbf_values(s, tab.loc['LoS'].min(1).values)
        s = self._textbf_values(s, tab.loc['mortality'].max(1).values)
        return s
    

class Results(ResultReader):
    """Uses the raw results of an experiment from ResultsReader pto produce
    the tables."""
    def __init__(self,
                 experiment_directory,
                 jobnames=None,
                 task=None,
                 los_metric='msle',
                 mort_metric='auc'):
        super().__init__(experiment_directory,
                         jobnames,
                         los_metric,
                         mort_metric)
        self.ordered_models = ['LSTM', 'Transformer', 'TPC']
        self.latexify = Latexify()

    def tab_main_experiment(self):
        
        tab = (self.metrics[[self.los_metric, self.mort_metric]]
               .droplevel(['epoch', 'percent_train_data', 'task', 'model'])
               .xs('test', level='set')
               .rename(columns={self.los_metric: 'LoS',
                                self.mort_metric: 'mortality'})
               .rename_axis('task', axis=1)
               .stack()
               .unstack('eval_on')
               .reorder_levels(['task', 'n_train', 'train_on', 'pretrained'])
               .rename_axis(['metric', 'n_train', 'train_on', 'pretrained'])
               .sort_index())

        tab = (tab
               .loc[tab.index.get_level_values('pretrained')!='last'])
    
        return tab
    
    def tab_ntrain(self):
        tab = (self.metrics
               .xs('amsterdam+hirid+mimic4+eicu', level='train_on')
               .xs('TPC', level='model')
               .xs('test', level='set')
               .droplevel('epoch')
               [[self.los_metric, self.mort_metric]]
               .groupby(level=('n_train', 'eval_on'))
               .mean()
               .rename(columns={self.los_metric: 'LoS',
                                self.mort_metric: 'mortality'})
               .rename_axis('task', axis=1)
               .stack()
               .unstack('eval_on')
               .swaplevel()
               .sort_index())
        
        tab['total'] = tab.mean(1)
        tab = tab.drop(columns=['amsterdam',
                                'eicu',
                                'hirid',
                                'mimic4'])
        return tab
        
    
    def tab_dataset_benchmark(self, n_train=None):
        multiple_n_train = (self.metrics
                            .reset_index('n_train')
                            .groupby('train_on')
                            .n_train.nunique().max()>1)
        
        n_patients = self.metrics.index.get_level_values('n_train').unique()[0]
        print(f"n_patients {n_patients}")
        
        if multiple_n_train:
            raise ValueError('multiplentrain')
        tab = (self.metrics
               [[self.los_metric, self.mort_metric]]
               .xs('test', level='set')
               .droplevel(('epoch', 'n_train'))
               .rename(columns={self.los_metric: 'LoS',
                                self.mort_metric: 'mortality'})
               .droplevel('task')
               .groupby(level=('train_on', 'model', 'eval_on'))
               .first()
               .rename_axis('task', axis=1))
        
        idx_train_on = tab.index.get_level_values('train_on')
        idx_eval_on = tab.index.get_level_values('eval_on')
        idx_internal = idx_train_on==idx_eval_on

        tab['validation_type'] = np.where(idx_internal, 'internal', 'external')

        tab = (tab.set_index('validation_type', append=True)
               .groupby(level=('model', 'eval_on', 'validation_type'))
               .mean()
               .stack()
               .unstack('validation_type')
               .reorder_levels(['task', 'model', 'eval_on'])
               .sort_index()
               [['internal', 'external']])
        
        if tab.index.get_level_values('model').nunique()==1:
            tab = tab.droplevel('model')
        
        latext = self.latexify.dataset_benchmark(tab)
        print('\n\n', latext, '\n\n')
        return tab
        
    def tab_model_benchmark(self, n_train=None):
        """
        Experiment where models are trained and evaluated on the same 
        datasets.
        Models are evaluated on their internal validation performance on the 
        full dataset.
        """
        n_patients = self.metrics.index.get_level_values('n_train').unique()[0]
        print(f"n_patients {n_patients}")
        
        if n_train is None:
            n_train = self.metrics.index.get_level_values('n_train').max()
        
        tab = (self.metrics
               .xs('test', level='set')
               .droplevel(['epoch', 'n_train'])
               .groupby(level=('model', 'eval_on', 'task'))
               .mean()
               .reindex([self.los_metric, self.mort_metric], axis=1)
               .rename(columns={self.los_metric: 'LoS',
                                self.mort_metric: 'mortality'})
               .droplevel('task')
               .groupby(level=('model', 'eval_on'))
               .first()
               .T
               .stack(future_stack=True)
               .reindex(self.ordered_models, axis=1)
               .rename_axis(('task', 'eval_on')))

        latext = self.latexify.model_benchmark(tab)
        print('\n\n', latext, '\n\n')
        return tab

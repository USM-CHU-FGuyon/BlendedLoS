from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

from utils.table_results import Results, Latexify


class Experiments:
    '''
    The Experiment class searches for results in the results/ directory.
    It creates a table or figure for each of the four experiments:
        model_benchmark
        dataset_benchmark
        main_experiment
        training_size_study
        
    Latex code for tables are printed.
    '''
    def __init__(self,
                 model_benchmark='model_benchmark',
                 dataset_benchmark='dataset_benchmark',
                 dataset_benchmark_nomed='dataset_benchmark_nomed',
                 training_size='training_size_study',
                 main_experiment='main_experiment',
                 los_metric='mape',
                 mort_metric='auc'):
        
        self.background_color='whitesmoke'
        self.gridcolor='lightgray'
        self.model_benchmark_dirname = model_benchmark
        self.dataset_benchmark_dirname = dataset_benchmark
        self.dataset_benchmark_nomed_dirname = dataset_benchmark_nomed
        self.training_size_dirname = training_size       
        self.main_experiment_dirname = main_experiment
        self.los_metric = los_metric
        self.mort_metric = mort_metric
        
        
    def figure_perf_ntrain(self, tab_n_train):

        fig, ax_composite = plt.subplots(figsize=(8,4.5))
        ax_mort = ax_composite.twinx()
        ax_los = ax_composite.twinx()
        ax_mort.grid(False)
        ax_los.grid(False)
        ax_composite.grid(color=self.gridcolor)
        ax_composite.grid(False, axis='y')
        lcolor = 'tab:blue'
        rcolor = 'tab:orange'
        composite_color = 'tab:gray'

        tab_los = tab_n_train.loc['LoS', 'total']
        tab_mort = tab_n_train.loc['mortality', 'total']
        tab_composite = tab_n_train.loc['composite', 'total']

        ax_los.plot(tab_los, color=lcolor, marker='x')
        ax_mort.plot(tab_mort, color=rcolor,  marker='x')
        ax_composite.plot(tab_composite, color=composite_color, marker='x')
        
        ax_los.annotate('RLoS',
                        (tab_los.index[-1], tab_los.values[-1]+0.15),
                        color=lcolor,
                        rotation=0,
                        ha='right')
        
        ax_mort.annotate('Mortality',
                     (tab_mort.index[-1], tab_mort.values[-1]+0.005),
                     ha='right',
                     rotation=0,
                     color=rcolor)
        
        ax_composite.annotate('Composite',
                      (tab_composite.index[-1], tab_composite.values[-1]+0.013),
                      ha='right',
                      rotation=-5,
                      color=composite_color)
        
        ax_los.set_facecolor(self.background_color)
        
        ax_composite.set_ylabel('Composite metric (arbitrary unit)')
        ax_mort.set_ylabel('Mortality (AUROC)')
        ax_los.set_ylabel(f'Remaining length of stay ({self.los_metric.upper()})')
        ax_composite.set_xlabel('Number of patients for training')
        
        ax_los.yaxis.label.set_color(lcolor)
        ax_mort.yaxis.label.set_color(rcolor)
        ax_composite.yaxis.label.set_color(composite_color)
        
        ax_mort.set_ylim(None, 0.85)
        ax_composite.set_ylim(0.7, 1.6)
        ax_composite.set_xlim(0, tab_los.index.max()*1.1)
        
        ax_los.tick_params(axis='y', colors=lcolor)
        ax_mort.tick_params(axis='y', colors=rcolor)
        ax_composite.tick_params(axis='y', colors=composite_color)
        
        ax_mort.spines['right'].set_position(('outward', 45))
        ax_los.spines['right'].set_position(('outward', 0))
        
        fig.tight_layout()
        savepath = 'figures/perf_training_size.png'
        Path(savepath).parent.mkdir(exist_ok=True)
        fig.savefig(savepath)
        print(f'Saved {savepath}')
        return tab_n_train
        
    def _add_composite_criterion(self, tab, baseline):
        baselined_los = tab.loc['LoS']/baseline.loc['LoS'] 
        baselined_mort = (1 - tab.loc['mortality'])/(1 - baseline.loc['mortality'])
        composite = (baselined_los + baselined_mort)/2
        
        composite = (composite.assign(task='composite')
                     .set_index('task', append=True)
                     .reorder_levels(['task', *composite.index.names])
                     .rename_axis(['metric', *composite.index.names]))
        tab = pd.concat([tab, composite])
        return tab
        
    def _remove_not_applicable(self, tab):
        tab.loc[tab.index.get_level_values('pretrained')=='all', ['mimic4', 'hirid', 'eicu']]= np.nan
        return tab
    
    def get_tables(self):
        self.tab_model_benchmark = self.table_model_benchmark()
        self.tab_dataset_benchmark = self.table_dataset_benchmark()
        self.baseline_main_experiment = self.get_baseline_main_experiment()
        self.baseline_ntrain = self.get_baseline_ntrain()
        
        self.tab_main_experiment = (self.table_main_experiment()
                                    .pipe(self._add_composite_criterion,
                                          baseline=self.baseline_main_experiment)
                                    .pipe(self._remove_not_applicable))
        
        self.tab_main_experiment_formatted = self.format_tab_main_experiment()
        
        self.tab_perf_ntrain = (self.table_ntrain()
                                .pipe(self._add_composite_criterion,
                                      baseline=self.baseline_ntrain))
             
        self.figure_perf_ntrain(self.tab_perf_ntrain)
        
    def get_baseline_main_experiment(self):
        return (self.tab_dataset_benchmark['internal']
                .unstack())

    def get_baseline_ntrain(self):
        return (self.tab_dataset_benchmark['internal']
                .unstack()
                .mean(1)
                .rename('total')
                .to_frame())

    def table_model_benchmark(self):
        experiment_directory = f'results/{self.model_benchmark_dirname}/'
        rp = Results(experiment_directory, los_metric=self.los_metric)
        return rp.tab_model_benchmark()

    def table_dataset_benchmark(self):
        experiment_directory = f'results/{self.dataset_benchmark_dirname}/'
        rp = Results(experiment_directory, los_metric=self.los_metric)
        return rp.tab_dataset_benchmark()

    def table_ntrain(self):
        experiment_directory = f'results/{self.training_size_dirname}/'
        rp = Results(experiment_directory, los_metric=self.los_metric)
        return rp.tab_ntrain()

    def table_main_experiment(self):
        rp = Results(
            experiment_directory=f'results/{self.main_experiment_dirname}/',
            los_metric=self.los_metric)
        return rp.tab_main_experiment()

    def table_annexe_benchmark_med(self):
        rp_med = Results(f'results/{self.dataset_benchmark_dirname}/',
                         los_metric=self.los_metric)
        tab_with_med = rp_med.tab_dataset_benchmark()

        rp_nomed = Results(f'results/{self.dataset_benchmark_nomed_dirname}/',
                         los_metric=self.los_metric)
        tab_no_med = rp_nomed.tab_dataset_benchmark()

        return tab_with_med, tab_no_med

    def format_tab_main_experiment(self):
        
        def new_index(tab_main_experiment):
            idx_run = (tab_main_experiment.index
                         .droplevel(['metric', 'n_train'])
                         .map({('amsterdam', False): 'amsterdam',
                               ('mimic4', False): 'mimic4',
                               ('amsterdam', 'all'): 'Transfer Learning',
                               ('amsterdam+mimic4', False): r'Data Pooling'})
                         .rename('run'))
            tab_main_experiment = (tab_main_experiment
                                   .assign(run=idx_run)
                                   .set_index('run', append=True)
                                   .droplevel(['train_on', 'pretrained']))
            return tab_main_experiment

        def latex_index(tab_main_experiment):
            idx = (tab_main_experiment.index.get_level_values('run')
                   +r', N='
                   +tab_main_experiment.index.get_level_values('n_train').astype(str))
            tab_main_experiment = (tab_main_experiment
                                   .assign(run_ltx=idx)
                                   .set_index('run_ltx', append=True)
                                   .droplevel(['n_train', 'run']))
            return tab_main_experiment

        lt = Latexify()
        tab = (self.tab_main_experiment
               .assign(Generalization=lambda x: x[['eicu', 'hirid']].mean(1))
               .pipe(new_index)
               .pipe(latex_index)
               .stack().unstack('run_ltx')
               .swaplevel()
               .sort_index()
               .reindex(['amsterdam', 'mimic4', 'Generalization'], level=0)
               .reindex(['mortality', 'LoS', 'composite'], level=1)
               [
                   ['amsterdam, N=3574',
                    'mimic4, N=10915',
                    'Transfer Learning, N=14489',
                    'Data Pooling, N=14489'
                ]])
        latext = lt.main_experiment(tab)
        print('\n\n', latext, '\n')
        return tab
        

self = Experiments(
    main_experiment='main_experiment_ok',
    model_benchmark='model_benchmark_ok',
    dataset_benchmark='dataset_benchmark_ok',
    dataset_benchmark_nomed='dataset_benchmark_nomed',
    #dataset_benchmark_nomed='dataset_benchmark_nomed_75',
    training_size='training_size_study_ok',
    )

self.get_tables()

tab_med, tab_nomed = self.table_annexe_benchmark_med()

tab_nomed['diff_'] = (tab_nomed['internal']-tab_nomed['external'])/tab_nomed['internal']
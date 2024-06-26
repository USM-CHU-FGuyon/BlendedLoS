from models.config import MultiExperimentConfig
from models.template import LSTMTemplate, TransformerTemplate, TPCTemplate


class Experiments:
    def __init__(self, config_fname, experiments):
        self.experiments = experiments
        self.models = {'lstm', 'transformer', 'tpc'}
        self.configs = {
            model: MultiExperimentConfig(config_fname, model)
            for model in self.models
            }
        
        self.templateconstructors = {
            'tpc': TPCTemplate, 
            'transformer': TransformerTemplate,
            'lstm': LSTMTemplate
            }
    
    def run_experiment(self, model, exp_name, init_only=False):
        config = self.configs[model]
        s = f'{model} - {exp_name}'
        if not config.run_experiment[exp_name]:
            print(f'Skipping {s}')
            return 
        print(s)
        templateconstructor = self.templateconstructors[model]
        run_configs = config[exp_name]
        for run_config in run_configs:
            self.template = templateconstructor(run_config)
            if init_only:
                raise RuntimeError('init_done.')
            self.template.run()
        
    def run_experiments(self, init_only=False):
        for exp_name in self.experiments:
            for model in self.models:
                self.run_experiment(model, exp_name, init_only)


config_fname = 'config.json'

experiments = [
    'model_benchmark',
    'dataset_benchmark',
    'dataset_benchmark_nomed_75',
    'main_experiment',
    'training_size_study',
    ]

e = Experiments(config_fname, experiments)

e.run_experiments(init_only=False)

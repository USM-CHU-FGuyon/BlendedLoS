import torch
import shap

from .experiment_template import ExperimentTemplate
from .initialise_arguments import read_config
from utils.chunkreader import ChunkReader

class ShapExperiment(ExperimentTemplate):
    def __init__(self, rundir):
        self.rundir = rundir
        self.config = read_config(self.rundir+'config.json')

        super().__init__(self.config, logging=False)
        self.model = torch.load(self.rundir+'best_model.pth')

        self.shap_datareader_fit = ChunkReader(f'{self.datareader_path}train',
                                           ts_variables=self.ts_variables,
                                           config=self.config,
                                           device=self.device,
                                           max_batches=12,
                                           batch_size=1)
    
    
        self.flat_batches = torch.cat([b[4] for b in self.shap_datareader_fit.batch_gen()])
        #print(batches)
        #self.explainer = shap.GradientExplainer(self.predict,
        #                                        batches)
    
    def predict_ts(self):
        self.X_ts = self.shap_datareader_fit.ts_input_array()
    def predict_flats(self):
        self.X_flat = self.shap_datareader_fit.flat_input_array()
        #self.model()
    
    
    def explain(self, n_patients):
        self.shap_datareader_exp = ChunkReader(f'{self.datareader_path}test',
                                           ts_variables=self.ts_variables,
                                           config=self.config,
                                           device=self.device,
                                           max_batches=10)
        batches = [b for b in self.shap_datareader_exp.batch_gen()]        

        print(self.explainer.shap_values(batches))
        
        
        
        

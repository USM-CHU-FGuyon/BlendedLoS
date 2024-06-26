import torch
import numpy as np

from models.experiment_template import ExperimentTemplate
from models.mean_median_model import Mean, Median

class MeanMedianTemplate(ExperimentTemplate):
    def __init__(self, config):
        super().__init__(config)
        
    def run(self):
        self.model.fit()
        self.test()
        
    def test(self):
        self.trainvaltest='test'
        
        for batch in self.test_datareader.batch_gen():
            
            patientids, padded, mask, diagnoses, flat, y_los, y_mort, seq_lengths = batch
            
            y_hat_shape = (padded.shape[0], int(max(seq_lengths)))
            
            y_hat_los = torch.ones(y_hat_shape, device=self.device) * self.model.pred_value
            y_hat_mort = torch.zeros(y_hat_shape, device=self.device)
        
            self.df_epoch = self._stack_output(y_los,
                                               y_hat_los,
                                               y_mort,
                                               y_hat_mort,
                                               patientids,
                                               mask)
            if self.df_epoch.index.nunique()>500:
                break
        self.loss_df.loc[self.epoch_idx, 'test'] = np.nan
        
        self._save_test_pred()
        self._report_epoch()
        
        
class MeanTemplate(MeanMedianTemplate):
    def __init__(self, config, **kwargs):
        config['exp_name'] = 'Mean'
        #config = initialise_meanmedian_arguments(config | kwargs)
        super().__init__(config)
        self.model = Mean(config,
                          train_datareader=self.train_datareader,
                          test_datareader=self.test_datareader,)
        
class MedianTemplate(MeanMedianTemplate):
    def __init__(self, config, **kwargs):
        config['exp_name'] = 'Median'
        #config = initialise_meanmedian_arguments(config | kwargs)
        super().__init__(config)
        self.model = Median(config,
                          train_datareader=self.train_datareader,
                          test_datareader=self.test_datareader,)    
    
from pathlib import Path

import pandas as pd

class ConstantModel:
    def __init__(self,
                 config,
                 train_datareader,
                 test_datareader):
        self.pred_fname = 'test_predictions_los.csv'
        self.savepath = Path(f'{config.savedir}/{self.pred_fname}')
        self.train_datareader = train_datareader
        self.test_datareader = test_datareader
        
        self.true_los = self.test_datareader.labels.loc[:,['lengthofstay']]

        self.pred_value = None
        self.predictions=pd.DataFrame(columns=['los_predictions', 'label'])

class Mean(ConstantModel):
    def fit(self):
        self.pred_value = self.train_datareader.labels.lengthofstay.mean()

class Median(ConstantModel):
    def fit(self):
        self.pred_value = self.train_datareader.labels.lengthofstay.median()   
   
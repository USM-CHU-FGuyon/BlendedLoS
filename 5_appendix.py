from pathlib import Path

import numpy as np
import pandas as pd

from database.BlendedICU import BlendedICU


class ErrorExplainer(BlendedICU):
    def __init__(self, dirname='dataset_benchmark_75_2404'):
        super().__init__()
        self.df_med_usage = pd.read_parquet(self.med_usage_pth)
        self.df_labels = pd.read_parquet(self.extracted_labels_pth)
        self.pth_results = 'results/'
        self.pth_resdir = self.pth_results + dirname
        self.fname_results = 'test_predictions_los.csv'
        self.pths_results = self._get_pths_results()
        
        self.pths_patients_scores = self._get_pth_patient_scores()
        self.df_errors = self._load_patient_predictions()
        

    def _get_pths_results(self):
        dic = {p.stem.split('_')[0]: p for p in Path(self.pth_resdir).iterdir() if p.is_dir()}
        return dic

    def _get_pth_patient_scores(self):
        dic = {}
        for dataset, pth in self.pths_results.items():
            files = [*pth.rglob('*'+self.fname_results)]
            if len(files)!=1:
                raise ValueError('Found None or several pred')
            dic[dataset] = files[0]
        return dic
    
    def _load_patient_predictions(self):
        dic = {dataset: pd.read_csv(pth, index_col='patientids')
               for dataset, pth in self.pths_patients_scores.items()}
        df = pd.concat(dic).rename_axis(['train_on', 'patientids'])#.set_index('eval_on', append=True)

        df['mape'] = self._mape(df.label, df.pred_los)
        return df

    @staticmethod
    def _mape(y_true, y_pred):
        epsilon = 4/24
        return 100 * (np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon))

self = ErrorExplainer('dataset_benchmark_75_2404')
df = (self.df_errors
      .assign(eval_on=(self.df_errors.index
                       .get_level_values('patientids')
                       .map(lambda x: x.split('-')[0])))
      .reset_index()
      .merge(self.df_med_usage, left_on='patientids', right_index=True)
      .merge(self.df_labels[['mortality']], left_on='patientids', right_index=True)
      .set_index(['train_on', 'patientids', 'eval_on']))

df.loc[df.mortality==1, ['pred_los', 'label', 'mape']] = np.nan

df.set_index('fentanyl', append=True).groupby(['train_on', 'eval_on', 'fentanyl']).mape.mean()

df.groupby(['train_on', 'patientids', 'eval_on']).mape.mean().groupby(['train_on', 'eval_on']).mean()



dff = df[['mape', 'fentanyl']]

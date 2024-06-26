from tqdm import trange

import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from database.BlendedICU import BlendedICU

class DataReader(BlendedICU):
    def __init__(self, trainvaltest, config, device):
        super().__init__()
        
        print(f'\nInitialize {trainvaltest} datareader.')
        self.incomplete_init = False
        self.trainvaltest = trainvaltest
        self.config = config
        self.device = device
        self.datasets = self.config.datasets
        self._dtype = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor
        
        self.data_pth = f'{self.data_pth}/{self.config.datareader_pth}/{trainvaltest}/'
        
        self.labels = self.load(self.data_pth+'labels.parquet')
        self.flat = self.load(self.data_pth+'flat.parquet', columns=['age',
                                                                     'sex',
                                                                     'height',
                                                                     'weight'])
        self._ts_pths = self._get_ts_pths()
        self.ts_variables = self._get_ts_variables()
        self.loadcols = self._get_loadcols()
        self.included_patients = self._get_included_patients()
        self.batch_size = self._get_batch_size()
        self.percentage_patients = self._get_percentage_patients()
        self.n_ts_variables = len(self.loadcols)
        self.n_flat_features = self.flat.shape[1]
        
        self.patient_sample = self._get_patient_sample()
        self.n_patients = self._get_n_patient()
        self.n_unique_persons = self._get_n_unique_persons()
        
        self.n_batch = self._get_n_batch()
        self.time_before_pred = self.config.time_before_pred
        self.D = 1
        self.F = self.n_ts_variables - 1 # variables minus time


    def read_ts(self, stay_id):
        '''To be used for debugging'''
        try:
            pth = self._ts_pths.loc[stay_id, 'ts_pth']
            print(f'Reading {pth}')        
            df = pd.read_parquet(pth)
        except KeyError:
            print(f'{pth} not found')
            df = pd.DataFrame()
        return df    
        
    def _get_included_patients(self):
        inclusion_msk = self._ts_pths[self.col_truelos]>self.config.min_los
        included_patients = self._ts_pths.loc[inclusion_msk, ['source_dataset']]
        return included_patients
    
    def _get_ts_variables(self):
        df = pd.read_csv('ts_variables.csv', sep=';')
        return df.loc[df.keep.astype(bool)].drop(columns='keep')
        
    def _get_n_unique_persons(self):
        '''returns the number of unique persons to read from.'''
        included_persons = self.labels.loc[self.patient_sample, 'uniquepid']
        n_unique_persons = int(included_persons.nunique())
        return n_unique_persons
        
    def _get_n_patient(self):
        n_patients = len(self.patient_sample)
        print(f'   -> trainval_dic {self.percentage_patients} percent: {n_patients} patients')
        return n_patients
    
    def _get_percentage_patients(self):
        if self.trainvaltest in ['train', 'val']:
            return self.config.percentage_trainval_dic
        return self.config.percentage_test_dic
    
    def _get_batch_size(self):
        return getattr(self.config, f'batch_size_{self.trainvaltest}')
    
    def _get_patient_sample(self):
        """
        Generates the full sample used by the datareader. 
        eg full train/test/val set.
        This ensures that the lengthofstay is larger than min_los.
        """
        
        patient_samples = []
        for dataset, df in self.included_patients.groupby('source_dataset'):
            patients = df.sort_index().index
            n_select = int(patients.nunique()*self.percentage_patients[dataset]/100)
            try:
                np.random.seed(974)
                patient_sample = np.random.choice(patients,
                                                  n_select,
                                                  replace=False)
                patient_samples.append(patient_sample)
            except ValueError:
                print(f'\n/!\ INCOMPLETE initialization: Specified '
                      f'larger than number of included patients='
                      f'{len(patients)} in {dataset}.')
                self.incomplete_init = True
                patient_sample = patients
                
        patient_samples = np.concatenate(patient_samples)
        return patient_samples
    
    def _get_n_batch(self):
        total_n_batch = len(self.patient_sample)//self.batch_size
        n_batch = min(total_n_batch, self.config.max_n_batch)
        if n_batch<1:
            print(f'\n/!\ INCOMPLETE initialization: Batch size {self.batch_size} >'
                  f' n_patient {len(self.patient_sample)}')
            self.incomplete_init = True
        return n_batch
    
    def _gen_patient_batch(self):
        """
        Generates a single patient batch from the patient sample created in 
        _get_patient_sample.
        """
        patient_batch = np.random.choice(self.patient_sample,
                                         self.batch_size,
                                         replace=False)
        return patient_batch
        
    def _get_loadcols(self):
        """Always keep time as the first column."""
        ts_categories = {
            'medication': self.config.use_med,
            'laboratory': self.config.use_lab,
            'vitals': self.config.use_vitals,
            'respiratory': self.config.use_resp
            }

        used_cat = [cat for cat, used in ts_categories.items() if used]
        keep_msk = self.ts_variables.category.isin(used_cat)
        return ['time'] + self.ts_variables.loc[keep_msk, 'variable'].to_list()
        
    def _get_ts_pths(self):
        ts_pths_pth = self.data_pth + self.ts_pths_file
        print(f'Loading {ts_pths_pth}')
        ts_pths = pd.read_csv(ts_pths_pth, index_col=0)
        print(f'   -> found: {len(ts_pths)} patients')
        ts_pths = ts_pths.join(self.labels[self.col_truelos])
        ts_pths = ts_pths.loc[ts_pths.source_dataset.isin(self.datasets)]
        return ts_pths
        
    def pad_sequences(self, ts_batch):
        
        groups = ts_batch.groupby(level='patient', sort=False)
        seq_lengths = self._tensorize(groups.time.count())
        ts_tensors = [self._tensorize(df) for _, df in groups]
        
        padded = (pad_sequence(ts_tensors, batch_first=True, padding_value=0)
                  .permute((0,2,1)))
        
        return padded, seq_lengths
    
    def _tensorize(self, df):
        return torch.tensor(df.values, device=self.device).type(self._dtype)
    
    def _gen_los_labels(self, patient_batch, seq_lengths):
        hmax = int(max(seq_lengths))
        batch_los = self._tensorize(self.labels.loc[patient_batch, self.col_truelos])
        los = (batch_los.unsqueeze(1).repeat(1, hmax)
               - torch.arange(hmax, device=self.device).div(24))
        los = los.clamp(0, self.config.max_los)
        return los[..., self.time_before_pred:]
    
    def _gen_mort_labels(self, patient_batch):
        batch_mort = self.labels.loc[patient_batch, self.col_mort]
        batch_mort_tensor = self._tensorize(batch_mort)
        return batch_mort_tensor
        
    def _gen_flats(self, patient_batch):
        batch_flat = self.flat.loc[patient_batch]
        return self._tensorize(batch_flat)
    
    def _build_msk(self, padded_seq, seq_lengths):
        msk = torch.zeros((len(seq_lengths), int(max(seq_lengths))),
                          device=self.device).bool()
        for i, l in enumerate(seq_lengths):
            msk[i, :int(l)] = 1
        return msk
    
    def _gen_ts_batch(self, patient_batch):
        """
        Time should always be the first column to run in TPC.
        """
        self.patient_ts_path = self._ts_pths.loc[patient_batch, 'ts_pth'].to_list()
        ts = self.load(self.patient_ts_path,
                       columns=self.loadcols,
                       verbose=False)
        ts['time'] = ts['time'].div(24)
        padded_seq, seq_lengths = self.pad_sequences(ts)
        msk = self._build_msk(padded_seq, seq_lengths)
        return padded_seq, seq_lengths, msk[..., self.time_before_pred:]
    
    def _repeat_patient_batch(self, patient_batch, msk):
        N_reps = msk.sum(1).cpu().numpy()
        repated_patientids = np.repeat(patient_batch, N_reps)
        return repated_patientids
    
    def _check_init(self):
        '''Checks that datareader initialization was complete before 
        generating the batch_gen'''
        if self.incomplete_init:
            raise ValueError(f'{self.trainvaltest} datareader Initialization '
                             'was incomplete.')
    
    def batch_gen(self):
        self._check_init()
        for i in trange(self.n_batch):
            patient_batch = self._gen_patient_batch()
            
            flat_batch = self._gen_flats(patient_batch)
            ts_batch, seq_lengths, msk = self._gen_ts_batch(patient_batch)
            los_labels = self._gen_los_labels(patient_batch, seq_lengths)
            
            repated_patientids = self._repeat_patient_batch(patient_batch, msk)
            mort_labels = self._gen_mort_labels(patient_batch)
            repeated_mort_labels = self._gen_mort_labels(repated_patientids)
            yield (patient_batch,
                   repated_patientids,
                   los_labels,
                   mort_labels,
                   repeated_mort_labels,
                   flat_batch,
                   ts_batch,
                   msk,
                   seq_lengths)
    
    
if __name__=='__main__':
    import os
    os.chdir('..')
    from models.config import MultiExperimentConfig
    
    c = MultiExperimentConfig('config.json', model="tpc")
    
    config = c['main_experiment_75_']
    device = torch.device('cpu')
    
    run_config = config[2]
    
    run_config.batch_size_train = 100
    
    self = DataReader('train', run_config, device)
    
    (patient_batch,
     repeated_patientids,
     true_los,
     true_mort,
     repeated_true_mort,
     flat, 
     ts_padded,
     msk,
     seq_lengths) = next(self.batch_gen())
    
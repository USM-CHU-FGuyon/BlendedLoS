from pathlib import Path
from functools import reduce
import pandas as pd

from database.BlendedICU import BlendedICU


class DataSplitter(BlendedICU):
    """
    n_patients_test: number of patients from each input dataset in the test set.
    This class divides icu stays into train, val, and test. 
    It ensures that the training and evaluation are done on the same icu stays.
    It only includes icu stays where timeseries files exist and lengthofstay is 
    larger than min_los.
    """
    def __init__(self,
                 seed=974,
                 split=0.15,
                 equal_samples=False,
                 recompute_index=False):
        '''recompute_index is only necessary if the database was moved after
        being generated. It recomputes the index file inside each chunk.'''
        super().__init__()
        self.equal_samples = equal_samples
        self.seed = seed
        self.split = split
        
        if recompute_index:
            self.build_full_index()
        
        self.labels_pth = f'{self.data_pth}/preprocessed_labels.parquet'
        self.flat_pth = f'{self.data_pth}/preprocessed_flat.parquet'
        
        self.labels = self.load(self.labels_pth)
        self.flat = self.load(self.flat_pth)
        
        self.ts_pths = self.build_index()

        self.grouped_patients, self.n_sample = self._get_unique_patient_per_dataset()
        self._save_extracted_sample()
        self.datasets = set(self.grouped_patients.groups.keys())
        
        self.n_testval = int(self.n_sample*self.split)
        self.n_train = self.n_sample - 2*self.n_testval

        self.train, self.val, self.test = self._trainvaltest()

    
    def _save_extracted_sample(self):
        
        uniquepids = pd.concat([g for _, g in self.grouped_patients]).index
        
        labels = self.labels.loc[self.labels.uniquepid.isin(uniquepids)]
        print(f'Saving {self.extracted_labels_pth}')
        labels.to_parquet(self.extracted_labels_pth)

    def build_index(self):
        index_pth = [p for p in Path(self.ts_pth).iterdir() if p.is_dir()]
        ts_pths_s = [self.read_index(p) for p in index_pth]
        ts_pths = pd.concat(ts_pths_s)
        ts_pths = ts_pths.join(self.labels[['uniquepid', 'source_dataset']])
        ts_pths.to_csv(self.ts_pth+'index.csv', sep=';')
        if ts_pths.empty:
            raise ValueError(f'Timeseries not found at {self.ts_pth}')
        return ts_pths
        
    def _split_indices(self, groupname, n, dropidx=[]):
        try:
            return (self.grouped_patients
                        .get_group(groupname)
                        .drop(dropidx)
                        .sample(n=n, random_state=self.seed)
                        .index)
        except ValueError:
            df = self.grouped_patients.get_group(groupname).drop(dropidx)
            raise ValueError(f'Cannot take sample of size {n} with dataset group'
                             f'{groupname} of size {int(df.count())} size')
    
    def _trainvaltest_uniquepid(self):
        test_uniquepids = {g: self._split_indices(g, n=self.n_testval)
                           for g in self.datasets}
        
        val_uniquepids = {g: self._split_indices(g,
                                                 dropidx=test_uniquepids[g],
                                                 n=self.n_testval)
                          for g in self.datasets}
        
        train_dropidx = {g: pd.Index.union(test_uniquepids[g],
                                           val_uniquepids[g])
                         for g in self.datasets}
        
        train_uniquepids = {g: self._split_indices(g,
                                                   dropidx=train_dropidx[g],
                                                   n=self.n_train)
                            for g in self.datasets}
        
        return train_uniquepids, val_uniquepids, test_uniquepids
    
    def _loc_patients(self, uniquepids):
        return self.ts_pths.loc[self.ts_pths.uniquepid.isin(uniquepids)].index
    
    def _trainvaltest(self):
        train_pid, val_pid, test_pid = self._trainvaltest_uniquepid()
        
        train = {g: self._loc_patients(pids) for g, pids in train_pid.items()}
        val = {g: self._loc_patients(pids) for g, pids in val_pid.items()}
        test = {g: self._loc_patients(pids) for g, pids in test_pid.items()}
        return train, val, test
        
    def _get_unique_patient_per_dataset(self):
        groups = (self.ts_pths.reset_index()
                        .groupby('uniquepid').first()
                        .groupby('source_dataset'))
        n_sample = None
        if self.equal_samples:
            n_sample = groups.patient.count().min()
            groups = (groups.sample(n=n_sample, random_state=self.seed)
                            .groupby('source_dataset'))
        return groups[['patient']], n_sample

    def _create_folder(self, train_on, folder):
        datareader_pth = self.datareader_path(train_on)
        folder_pth = f'{self.data_pth}{datareader_pth}{folder}'
        Path(folder_pth).mkdir(exist_ok=True, parents=True)
        return folder_pth
     
    def _split_timeseries_pths(self, stays, target_pth):
        savepath = f'{target_pth}/timeseries_pths.csv'
        df = self.ts_pths.loc[stays]
        df.to_csv(savepath)
        print(f'   saving {savepath}')
     
    def run(self, train_on):
        test_on = self.datasets
        
        tables = {'labels': self.labels,
                  'flat': self.flat}
        
        partitionned_stays = {
            'train': reduce(pd.Index.union, (self.train[d] for d in train_on)),
            'val': reduce(pd.Index.union, (self.val[d] for d in train_on)),
            'test': reduce(pd.Index.union, (self.test[d] for d in test_on))
        }
        
        for trainvaltest, stays in partitionned_stays.items():
            folder_pth = self._create_folder(train_on, trainvaltest)
            self._split_timeseries_pths(stays, folder_pth)
            
            for table_name, table in tables.items():
                partitioned_table = table.loc[stays]
                savepath = f'{folder_pth}/{table_name}.parquet'
                self.save(partitioned_table, savepath)
    
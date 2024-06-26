"""
Class for loading the blendedICU dataset.

"""
import json
from pathlib import Path

import pandas as pd
import numpy as np

class BlendedICU:
    def __init__(self):
        paths = json.load(open('paths.json', 'r'))
        self.ts_variables = self._get_ts_variables()
        self.kept_meds = self._kept_meds()
        self.data_pth = paths['blendedICU']
        self.formatted_ts_pth = self.data_pth+'formatted_timeseries/'
        self.ts_pths_file = '/timeseries_pths.csv'
        self.ts_pth = self.data_pth + 'preprocessed_timeseries/'
        self.col_los = 'lengthofstay'
        self.col_truelos = 'true_lengthofstay'
        self.col_mort = 'mortality'
        self.extraction_dirname = 'extraction'
        self.extraction_pth = f'{self.data_pth}/{self.extraction_dirname}/'
        self.extracted_labels_pth = f'{self.extraction_pth}extracted_labels.parquet'
        self.med_usage_pth = f'{self.extraction_pth}med_usage.parquet'
        Path(self.extracted_labels_pth).parent.mkdir(exist_ok=True)
    
    def med_usage_table(self):
        ts_pths = self.extracted_labels.ts_pth.to_list()
        ts_pths_chunks = np.array_split(ts_pths, 100)
        med_usage = pd.DataFrame(columns=self.kept_meds)
        temp_pth = f'{self.extraction_pth}med_usage_temp.parquet'
        for i, pth_chunk in enumerate(ts_pths_chunks):
            print(f'chunk {i}/100')
            df = pd.read_parquet(pth_chunk.tolist(), columns=self.kept_meds)
            
            chunk_patient_usage = df.groupby(level='patient').any().astype(int)
            med_usage = pd.concat([chunk_patient_usage, med_usage])
            
            med_usage.to_parquet(temp_pth)
        med_usage.to_parquet(self.med_usage_pth)
        Path(temp_pth).unlink()
        
    def _kept_meds(self):
        return self.ts_variables.loc[self.ts_variables.category=='medication', 'variable'].to_list()
    
    def _get_ts_variables(self):
        ts_var = pd.read_csv('ts_variables.csv', sep=';')
        ts_var = (ts_var
                  .loc[ts_var.keep.astype(bool)]
                  .drop(columns='keep')
                  .reset_index(drop=True))
        return ts_var
    
    def datareader_path(self, train_on):
        train_txt = "_".join(sorted(train_on))
        return f'{self.extraction_dirname}/train_on_{train_txt}/'
    
    def _build_index(self, ts_dir):
        """
        Lists the files in a timeseries processing chunk and saves an index 
        file with the list of paths to files of this folder.
        """
        index_pth = f'{ts_dir}/index.csv'
        dic = {p.stem: p.resolve() for p in Path(ts_dir).glob('*parquet')}
        index_df = (pd.DataFrame.from_dict(dic,
                                           orient='index',
                                           columns=['ts_pth'])
                    .rename_axis('patient'))
        print(f'Saving index file {index_pth}')
        index_df.to_csv(index_pth, sep=';')
        return index_df
    
    def build_full_index(self):
        print('Building full index...')
        index_pth = self.ts_pth+'index.csv'
        index_dfs = [self._build_index(p) for p in Path(self.ts_pth).iterdir() if p.is_dir()]
        index_df = pd.concat(index_dfs)
        print(f'Saving index file {index_pth}')
        index_df.to_csv(index_pth, sep=';')
        return index_df
    
    def load(self, pth, verbose=True, **kwargs):
        if verbose:
            print(f'Loading {pth}')
        return pd.read_parquet(pth, **kwargs)
    
    def save(self, df, savepath, pyarrow_schema=None):
        """
        convenience function: save safely a file to parquet by creating the 
        parent directory if it does not exist.
        """
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        print(f'   saving {savepath}')
        df.to_parquet(savepath, schema=pyarrow_schema)
        return df
    
    def read_index(self, ts_dir):
        index_pth = f'{ts_dir}/index.csv'
        try: 
            index_df = pd.read_csv(index_pth, sep=';', index_col='patient')
        except FileNotFoundError:
            print(f'index file {index_pth} not found')
            index_df = pd.DataFrame(columns=['ts_pths']).rename_axis('patient')
        return index_df
    
if __name__=='__main__':
    import os
    os.chdir('..')
    self = BlendedICU()    

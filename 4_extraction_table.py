import random
from pathlib import Path

import polars as pl
import pandas as pd

from utils.datasplitter import DataSplitter


class ExtractionTable(DataSplitter):
    """
    This class Produces Table 1. from the article.
    get_flat_table describes the demographics
    get_med_table describes the drug exposure
    """
    def __init__(self):
        super().__init__(equal_samples=True, recompute_index=False)
        random.seed(974)
        self.drug_exposure_savepath = 'figures/main_table/drug_exposures.parquet'
        self.df = pl.scan_parquet(self.extracted_labels_pth)
        
        self.extracted_ts_pths = self._get_extraction_ts_pths()
        
    def get_flat_table(self):
        tab = (self.df
               .group_by('source_dataset')
               .agg(
                   pl.col('original_uniquepid').n_unique().alias('patient'),
                   pl.col('patient').n_unique().alias('stays'),
                   pl.col('raw_age').mean().alias('age'),
                   pl.col('raw_age').std().alias('age_std'),
                   pl.col('lengthofstay').median().mul(24).alias('LoS'),
                   pl.col('lengthofstay').mean().alias('mean_LoS'),
                   pl.col('lengthofstay').mul(24).std().alias('LoS_std'),
                   pl.col('lengthofstay').ge(10).sum().alias('n_long_los'),
                   pl.col('lengthofstay').ge(10).mean().mul(100).alias('p_long_los'),
                   pl.col('mortality').mean().mul(100).alias('p_mortality'),
                   pl.col('mortality').ge(1).sum().alias('n_mortality'),
                )
               .filter(pl.col('source_dataset').ne('mimic3'))
               .collect()
               .to_pandas())
        return tab

    def _get_extraction_ts_pths(self):
        extracted_patients = (self.df
                              .filter(pl.col('source_dataset').ne('mimic3'))
                              .select('patient')
                              .unique()
                              .collect()
                              .to_pandas()
                              .patient)
        extracted_ts_pths = self.ts_pths.loc[extracted_patients, 'ts_pth'].to_list()
        return extracted_ts_pths
        
    def get_med_table(self, n_patients=None):
        '''
        n_patients: Number of patients to read med exposure from. by default None
        reads all patients (takes several minutes).
        '''
        pths = (self.extracted_ts_pths 
                if n_patients is None 
                else pd.Series(self.extracted_ts_pths).sample(n=n_patients).to_list())
        
        df = pl.scan_parquet(pths)
        source_datasets = self.df.select('source_dataset', 'patient')


        patient_counts = (df.select('patient')
                          .unique()
                          .join(source_datasets, on='patient')
                          .group_by('source_dataset')
                          .n_unique()
                          .rename({'patient': 'patient_count'}))

        tab = (df.select(*self.kept_meds, 'patient')
               .melt(id_vars='patient')
               .filter(pl.col('value')>0)
               .select('patient', 'variable')
               .join(source_datasets, on='patient')
               .unique()
               .group_by('variable', 'source_dataset')
               .n_unique()
               .rename({'variable': 'drug',
                        'patient': 'drug_count'})
               .join(patient_counts, on='source_dataset')
               .with_columns(
                   drug_percentage=pl.col('drug_count').truediv(pl.col('patient_count'))
                   )
               .select('source_dataset', 'drug', 'drug_percentage', 'drug_count')
               .collect()
               .to_pandas())
        Path(self.drug_exposure_savepath).parent.mkdir(exist_ok=True)
        tab.to_parquet(self.drug_exposure_savepath)
        return tab
        
    @staticmethod
    def make_table(df_flat, df_meds):
        flats = (df_flat
                 .round(1)
                 .astype(str)
                 .assign(age_with_std=lambda x: x.age + ' ['+ x.age_std+']',
                         los_with_std=lambda x: x.LoS+ ' ['+ x.LoS_std+']',
                         long_los=lambda x: x.n_long_los + ' ('+ x.p_long_los+'\%)',
                         mortality=lambda x: x.n_mortality + ' ('+ x.p_mortality+'\%)'
                         )
                 .drop(columns=['age',
                                'age_std',
                                'LoS',
                                'LoS_std',
                                'p_long_los',
                                'n_long_los',
                                'n_mortality',
                                'p_mortality',
                                'mean_LoS'
                                ])
                 .set_index('source_dataset').T
                 .rename({
                     'age_with_std': r'Age (years)\textsuperscript{*}',
                     'los_with_std': r'LoS (hours)\textsuperscript{*}',
                     'stays': r'ICU Stays (number)\textsuperscript{1}',
                     'LoS': r'LoS (hours)\textsuperscript{*}',
                     'long_los': r'Long LoS\textsuperscript{2}',
                     'mortality': r'Mortality'
                     }))
        
        meds = (df_meds
                 .set_index(['source_dataset', 'drug'])
                 .unstack('source_dataset')
                 .fillna(0))
        
        med_number_table = meds['drug_count'].astype(int).astype(str)
        med_percent_table = ' ('+meds['drug_percentage'].mul(100).round(1).astype(str)+'\%)'
        
        formatted_meds = med_number_table + med_percent_table
        
        formatted_meds.index = formatted_meds.index.str.capitalize()
        
        tab = (pd.concat([flats, formatted_meds])
               .rename_axis(None, axis=1)
               .rename(columns={'amsterdam': r'\amsterdam',
                                'hirid': '\hirid',
                                'eicu': '\eicu',
                                'mimic3': '\mimicthree',
                                'mimic4': '\mimicfour',
                                }))
        
        first_med = df_meds.drug.sort_values().values[0].capitalize()

        latext = (tab.to_latex(float_format="{:.1f}".format,
                               column_format='lccccc')
                  .replace('patient', r'\textbf{Demographics}\\'+'\nPatients (number)')
                  .replace(first_med, '\n'+r'\midrule \textbf{Drug Exposure\textsuperscript{3}}\\'+f'\n{first_med}'))
        
        print('\n',latext)
    
    
if __name__=="__main__":
    self = ExtractionTable()
    
    df_flat = self.get_flat_table()
    df_meds = self.get_med_table()
    
    self.make_table(df_flat, df_meds)

    
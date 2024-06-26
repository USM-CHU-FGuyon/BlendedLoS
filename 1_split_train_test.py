"""
Prepare the train, val, and test data from the BlendedICU dataset.
"""
from utils.datasplitter import DataSplitter


ds = DataSplitter(equal_samples=True)

for train_on in (['amsterdam'],
                 ['mimic4'],
                 ['hirid'],
                 ['eicu'],
                 ['hirid', 'mimic4'],
                 ['amsterdam', 'mimic4'],
                 ['amsterdam', 'eicu'],
                 ['hirid', 'mimic4', 'eicu', 'amsterdam'],
                 ):

    ds.run(train_on=train_on)

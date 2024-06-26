from sklearn import metrics
import numpy as np

class Metrics:
    def __init__(self,
                 key_true='y_true',
                 key_pred='y_pred',
                 ignore_los_dc=False):
        self.key_true = key_true
        self.key_pred = key_pred

    def mad(self, df):
        return (df[self.key_true] - df[self.key_pred]).abs().mean()
    
    def mse(self, df):
        return ((df[self.key_true] - df[self.key_pred])**2).mean()
    
    def mape(self, df):
        return ((df[self.key_true] - df[self.key_pred]).abs() / df[self.key_true].clip(lower=4/24)).mean()*100
    
    def msle(self, df):
        return ((df[self.key_true]/df[self.key_pred]).apply(np.log)**2).mean()
    
    def r2(self, df):
        return metrics.r2_score(df[self.key_true], df[self.key_pred])
    
    def auc(self, df):
        if df[self.key_true].nunique()==1:
            return np.nan
        return metrics.roc_auc_score(df[self.key_true], df[self.key_pred])
    
    def bce(self, df):
        return metrics.log_loss(df[self.key_true],
                                df[self.key_pred],
                                labels=[0,1])

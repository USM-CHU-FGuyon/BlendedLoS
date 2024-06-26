import torch
import torch.nn as nn

from models.loss import Loss


def init_lstm_weights(model):
    # same initialisation as in keras
    for module in model.modules():
        if isinstance(module, nn.LSTM):
            nn.init.xavier_uniform_(module.weight_ih_l0)
            nn.init.orthogonal_(module.weight_hh_l0)
            for names in module._all_weights:
                for name in filter(lambda n: 'bias' in n, names):
                    bias = getattr(module, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)

class Model(nn.Module):
    def __init__(self, config, device, F, D, n_flat_features):        
        super().__init__()
        self.n_train = 0
        self.task = config.task
        self.n_layers = config.n_layers
        self.main_dropout_rate = config.main_dropout_rate
        self.batchnorm = config.batchnorm
        self.last_linear_size = config.last_linear_size
        self.F = F
        self.D = D
        self.n_flat_features = n_flat_features
        self.no_exp = config.no_exp
        self.alpha = config.alpha
        self.sum_losses = config.sum_losses
        self.device = device
        self.loss_type = config.loss
        self.bool_type = torch.cuda.BoolTensor if device == torch.device('cuda') else torch.BoolTensor
        self.ignore_los_dc = config.ignore_los_dc
        self.momentum = 0.1
        self.loss = Loss(config.los_task,
                         config.mort_task,
                         self.loss_type,
                         self.sum_losses,
                         self.alpha,
                         self.ignore_los_dc)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        max_prediction_time = 14
        min_prediction_time = 1/48
        self.hardtanh = nn.Hardtanh(min_prediction_time, max_prediction_time) 
        self.main_dropout = nn.Dropout(p=self.main_dropout_rate)
        self.EmptyModule = EmptyModule

        self.bn_point_last_los = self.init_batchnorm(self.last_linear_size,
                                                     self.momentum)
        self.bn_point_last_mort = self.init_batchnorm(self.last_linear_size,
                                                      self.momentum)

        self.point_final_los = nn.Linear(self.last_linear_size, 1)
        self.point_final_mort = nn.Linear(self.last_linear_size, 1)

    def remove_none(self, X):
        return tuple(x for x in X if x is not None)
    
    def init_batchnorm(self, num_features, momentum=None):
        if self.batchnorm:
            return nn.BatchNorm1d(num_features=num_features, momentum=momentum)
        return self.EmptyModule()
           

class EmptyModule(nn.Module):
    def __init__(self, *args):
        super().__init__()
        
    def forward(self, X):
        return X      
        
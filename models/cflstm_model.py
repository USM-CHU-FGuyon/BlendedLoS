import torch.nn as nn
from torch import exp

from models.model import Model, init_lstm_weights

class cfLSTM(Model):
    def __init__(self, config, device, F, D, n_flat_features):
        # The timeseries data will be of dimensions B * (2F + 2) * T where:
        #   B is the batch size
        #   F is the number of features for convolution (N.B. we start with 2F because there are corresponding mask features)
        #   T is the number of timepoints
        #   The other 2 features represent the sequence number and the hour in the day

        # The diagnoses data will be of dimensions B * D where:
        #   D is the number of diagnoses
        # The flat data will be of dimensions B * no_flat_features
        super().__init__(config, device, F, D, n_flat_features)

        self.hidden_size = config.hidden_size
        self.lstm_dropout_rate = config.lstm_dropout_rate

        init_lstm_weights(self)

        self.lstm_dropout = nn.Dropout(p=self.lstm_dropout_rate)
        
        self.point_h0 = nn.Linear(in_features=self.n_flat_features,
                                  out_features=self.hidden_size*self.n_layers)
        self.point_c0 = nn.Linear(in_features=self.n_flat_features,
                                  out_features=self.hidden_size*self.n_layers)

        self.lstm = nn.LSTM(input_size=self.F + 1,
                            hidden_size=self.hidden_size,
                            num_layers=self.n_layers,
                            bidirectional=False,
                            dropout=self.lstm_dropout_rate,
                            batch_first=True)

        self.point_los = nn.Linear(in_features=self.hidden_size,
                                   out_features=self.last_linear_size)
        self.point_mort = nn.Linear(in_features=self.hidden_size,
                                    out_features=self.last_linear_size)

        # input shape: (B * T) * last_linear_size
        self.bn_point_last_los = self.init_batchnorm(num_features=self.last_linear_size)
        self.bn_point_last_mort = self.init_batchnorm(num_features=self.last_linear_size)

        # input shape: (B * T) * last_linear_size
        # output shape: (B * T) * 1
        self.point_final_los = nn.Linear(in_features=self.last_linear_size,
                                         out_features=1)
        self.point_final_mort = nn.Linear(in_features=self.last_linear_size,
                                          out_features=1)


    def forward(self, X, flat, time_before_pred=5):
        # flat is B * no_flat_features
        # diagnoses is B * D
        # X is B * (2F + 2) * T
        # X_mask is B * T
        # (the batch is padded to the longest sequence)
        B, _, T = X.shape

        # note that we cut off at time_before_pred hours here because the model is only valid from time_before_pred hours onwards
            
        h_0 = (self.point_h0(flat)
               .view((B, self.hidden_size, self.n_layers))
               .permute(2,0,1).contiguous())
        c_0 = (self.point_c0(flat)
               .view((B, self.hidden_size, self.n_layers))
               .permute(2,0,1).contiguous())

        lstm_output, (h_n, c_n) = self.lstm(X.permute(0, 2, 1),
                                            (h_0, c_0))  # T * B * hidden_size
        
        last_point_los = self.relu(
                         self.main_dropout(
                         self.bn_point_last_los(
                         self.point_los(
                         lstm_output).permute(1,2,0)
                         ))).permute(0,2,1)

        last_point_mort = self.relu(
                          self.main_dropout(
                          self.bn_point_last_mort(
                          self.point_mort(
                          lstm_output).permute(1,2,0)
                          ))).permute(0,2,1)
        
        self.last_point_mort = last_point_mort
        
        self.pred = self.point_final_los(last_point_los)
        
        los_predictions = (self.point_final_los(last_point_los)
                           .squeeze()
                           .permute(1,0)[:, time_before_pred:])

        if not self.no_exp:
            los_predictions = exp(los_predictions)
        
        los_predictions = self.hardtanh(los_predictions)
        mort_predictions = (self.sigmoid(
                           self.point_final_mort(last_point_mort)
                           ).squeeze()
                            .permute(1,0)[:, time_before_pred:])

        return los_predictions, mort_predictions
    
import torch.nn as nn
import torch
from models.model import Model
from torch import exp, cat

#from models.loss import Loss

class BaseLSTM(Model):
    def __init__(self, config, device, F, D, n_flat_features):
        # The timeseries data will be of dimensions B * (2F + 2) * T where:
        #   B is the batch size
        #   F is the number of features for convolution (N.B. we start with 2F because there are corresponding mask features)
        #   T is the number of timepoints
        #   The other 2 features represent the sequence number and the hour in the day

        # The diagnoses data will be of dimensions B * D where:
        #   D is the number of diagnoses
        # The flat data will be of dimensions B * n_flat_features
        super().__init__(config, device, F, D, n_flat_features)
        self.hidden_size = config.hidden_size
        self.channelwise = config.channelwise
        self.bidirectional = config.bidirectional
        self.lstm_dropout_rate = config.lstm_dropout_rate
        self.main_dropout_rate = config.main_dropout_rate

        self.n_units = self.hidden_size // 2 if self.bidirectional else self.hidden_size
        self.n_dir = 2 if self.bidirectional else 1

        # use the same initialisation as in keras
        for m in self.modules():
            self.init_weights(m)

        self.lstm_dropout = nn.Dropout(p=self.lstm_dropout_rate)

        lstm_kwargs = {
                  'hidden_size':self.n_units,
                  'num_layers': self.n_layers,
                  'bidirectional': self.bidirectional,
                  'dropout': self.lstm_dropout_rate}

        if self.channelwise:
            self.channelwise_lstm_list = nn.ModuleList([nn.LSTM(input_size=2,
                                                                **lstm_kwargs)
                                                        for _ in range(self.F)])
        else :
            # note if it's bidirectional, then we can't assume there's no influence from future timepoints on past ones
            self.lstm = nn.LSTM(input_size=self.F+1, **lstm_kwargs)

        # input shape: B * D
        # output shape: B * diagnosis_size

        # input shape: (B * T) * (n_units + diagnosis_size + n_flat_features)
        # output shape: (B * T) * last_linear_size
        n_channels = self.F if self.channelwise else 1
        input_size = self.n_units * n_channels + self.n_flat_features
        
        self.point_los = nn.Linear(in_features=input_size, out_features=self.last_linear_size)
        self.point_mort = nn.Linear(in_features=input_size, out_features=self.last_linear_size)

        # input shape: (B * T) * last_linear_size
        self.bn_point_last_los = self.init_batchnorm(num_features=self.last_linear_size)
        self.bn_point_last_mort = self.init_batchnorm(num_features=self.last_linear_size)

        # input shape: (B * T) * last_linear_size
        # output shape: (B * T) * 1
        self.point_final_los = nn.Linear(in_features=self.last_linear_size, out_features=1)
        self.point_final_mort = nn.Linear(in_features=self.last_linear_size, out_features=1)


    def init_weights(self, m):
        if isinstance(m, nn.LSTM):
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            for names in m._all_weights:
                for name in filter(lambda n: 'bias' in n, names):
                    bias = getattr(m, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)


    def init_hidden(self, B, device):
        h0 = torch.zeros(self.n_layers*self.n_dir, B, self.n_units).to(device)
        c0 = torch.zeros(self.n_layers*self.n_dir, B, self.n_units).to(device)
        return (h0, c0)

    def forward(self, X, flat, time_before_pred=5):

        # flat is B * n_flat_features
        # diagnoses is B * D
        # X is B * (2F + 2) * T
        # X_mask is B * T
        # (the batch is padded to the longest sequence)

        B, _, T = X.shape

        if self.channelwise:
            # take time and hour fields as they are not useful when processed on their own (they go up linearly. They were also taken out for temporal convolution so the comparison is fair)
            X_separated = torch.split(X[:, 1:, :], self.F, dim=1)  # tuple ((B * F * T), (B * F * T))
            X_rearranged = torch.stack(X_separated, dim=2)  # B * F * 2 * T
            lstm_output = None
            for i in range(self.F):
                X_lstm, hidden = self.channelwise_lstm_list[i](X_rearranged[:, i].permute(2, 0, 1))
                lstm_output = cat(self.remove_none((lstm_output, X_lstm)), dim=2)

        else:
            # the lstm expects (seq_len, batch, input_size)
            # N.B. the default hidden state is zeros so we don't need to specify it
            lstm_output, hidden = self.lstm(X.permute(2, 0, 1))  # T * B * hidden_size

        X_final = self.relu(self.lstm_dropout(lstm_output.permute(1, 2, 0)))
        # (B * (T - time_before_pred)) * n_flat_features
        flats = flat.repeat_interleave(T - time_before_pred, dim=0)

        timeseries = (X_final[..., time_before_pred:]
                      .permute(0, 2, 1)
                      .contiguous()
                      .view(B * (T - time_before_pred), -1))
        # note that we cut off at time_before_pred hours here because the model is only valid from time_before_pred hours onwards

        combined_features = cat((flats, timeseries), dim=1)

        last_point_los = self.relu(
                         self.main_dropout(
                         self.bn_point_last_los(
                         self.point_los(
                         combined_features))))
        
        last_point_mort = self.relu(
                          self.main_dropout(
                          self.bn_point_last_mort(
                          self.point_mort(
                          combined_features))))

        pt_final_los = self.point_final_los(last_point_los
                                            ).view(B, T - time_before_pred)

        if not self.no_exp:
            pt_final_los = exp(pt_final_los)
       
        los_predictions = self.hardtanh(pt_final_los)
        mort_predictions = self.sigmoid(
                           self.point_final_mort(
                           last_point_mort
                           ).view(B, T - time_before_pred))  # B * (T - time_before_pred)

        return los_predictions, mort_predictions

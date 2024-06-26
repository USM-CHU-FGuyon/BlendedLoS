import math
import torch
import torch.nn as nn
from torch import exp, cat

from models.model import Model

# PositionalEncoding adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html. I made the following
# changes:
    # Took out the dropout
    # Changed the dimensions/shape of pe
# I am using the positional encodings suggested by Vaswani et al. as the Attend and Diagnose authors do not specify in
# detail how they do their positional encodings.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=14*24):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)  # changed from max_len * d_model to 1 * d_model * max_len
        self.register_buffer('pe', pe)

    def forward(self, X):
        # X is B * d_model * T
        # self.pe[:, :, :X.size(2)] is 1 * d_model * T but is broadcast to B when added
        X = X + self.pe[:, :, :X.size(2)]  # B * d_model * T
        return X  # B * d_model * T


class TransformerEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 d_model,
                 num_layers,
                 num_heads,
                 feedforward_size,
                 dropout,
                 pe,
                 device):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.pe = pe  # boolean variable indicating whether or not the positional encoding should be applied
        self.input_embedding = nn.Conv1d(input_size, d_model, kernel_size=1)  # B * C * T
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.tf_encoder_layer = nn.TransformerEncoderLayer(
                                        d_model=d_model,
                                        nhead=num_heads,
                                        dim_feedforward=feedforward_size,
                                        dropout=dropout,
                                        activation='relu')
        
        self.transformer_encoder = nn.TransformerEncoder(
                                        encoder_layer=self.tf_encoder_layer,
                                        num_layers=num_layers)

    def _causal_mask(self, size=None):
        mask = (torch.triu(torch.ones(size, size).to(self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask  # T * T

    def forward(self, X, T):
        # X is B * (2F + 2) * T

        # multiplication by root(d_model) as described in Vaswani et al. 2017 section 3.4
        X = self.input_embedding(X) * math.sqrt(self.d_model)  # B * d_model * T
        if self.pe:  # apply the positional encoding
            X = self.pos_encoder(X)  # B * d_model * T
        X = self.transformer_encoder(src=X.permute(2, 0, 1), mask=self._causal_mask(size=T))  # T * B * d_model
        return X.permute(1, 2, 0)  # B * d_model * T


class Transformer(Model):
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
        self.d_model = config.d_model 
        self.tf_dropout_rate = config.tf_dropout_rate
        self.tf_dropout = nn.Dropout(p=self.tf_dropout_rate)
        
        self.transformer = TransformerEncoder(
                                input_size=self.F+1,
                                d_model=self.d_model,
                                num_layers=self.n_layers,
                                num_heads=config.n_heads,
                                feedforward_size=config.feedforward_size,
                                dropout=self.tf_dropout_rate,
                                pe=config.positional_encoding,
                                device=device)

        # input shape: (B * T) * (d_model + diagnosis_size + no_flat_features)
        # output shape: (B * T) * last_linear_size
        input_size = self.d_model + self.n_flat_features

        self.point_los = nn.Linear(input_size, self.last_linear_size)
        self.point_mort = nn.Linear(input_size, self.last_linear_size)


    def forward(self, X, flat, time_before_pred=5):
        # flat is B * no_flat_features
        # diagnoses is B * D
        # X is B * (2F + 2) * T
        # X_mask is B * T
        # (the batch is padded to the longest sequence)
        B, _, T = X.shape  # B * (2F + 2) * T

        T_pred = T - time_before_pred

        # B * d_model * T
        X_final = self.relu(
                  self.tf_dropout(
                  self.transformer(
                  X, T)))  

        flats = flat.repeat_interleave(T_pred, dim=0) # (B * (T - time_before_pred)) * no_flat_features

        ts = X_final[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * T_pred, -1)
    
        combined_features = cat((flats, ts), dim=1)

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
        
        los_predictions = self.point_final_los(last_point_los).view(B, T_pred) # B * (T - time_before_pred)

        if not self.no_exp:
            los_predictions = exp(los_predictions)
        
        los_predictions = self.hardtanh(los_predictions)
        mort_predictions = self.sigmoid(self.point_final_mort(last_point_mort).view(B, T_pred))  # B * (T - time_before_pred)

        return los_predictions, mort_predictions

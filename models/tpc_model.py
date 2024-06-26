import torch.nn as nn
from torch import cat, exp
from torch.nn.functional import pad

from models.model import Model


class TempPointConv(Model):
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
        self.Y = 0
        self.Z = 0
        self.Zt = 0

        self.share_weights = config.share_weights
        self.temp_dropout_rate = config.temp_dropout_rate
        self.kernel_size = config.kernel_size
        self.temp_kernels = config.temp_kernels
        self.point_sizes = config.point_sizes
        
        self.temp_dropout = nn.Dropout(p=self.temp_dropout_rate)
        self.init_tpc()


    def init_tpc(self):
        # non-module layer attributes
        self.layers = []
        for i in range(self.n_layers):
            dilation = i * (self.kernel_size - 1) if i > 0 else 1  # dilation = 1 for the first layer, after that it captures all the information gathered by previous layers
            temp_k = self.temp_kernels[i]
            point_size = self.point_sizes[i]
            self.update_layer_info(layer=i,
                                   temp_k=temp_k,
                                   point_size=point_size,
                                   dilation=dilation,
                                   stride=1)

        self.layer_modules = self.create_temp_pointwise_layers()

        # input shape: (B * T) * ((F + Zt) * (1 + Y) + diagnosis_size + n_flat_features)
        # output shape: (B * T) * last_linear_size

        input_size = (self.F + self.Zt) * (1 + self.Y) + self.n_flat_features

        self.point_last_los = nn.Linear(input_size, self.last_linear_size)
        self.point_last_mort = nn.Linear(input_size, self.last_linear_size)

    def update_layer_info(self, layer, temp_k, point_size, dilation, stride):

        self.layers.append({})
        if point_size is not None:
            self.layers[layer]['point_size'] = point_size
        if temp_k is not None:
            padding = [(self.kernel_size - 1) * dilation, 0]  # [padding_left, padding_right]
            self.layers[layer]['temp_kernels'] = temp_k
            self.layers[layer]['dilation'] = dilation
            self.layers[layer]['padding'] = padding
            self.layers[layer]['stride'] = stride

    
    def _get_dims(self, layer, i):
        n_groups = self.F + self.Zt
        temp_in_channels = n_groups * (1 + self.Y) if i > 0 else self.F  # (F + Zt) * (Y + 1)
        linear_input_dim = (n_groups - self.Z) * self.Y + self.Z + self.F + 1 + self.n_flat_features  # (F + Zt-1) * Y + Z + F + 2 + n_flat_features
        temp_out_channels = n_groups * layer['temp_kernels']  # (F + Zt) * temp_kernels
        linear_output_dim = layer['point_size']
        
        return (temp_in_channels,
                linear_input_dim,
                temp_out_channels,
                linear_output_dim,
                n_groups)

    def _get_temp_point(self, layer, i):

        (temp_in_channels,
         linear_input_dim,
         temp_out_channels,
         linear_output_dim,
         n_groups) = self._get_dims(layer, i)
    
        temp = nn.Conv1d(temp_in_channels,
                         temp_out_channels,
                         kernel_size=self.kernel_size,
                         stride=layer['stride'],
                         dilation=layer['dilation'],
                         groups=n_groups)

        point = nn.Linear(linear_input_dim, linear_output_dim)
    
        bn_temp = self.init_batchnorm(temp_out_channels, self.momentum)
        bn_point = self.init_batchnorm(linear_output_dim, self.momentum)
    
        self.Z = linear_output_dim
    
        return temp, point, bn_temp, bn_point
        
    
    def create_temp_pointwise_layers(self):
        ### Notation used for tracking the tensor shapes ###

        # Z is the number of extra features added by the previous pointwise layer (could be 0 if this is the first layer)
        # Zt is the cumulative number of extra features that have been added by all previous pointwise layers
        # Zt-1 = Zt - Z (cumulative number of extra features minus the most recent pointwise layer)
        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)

        layer_modules = nn.ModuleDict()

        for i, layer in enumerate(self.layers):
            temp, point, bn_temp, bn_point = self._get_temp_point(layer, i)

            layer_modules[str(i)] = nn.ModuleDict({'temp': temp,
                                                   'bn_temp': bn_temp,
                                                   'point': point,
                                                   'bn_point': bn_point})
            self.Y = layer['temp_kernels']
            self.Zt += self.Z
        
        return layer_modules


    # This is really where the crux of TPC is defined. This function defines one TPC layer, as in Figure 3 in the paper:
    # https://arxiv.org/pdf/2007.09483.pdf
    def temp_pointwise(self,
                       B=None,
                       T=None,
                       X=None,
                       repeat_flat=None,
                       X_orig=None,
                       temp=None,
                       bn_temp=None,
                       point=None,
                       bn_point=None,
                       temp_kernels=None,
                       point_size=None,
                       padding=None,
                       prev_temp=None,
                       prev_point=None,
                       point_skip=None):

        ### Notation used for tracking the tensor shapes ###

        # Z is the number of extra features added by the previous pointwise layer (could be 0 if this is the first layer)
        # Zt is the cumulative number of extra features that have been added by all previous pointwise layers
        # Zt-1 = Zt - Z (cumulative number of extra features minus the most recent pointwise layer)
        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)
        # X shape: B * ((F + Zt) * (Y + 1)) * T; N.B exception in the first layer where there are also mask features, in this case it is B * 2F * T
        # repeat_flat shape: (B * T) * n_flat_features
        # X_orig shape: (B * T) * (2F + 2)
        # prev_temp shape: (B * T) * ((F + Zt-1) * (Y + 1))
        # prev_point shape: (B * T) * Z
        
        Z = prev_point.shape[1] if prev_point is not None else 0

        X_padded = pad(X, padding, 'constant', 0)  # B * ((F + Zt) * (Y + 1)) * (T + padding)
        
        X_temp = self.temp_dropout(bn_temp(temp(X_padded)))  # B * ((F + Zt) * temp_kernels) * T


        X_concat = cat(self.remove_none((prev_temp,  # (B * T) * ((F + Zt-1) * Y)
                                         prev_point,  # (B * T) * Z
                                         X_orig,  # (B * T) * (2F + 2)
                                         repeat_flat)),  # (B * T) * n_flat_features
                       dim=1)  # (B * T) * (((F + Zt-1) * Y) + Z + 2F + 2 + n_flat_features)
        


        point_output = self.main_dropout(bn_point(point(X_concat)))  # (B * T) * point_size

        # point_skip input: B * (F + Zt-1) * T
        # prev_point: B * Z * T
        # point_skip output: B * (F + Zt) * T
        point_skip = cat((point_skip, prev_point.view(B, T, Z).permute(0, 2, 1)), dim=1) if prev_point is not None else point_skip

        temp_skip = cat((point_skip.unsqueeze(2),  # B * (F + Zt) * 1 * T
                         X_temp.view(B, point_skip.shape[1], temp_kernels, T)),  # B * (F + Zt) * temp_kernels * T
                        dim=2)  # B * (F + Zt) * (1 + temp_kernels) * T

        X_point_rep = point_output.view(B, T, point_size, 1).permute(0, 2, 3, 1).repeat(1, 1, (1 + temp_kernels), 1)  # B * point_size * (1 + temp_kernels) * T

        X_combined = self.relu(cat((temp_skip, X_point_rep), dim=1))  # B * (F + Zt) * (1 + temp_kernels) * T

        next_X = X_combined.reshape(B, (point_skip.shape[1] + point_size) * (1 + temp_kernels), T)  # B * ((F + Zt + point_size) * (1 + temp_kernels)) * T

        temp_output = X_temp.permute(0, 2, 1).contiguous().view(B * T, point_skip.shape[1] * temp_kernels)  # (B * T) * ((F + Zt) * temp_kernels)

        return (temp_output,  # (B * T) * ((F + Zt) * temp_kernels)
                point_output,  # (B * T) * point_size
                next_X,  # B * ((F + Zt) * (1 + temp_kernels)) * T
                point_skip)  # for keeping track of the point skip connections; B * (F + Zt) * T

    def _split_msk(self, X):
        '''
        starts with 1 because time is the first column
        stops at self.F+1 because masks start at self.F+1
        '''
        ts = X[:, 1: self.F+1]
        msk = X[:, self.F+1:]
        return ts, msk

    def forward(self, X, flat, time_before_pred=5):

        # flat is B * n_flat_features
        # diagnoses is B * D
        # X is B * (2F + 2) * T
        # X_mask is B * T
        # (the batch is padded to the longest sequence, the + 1 is the time which is not for temporal convolution)
        X_ts, X_msk = self._split_msk(X)  # tuple ((B * F * T), (B * F * T))

        # prepare repeat arguments and initialise layer loop
        B, _, T = X_ts.shape

        repeat_flat = flat.repeat_interleave(T, dim=0)  # (B * T) * n_flat_features
        X_orig = (cat((X_ts,
                      X[:, 0, :].unsqueeze(1)), dim=1)
                  .permute(0, 2, 1)
                  .contiguous()
                  .view(B * T, self.F + 1))  # (B * T) * (F + 2)
        repeat_args = {'repeat_flat': repeat_flat,
                       'X_orig': X_orig,
                       'B': B,
                       'T': T}

        next_X = X_ts
        point_skip = X_ts  # keeps track of skip connections generated from linear layers; B * F * T
        temp_output = None
        point_output = None
        for i in range(self.n_layers):

            kwargs = dict(self.layer_modules[str(i)], **repeat_args)
            padding = self.layers[i]['padding']
            temp_kernels = self.layers[i]['temp_kernels']
            point_size = self.layers[i]['point_size']

            (temp_output,
             point_output,
             next_X,
             point_skip) = self.temp_pointwise(X=next_X,
                                               point_skip=point_skip,
                                               prev_temp=temp_output,
                                               prev_point=point_output,
                                               temp_kernels=temp_kernels,
                                               padding=padding,
                                               point_size=point_size,
                                               **kwargs)

        # (B * (T - time_before_pred)) * n_flat_features
        _flat = flat.repeat_interleave(T - time_before_pred, dim=0)
                    
        # (B * (T - time_before_pred)) * (((F + Zt) * (1 + Y)) + n_flat_features) for tpc
        _ts = (next_X[:, :, time_before_pred:]
               .permute(0, 2, 1)
               .contiguous()
               .view(B * (T - time_before_pred), -1))

        combined_features = cat((_flat, _ts), dim=1) 
            
        last_point_los = self.relu(
                         self.main_dropout(
                         self.bn_point_last_los(
                         self.point_last_los(
                         combined_features))))
        last_point_mort = self.relu(
                          self.main_dropout(
                          self.bn_point_last_mort(
                          self.point_last_mort(
                          combined_features))))
        
        point_final_los = self.point_final_los(last_point_los).view(B, T - time_before_pred)
        point_final_mort = self.point_final_mort(last_point_mort).view(B, T - time_before_pred)
        if not self.no_exp:
            los_predictions = exp(point_final_los)
        # B * (T - time_before_pred)
        los_predictions = self.hardtanh(point_final_los)
        # B * (T - time_before_pred)
        mort_predictions = self.sigmoid(point_final_mort)

        return los_predictions, mort_predictions

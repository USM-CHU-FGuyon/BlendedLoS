import torch.nn as nn
import torch

class MSEcalc:
    def __init__(self, sum_losses):
        self.sum_losses = sum_losses
        self.squared_error = nn.MSELoss(reduction='none')

    def mseloss(self, pred_los, true_los, mask, seq_length):

        pred_los = pred_los.where(mask, torch.zeros_like(true_los))
        true_los = true_los.where(mask, torch.zeros_like(true_los))
        
        loss = self.squared_error(pred_los, true_los).sum(dim=-1)
        if not self.sum_losses:
            loss = loss / seq_length.clamp(min=1)
        return loss.mean()

class MSELoss(nn.Module):
    def __init__(self, sum_losses=False):
        super().__init__()
        self.mseloss = MSEcalc(sum_losses).mseloss

    def forward(self, pred_los, true_los, mask, seq_length):
        return self.mseloss(pred_los, true_los, mask, seq_length)

class MSLELoss(nn.Module):
    def __init__(self, sum_losses=False):
        super().__init__()
        self.mseloss = MSEcalc(sum_losses).mseloss
        
    def forward(self, pred_los, true_los, mask, seq_length):
        return self.mseloss(pred_los.log(),
                            true_los.log(),
                            mask,
                            seq_length)

class MortLoss(nn.BCELoss):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred, label):
        label = label.unsqueeze(1).repeat(1, pred.shape[1])
        return super().forward(pred, label)*self.alpha

class Loss(nn.Module):
    def __init__(self,
                 los_task,
                 mort_task,
                 loss_type,
                 sum_losses,
                 alpha,
                 ignore_los_dc):
        super().__init__()
        
        self.msle_loss = MSLELoss(sum_losses)
        self.mse_loss = MSELoss(sum_losses)
        
        self.loss_type = loss_type
        self.los_task = los_task
        self.mort_task = mort_task
        self.ignore_los_dc = ignore_los_dc
        self.los_loss = self._los_loss()
        self.mort_loss = MortLoss(alpha)
        
        self.loss_value_mort = torch.tensor(0)
        self.loss_value_LoS = torch.tensor(0)
        self.loss_value_tot = torch.tensor(0)
        
    def _los_loss(self):
        return self.msle_loss if self.loss_type == 'msle' else self.mse_loss

    def forward(self,
                pred_los,
                pred_mort,
                true_los,
                true_mort,
                mask,
                seq_lengths):
        if self.ignore_los_dc:
            mask = mask.T.where(~true_mort.type(torch.bool), False).T
            
        if self.mort_task:
            self.loss_value_mort = self.mort_loss(pred_mort, true_mort)            

        if self.los_task:
            self.loss_value_LoS = self.los_loss(pred_los,
                                                true_los,
                                                mask,
                                                seq_lengths)

        self.loss_value_tot = self.loss_value_mort + self.loss_value_LoS
        return self.loss_value_mort, self.loss_value_LoS, self.loss_value_tot

from torch.optim import Adam

from models.cflstm_model import cfLSTM
from models.template import Template


class cfLSTMTemplate(Template):
    def __init__(self, config):
        super().__init__(config)
        self.model = cfLSTM(config=self.config,
                            device=self.device,
                            F=self.datareaders['train'].F,
                            D=self.datareaders['train'].D,
                            n_flat_features=self.datareaders['train'].n_flat_features
                            ).to(device=self.device)
        
        print(self.model)
        self.optimiser = Adam(self.model.parameters(), 
                              lr=self.config.learning_rate, 
                              weight_decay=self.config.L2_regularisation)

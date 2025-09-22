from typing import Tuple
import copy
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch
from easyGOOD.utils.register import register

@register.model_register
class Mixup(torch.nn.Module):
    r"""
    Implementation of the Mixup algorithm from "Mixup for Node and Graph Classification"
    (https://dl.acm.org/doi/abs/10.1145/3442381.3449796) paper
    """
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts):
        super(Mixup, self).__init__()
        self.lam = None
        self.data_perm = None
        self.id_a2b = None
        config = args_dicts['config']
        self.config = config        

        self.encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        self.classifier = torch.nn.Linear(hidden, output_dim)
    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.classifier(x)
        return x        
    def input_preprocess(self, data, targets, mask, node_norm, training, device):
        """
        Set input data and mask format to prepare for mixup
        """
        if training:
            targets = targets.float()
            alpha = self.config.ood.ood_param  # 2,4
            self.lam = np.random.beta(alpha, alpha)
            mixup_size = data.y.shape[0]
            self.id_a2b = torch.randperm(mixup_size)
            if node_norm is not None:
                self.data_perm, self.id_a2b = self._shuffle_data(data)
            mask = mask & mask[self.id_a2b]
        else:
            self.lam = 1
            self.id_a2b = torch.arange(data.num_nodes, device=self.config.device)

        return data, targets, mask, node_norm

    def loss_calculate(self, raw_pred, targets, mask, node_norm, config):
        """
        Calculate loss based on Mixup algorithm
        """
        loss_a = self.config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        loss_b = self.config.metric.loss_func(raw_pred, targets[self.id_a2b], reduction='none') * mask
        
        if self.config.model.model_level == 'node':
            loss_a = loss_a * node_norm * mask.sum()
            loss_b = loss_b * node_norm[self.id_a2b] * mask.sum()
            
        loss = self.lam * loss_a + (1 - self.lam) * loss_b
        return loss

    def _shuffle_data(self, data):
        """
        Prepare data and index for node mixup
        """
        data = copy.deepcopy(data)
        data.train_id = torch.nonzero(data.train_mask)
        data.val_id = torch.nonzero(data.val_mask)
        data.test_id = torch.nonzero(data.test_mask)
        
        # Identify new id by providing old id value
        id_a2b = torch.arange(data.num_nodes, device=self.config.device)
        train_id_shuffle = copy.deepcopy(data.train_id)
        train_id_shuffle = train_id_shuffle[torch.randperm(train_id_shuffle.shape[0])]
        id_a2b[data.train_id] = train_id_shuffle
        data = self._id_node(data, id_a2b)

        return data, id_a2b
    
    def _id_node(self, data, id_a2b):
        """
        Mixup node according to given index
        """
        data.x = None
        data.y[data.val_id] = -1
        data.y[data.test_id] = -1
        data.y = data.y[id_a2b]

        data.train_id = None
        data.test_id = None
        data.val_id = None

        id_b2a = torch.zeros(id_a2b.shape[0], dtype=torch.long, device=self.config.device)
        id_b2a[id_a2b] = torch.arange(0, id_a2b.shape[0], dtype=torch.long, device=self.config.device)
        row = data.edge_index[0]
        col = data.edge_index[1]
        row = id_b2a[row]
        col = id_b2a[col]
        data.edge_index = torch.stack([row, col], dim=0)

        return data
    
    def output_postprocess(self, model_output):
        return model_output
    
    def loss_postprocess(self, loss, data, mask, config):
        mean_loss = loss.sum() / mask.sum()
        return mean_loss
    def reset_classifier(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, 0)
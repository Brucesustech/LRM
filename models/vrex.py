from models.encoder import GCN_Encoder, GAT_Encoder
import torch
from easyGOOD.utils.register import register
import numpy as np


@register.model_register
class VREx(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts):
        super(VREx, self).__init__()

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
    
    def output_postprocess(self, model_output):
        return model_output
    
    def loss_postprocess(self, loss, data, mask, config):
        loss_list = []
        for i in range(config.dataset.num_envs):
            env_idx = data.env_id == i
            if loss[env_idx].shape[0] > 0 and mask[env_idx].sum() > 0:
                loss_list.append(loss[env_idx].sum() / mask[env_idx].sum())
        spec_loss = config.ood.ood_param * torch.var(torch.stack(loss_list))
        if torch.isnan(spec_loss):
            spec_loss = 0            
        mean_loss = loss.sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0, device=loss.device)
        loss = spec_loss + mean_loss
        return loss
    
    def reset_classifier(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, 0)
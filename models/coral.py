"""
Implementation of the Deep Coral algorithm from `"Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
<https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35>`_ paper
"""
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from easyGOOD.utils.register import register


def compute_covariance(input_data, device):
    r"""
    Compute Covariance matrix of the input data

    Args:
        input_data (Tensor): feature of the input data
        device: torch device

    Returns (Tensor):
        covariance value of the input features
    """
    n = input_data.shape[0]  # batch_size

    id_row = torch.ones((1, n), device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c


@register.model_register
class Coral(nn.Module):
    r"""
    The Graph Neural Network modified from the `"Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
    <https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35>`_ paper.
    """

    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts):
        config = args_dicts['config']
        self.config = config
        super(Coral, self).__init__()
        
        # Use the dynamically selected encoder
        self.encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        
        # Classifier
        self.classifier = torch.nn.Linear(hidden, output_dim)
        
        # For storing intermediate values
        self.feat = None

    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        r"""
        The Deep Coral model implementation.

        Returns:
            During training: Tuple[Tensor, Tensor] - [label predictions, features]
            During testing: Tensor - label predictions
        """
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                out_readout = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            out_readout = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        if self.training:
            # When training, expect x and edge_index as separate parameters
            self.feat = out_readout  # Store features for later use in loss calculation
            out = self.classifier(out_readout)
            return out, out_readout
        else:
            # During testing, only return the predictions
            out = self.classifier(out_readout)
            return out
    
    def reset_classifier(self):
        """Reset the parameters of the classifier layer"""
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, 0)
        
    def output_postprocess(self, model_output):
        """Process the model output depending on whether it's in training or evaluation mode"""
        # During training, model_output is a tuple of (class_output, features)
        # During testing, model_output is just class_output
        if isinstance(model_output, tuple):
            self.feat = model_output[1]
            return model_output[0]
        return model_output

    def loss_postprocess(self, loss, data, mask, config, device):
        """
        Process loss based on Deep Coral algorithm

        Args:
            loss: Base loss between model predictions and input labels
            data: Input data batch
            mask: Mask for valid data points
            config: Configuration object

        Returns:
            Modified loss incorporating the Coral regularization
        """
        loss_list = []
        covariance_matrices = []
        
        # Calculate covariance matrices for each environment
        for i in range(config.dataset.num_envs):
            env_idx = data.env_id == i
            env_feat = self.feat[env_idx]
            if env_feat.shape[0] > 1:
                covariance_matrices.append(compute_covariance(env_feat, device))
            else:
                covariance_matrices.append(None)

        # Calculate pairwise covariance differences
        for i in range(config.dataset.num_envs):
            for j in range(config.dataset.num_envs):
                if i != j and covariance_matrices[i] is not None and covariance_matrices[j] is not None:
                    dis = covariance_matrices[i] - covariance_matrices[j]
                    cov_loss = torch.mean(torch.mul(dis, dis)) / 4
                    loss_list.append(cov_loss)

        # Calculate total Coral loss
        if len(loss_list) == 0:
            coral_loss = torch.tensor(0, device=config.device)
        else:
            coral_loss = sum(loss_list) / len(loss_list)
        
        # Add Coral loss to original loss
        spec_loss = config.ood.ood_param * coral_loss
        if torch.isnan(spec_loss):
            spec_loss = 0
        
        mean_loss = loss.sum() / mask.sum()
        loss = mean_loss + spec_loss
        
        
        return loss
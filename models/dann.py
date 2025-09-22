from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
from easyGOOD.utils.register import register


class GradientReverseLayerF(Function):
    r"""
    Gradient reverse layer for DANN algorithm.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        r"""
        gradient forward propagation

        Args:
            ctx (object): object of the GradientReverseLayerF class
            x (Tensor): feature representations
            alpha (float): the GRL learning rate

        Returns (Tensor):
            feature representations

        """
        ctx.alpha = alpha
        return x.view_as(x)  # * alpha

    @staticmethod
    def backward(ctx, grad_output):
        r"""
        gradient backpropagation step

        Args:
            ctx (object): object of the GradientReverseLayerF class
            grad_output (Tensor): raw backpropagation gradient

        Returns (Tensor):
            backpropagation gradient

        """
        output = grad_output.neg() * ctx.alpha
        return output, None


@register.model_register
class DANN(nn.Module):
    r"""
    The Graph Neural Network modified from the `"Domain-Adversarial Training of Neural Networks"
    <https://www.jmlr.org/papers/volume17/15-239/15-239.pdf>`_ paper and `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.
    """

    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts):
        config = args_dicts['config']
        self.config = config
        super(DANN, self).__init__()
        self.dc_pred = None
        # Use the dynamically selected encoder
        self.encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        
        # Classifier and domain classifier
        self.classifier = torch.nn.Linear(hidden, output_dim)
        self.dc = nn.Linear(hidden, config.dataset.num_envs)

    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        r"""
        The DANN model implementation.

        Returns (Tensor):
            [label predictions, domain predictions]
        """
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                out_readout = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            out_readout = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        if self.training:
            # When training, expect x and edge_index as separate parameters
            dc_out = GradientReverseLayerF.apply(out_readout, self.config.train.alpha)
            dc_out = self.dc(dc_out)
            out = self.classifier(out_readout)
            return out, dc_out
        else:
            # During testing, simplified forward pass
            out = self.classifier(out_readout)
            return out
    
    def reset_classifier(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, 0)
        torch.nn.init.xavier_uniform_(self.dc.weight.data)
        torch.nn.init.constant_(self.dc.bias.data, 0)
        
    def output_postprocess(self, model_output):
        # During training, model_output is a tuple of (class_output, domain_output)
        # During testing, model_output is just class_output
        if isinstance(model_output, tuple):
            self.dc_pred = model_output[1]
            return model_output[0]
        return model_output
    def loss_postprocess(self, loss, data, mask, config):
        if config.model.model_level == 'node':
            dc_loss: torch.Tensor = config.metric.cross_entropy_with_logit(self.dc_pred[data.train_mask],
                                                             data.env_id[data.train_mask], reduction='none')
        else:
            dc_loss: torch.Tensor = config.metric.cross_entropy_with_logit(self.dc_pred, data.env_id, reduction='none')
        # else:
        # dc_loss: torch.Tensor = binary_cross_entropy_with_logits(dc_pred, torch.nn.functional.one_hot(data.env_id % config.dataset.num_envs, num_classes=config.dataset.num_envs).float(), reduction='none') * mask
        spec_loss = config.ood.ood_param * dc_loss.mean()
        mean_loss = loss.sum() / mask.sum()
        loss = mean_loss + spec_loss
        # self.mean_loss = mean_loss
        # self.spec_loss = spec_loss
        return loss
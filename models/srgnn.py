"""
Implementation of the SRGNN algorithm from `"Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data"
<https://proceedings.neurips.cc/paper/2021/hash/eb55e369affa90f77dd7dc9e2cd33b16-Abstract.html>`_ paper
"""
import torch
from torch import Tensor
import torch_geometric
from torch_geometric.data import Batch
import numpy as np
from typing import Tuple
from easyGOOD.utils.register import register
from torch_geometric.utils import subgraph
from cvxopt import matrix, solvers


def pairwise_distances(x, y=None):
    r"""
    Computation tool for pairwise distances

    Args:
        x (Tensor): a Nxd matrix
        y (Tensor): an optional Mxd matirx

    Returns (Tensor):
        dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
        if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def KMM(X, Xtest, device, _A=None, _sigma=1e1, beta=0.2):
    r"""
    Kernel mean matching (KMM) to compute the weight for each training instance

    Args:
        X (Tensor): training instances to be matched
        Xtest (Tensor): IID samples to match the training instances
        device: torch device
        _A (numpy array): one hot matrix of the training instance labels
        _sigma (float): normalization term
        beta (float): regularization weight

    Returns:
        - KMM_weight (numpy array) - KMM_weight to match each training instance
        - MMD_dist (Tensor) - MMD distance
    """
    H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(
        - 1e-3 * pairwise_distances(X))
    f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(
        - 1e-3 * pairwise_distances(X, Xtest))
    z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(
        - 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
    H /= 3
    f /= 3
    MMD_dist = H.mean() - 2 * f.mean() + z.mean()

    nsamples = X.shape[0]
    f = - X.shape[0] / Xtest.shape[0] * f.matmul(torch.ones((Xtest.shape[0], 1), device=device))
    G = - np.eye(nsamples)
    _A = _A[~np.all(_A == 0, axis=1)]
    b = _A.sum(1)
    h = - beta * np.ones((nsamples, 1))

    solvers.options['show_progress'] = False
    sol = solvers.qp(matrix(H.cpu().numpy().astype(np.double)), 
                    matrix(f.cpu().numpy().astype(np.double)), 
                    matrix(G), matrix(h), matrix(_A), matrix(b))
    return np.array(sol['x']), MMD_dist.item()


def l2diff(x1, x2):
    r"""
    Standard euclidean norm
    """
    return (x1 - x2).norm(p=2)


def moment_diff(sx1, sx2, k):
    r"""
    Difference between moments
    """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    return l2diff(ss1, ss2)


def cmd(X, X_test, K=5):
    r"""
    Central moment discrepancy (CMD).
    From "Central moment discrepancy (CMD) for domain-invariant representation learning", ICLR, 2017.

    Args:
        X (Tensor): training instances
        X_test (Tensor): IID samples
        K (int): number of approximation degrees

    Returns (Tensor):
        Central moment discrepancy
    """
    x1 = X
    x2 = X_test
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1, mx2)
    scms = [dm]
    for i in range(K - 1):
        # moment diff of centralized samples
        scms.append(moment_diff(sx1, sx2, i + 2))
    return sum(scms)


@register.model_register
class SRGNN(torch.nn.Module):
    r"""
    Implementation of the SRGNN algorithm from `"Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data"
    <https://proceedings.neurips.cc/paper/2021/hash/eb55e369affa90f77dd7dc9e2cd33b16-Abstract.html>`_ paper
    """

    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", 
                 dropout=0.5, use_bn=True, last_activation=False, **args_dicts):
        super(SRGNN, self).__init__()
        config = args_dicts['config']
        self.config = config        
        
        # Use dynamically selected encoder
        self.encoder = register.encoders[args_dicts['encoder_name']](
            input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        
        # Classifier layer
        self.classifier = torch.nn.Linear(hidden, output_dim)
        
        # For storing intermediate values
        self.feat = None
        self.kmm_weight = None


    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        """
        Forward pass that returns classification output and features for SRGNN
        """
        if self.training:            
            # Get feature representations from encoder
            features = self.encoder(x, edge_index)
            self.feat = features  # Store features for later use in loss calculation            
            # Get classification output
            out = self.classifier(features)            
            return out, features
        else:
            # During testing           
            features = self.encoder(x, edge_index)
            out = self.classifier(features)            
            return out
    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        """
        Forward pass that returns classification output and features for SRGNN
        """
        # Handle frozen parameter
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                features = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            features = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        
        # Store features during training mode
        if self.training:
            self.feat = features  # Store features for later use in loss calculation
            out = self.classifier(features)            
            return out, features
        else:
            # During testing           
            out = self.classifier(features)            
            return out
        
    def reset_classifier(self):
        """Reset the parameters of the classifier layer"""
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, 0)
    
    def input_preprocess(self, data, targets, mask, node_norm, training, device):
        """
        Set input data and mask format to prepare for SRGNN
        """
        if training:
            # Initialize KMM weights
            self.kmm_weight = torch.zeros(data.y.shape[0], device=device)
            
            # Get adjacency matrix for training nodes
            Z_all = torch_geometric.utils.to_dense_adj(
                data.edge_index[:, data.train_mask[data.edge_index[0]] | data.train_mask[data.edge_index[1]]],
                torch.zeros(data.y.shape[0], dtype=torch.long, device=device)).squeeze()
            
            # Get adjacency matrix for validation nodes
            Z_val = torch_geometric.utils.to_dense_adj(
                data.edge_index[:, data.val_mask[data.edge_index[0]] | data.val_mask[data.edge_index[1]]],
                torch.zeros(data.y.shape[0], dtype=torch.long, device=device)).squeeze()
            
            # Calculate KMM weights for each environment
            for i in range(self.config.dataset.num_envs):
                env_idx = (data.env_id == i).clone().detach()
                if data.y[env_idx].shape[0] > 0:
                    Z_train = Z_all[env_idx]
                    
                    # Select test samples from validation set
                    if data.val_mask.sum() >= data.y[env_idx].shape[0]:
                        Z_test = Z_val[data.val_mask][torch.randperm(data.val_mask.sum())][:(data.y[env_idx].shape[0])]
                    else:
                        Z_test = Z_val[torch.randperm(Z_val.shape[0])][:(data.y[env_idx].shape[0])]
                    
                    # Handle binary classification case
                    if self.config.dataset.num_classes == 1:
                        num_classes = 2
                    else:
                        num_classes = self.config.dataset.num_classes
                    
                    # Create label balance constraints
                    label_balance_constraints = torch.eye(num_classes, device=device)[data.y[env_idx].long().squeeze()].T.double().cpu().detach().numpy()
                    
                    # Calculate KMM weights
                    kmm_weight_env, MMD_dist = KMM(Z_train, Z_test, device, label_balance_constraints, beta=0.2)
                    self.kmm_weight[[env_idx.nonzero().squeeze()]] = torch.from_numpy(kmm_weight_env).float().to(device=device).squeeze()

        return data, targets, mask, node_norm

    def output_postprocess(self, model_output):
        """
        Process the raw output of model; get feature representations
        """
        # During training, model_output is a tuple (predictions, features)
        if isinstance(model_output, tuple):
            self.feat = model_output[1]
            return model_output[0]
        # During testing, model_output is just predictions
        return model_output

    def loss_postprocess(self, loss, data, mask, config, device):
        """
        Process loss based on SRGNN algorithm
        """
        SRloss_list = []
        
        # Calculate shift-robust loss for each environment
        for i in range(config.dataset.num_envs):
            env_idx_1 = data.env_id == i
            env_feat_1 = self.feat[env_idx_1]
            
            if env_feat_1.shape[0] > 1:
                # Select features from validation set or randomly
                if data.val_mask.sum() >= data.y[env_idx_1].shape[0]:
                    env_feat_2 = self.feat[data.val_mask][torch.randperm(data.val_mask.sum())][:(data.y[env_idx_1].shape[0])]
                else:
                    env_feat_2 = self.feat[torch.randperm(self.feat.shape[0])][:(data.y[env_idx_1].shape[0])]

                # Calculate central moment discrepancy
                shift_robust_loss = cmd(env_feat_1, env_feat_2)
                SRloss_list.append(shift_robust_loss)

        # Calculate total shift-robust loss
        if len(SRloss_list) == 0:
            SRloss = torch.tensor(0, device=device)
        else:
            SRloss = sum(SRloss_list) / len(SRloss_list)
        
        # Add SR loss to original loss with KMM weighting
        spec_loss = config.ood.ood_param * SRloss
        if torch.isnan(spec_loss):
            spec_loss = 0

        if self.kmm_weight.size(0) != loss.size(0):
        
            if self.kmm_weight.size(0) > loss.size(0):
         
                self.kmm_weight = self.kmm_weight[:loss.size(0)]
            else:
          
                padding = torch.ones(loss.size(0) - self.kmm_weight.size(0), device=config.device)
                self.kmm_weight = torch.cat([self.kmm_weight, padding])        
        mean_loss = (self.kmm_weight * loss).sum() / mask.sum()
        total_loss = mean_loss + spec_loss
        
        
        return total_loss
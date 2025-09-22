import torch
from torch import Tensor
from typing import Optional, Tuple
from torch_geometric.utils import dropout_adj
import torch.nn.functional as F
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def DropNode(x, mask_prob):
    mask = torch.rand(x.size(0), device=x.device) < mask_prob  
    x = x.clone()
    x[mask] = 0  
    return x

def shuffle_feature(x, shuffle_prob=0.2):
    x = x.clone()
    num_nodes = x.size(0)
    mask = torch.rand(num_nodes, device=x.device) < shuffle_prob  
    perm = torch.randperm(num_nodes, device=x.device) 
    x[mask] = x[perm[mask]]  
    return x
def mixup_feature(x, alpha=0.2):
    x = x.clone() 
    lam = torch.distributions.Beta(alpha, alpha).sample((x.size(0), 1)).to(x.device)  
    perm = torch.randperm(x.size(0), device=x.device)  
    x = lam * x + (1 - lam) * x[perm]  
    return x
# Add Graph_Editer class for edge perturbation
class Graph_Editer(torch.nn.Module):
    def __init__(self, K, edge_num, device):
        super(Graph_Editer, self).__init__()
        self.K = K
        self.edge_num = edge_num
        self.S = 0.01
        self.sample_size = int(self.S * edge_num)
        self.B = torch.nn.Parameter(torch.FloatTensor(K, edge_num))
        self.epsilon = 0.000001
        self.temperature = 1.0
        self.device = device

    def reset_parameters(self):
        torch.nn.init.uniform_(self.B)

    def kld(self, mask):
        pos = mask
        neg = 1 - mask
        kld_loss = torch.mean(pos * torch.log(pos/0.5 + 0.00000001) + neg * torch.log(neg/0.5 + 0.000000001))
        return kld_loss

    def forward(self, k):
        Bk = self.B[k]
        mask = torch.clamp(Bk, -10, 10).to(self.device)
        mask = torch.sigmoid(mask)
        sample_mask = self.straight_through(mask)
        kld_loss = self.kld(mask)
        return sample_mask, kld_loss

    def sample(self, k):
        Bk = self.B[k]
        mask = torch.clamp(Bk, -10, 10).to(self.device)
        mask = torch.sigmoid(mask)
        mask = self.straight_through(mask)
        return mask

    def straight_through(self, mask):
        _, idx = torch.topk(mask, self.sample_size)
        sample_mask = torch.zeros_like(mask).to(self.device)
        sample_mask[idx]=1
        return sample_mask + mask - mask.detach()
# New edge perturbation function based on Graph_Editer
def edge_perturbation(edge_index, K, device):
    """
    Creates multiple perturbed versions of the edge_index using Graph_Editer
    
    Args:
        edge_index: Original edge index
        K: Number of perturbations to create
        device: Device to place tensors on
        
    Returns:
        List of perturbed edge indices
    """
    num_edges = edge_index.size(1)
    editer = Graph_Editer(K, num_edges, device)
    editer.reset_parameters()
    
    perturbed_edges = []
    for k in range(K):
        mask, _ = editer(k)
        # Apply mask to edge_index
        selected_edges = torch.where(mask > 0.5)[0]
        perturbed_edge_index = edge_index[:, selected_edges]
        perturbed_edges.append(perturbed_edge_index)
        
    return perturbed_edges

def flag_feature_augmentation(x, edge_index, model, step_size=0.01, m=3, device=None):
    """
    Apply FLAG method to enhance node features
    
    Args:
        x: Input node features
        edge_index: Edge indices
        model: Model used for gradient computation
        step_size: Perturbation step size
        m: Number of gradient ascent steps
        device: Computation device
        
    Returns:
        Enhanced features with adversarial perturbation
    """
    if device is None:
        device = x.device
        
    # Create a copy of features that requires gradients
    x_perturb = x.clone().detach().requires_grad_(True)
    
    # Get model output for gradient computation
    out = model(x_perturb, edge_index)
    
    # Create a simple target to compute gradients (maximizing output magnitude)
    target = torch.ones_like(out).to(device)
    loss = F.mse_loss(out, target)
    
    # Compute gradients
    loss.backward()
    grad = x_perturb.grad.detach()
    
    # Apply FLAG perturbation
    perturb = step_size * torch.sign(grad)
    x_augmented = x.detach() + perturb
    
    return x_augmented

def adversarial_aug_train(model, node_attack, perturb_shapes, step_size, m, device):
    model.train()

    perturb1 = torch.FloatTensor(*perturb_shapes[0]).uniform_(-step_size, step_size).to(device)
    perturb2 = torch.FloatTensor(*perturb_shapes[1]).uniform_(-step_size, step_size).to(device)
    
    perturb1.requires_grad_()
    perturb2.requires_grad_()

    loss = node_attack(perturb1, perturb2)
    loss /= m

    for i in range(m - 1):
        loss.backward()

        perturb1_data = perturb1.detach() + step_size * torch.sign(perturb1.grad.detach())
        perturb2_data = perturb2.detach() + step_size * torch.sign(perturb2.grad.detach())


        perturb1.data = perturb1_data.data
        perturb2.data = perturb2_data.data

        perturb1.grad[:] = 0
        perturb2.grad[:] = 0

        loss = node_attack(perturb1, perturb2)
        loss /= m

    return loss


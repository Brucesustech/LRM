from easyGOOD.utils.register import register
from easyGOOD.data import create_dataloader, load_dataset 
from easyGOOD.utils.initial import reset_random_seed
from easyGOOD.utils.config_reader import config_summoner
from easyGOOD.utils.args import args_parser
from eval_utils import nan2zero_get_mask, evaluate, evaluate_all_with_scores, k_fold,evaluate_all_with_calibrated_scores
from augmentation import drop_feature, adversarial_aug_train, DropNode, shuffle_feature, mixup_feature,edge_perturbation, flag_feature_augmentation
from writer import write_all_in_pic, write_res_in_log, save_ckpts, load_ckpts
from optimizers import CosineDecayScheduler
from models import load_model
import torch
from tqdm import tqdm
from torch_geometric.utils import dropout_adj
from datetime import datetime
from rich.progress import track
from collections import defaultdict, Counter
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import matplotlib
import gc
import numpy as np
import json
import matplotlib.pyplot as plt
import time  
import matplotlib.pyplot as plt
import os

def visualize_misclassification2(misclassified_counts, save_path_base, title='Misclassification Distribution'):
    """Visualize misclassification distribution with a histogram"""
    matplotlib.use('Agg')  # Non-interactive backend
    plt.rcParams['figure.max_open_warning'] = 1
    plt.rcParams['interactive'] = False
    
    plt.close('all')
    
    dpi = 600
    
    datasets = ['id_test', 'ood_test']
    dataset_titles = ['ID Test', 'OOD Test']
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=dpi)
        axes = axes.flatten()
        
        for idx, (dataset, dataset_title) in enumerate(zip(datasets, dataset_titles)):
            ax = axes[idx]
            
            target_counts = misclassified_counts[dataset]['target_counts']
            total_errors = misclassified_counts[dataset]['total_errors']
            
            if not target_counts:
                ax.text(0.5, 0.5, f"No errors in {dataset_title}", 
                       ha='center', va='center', fontsize=18)  
                continue
            
            # Show top 20 most common misclassification targets
            top_targets = target_counts.most_common(20)
            labels = [f"Label {label}" for label, _ in top_targets]
            counts = [count for _, count in top_targets]

            labels.append("Total Errors")
            counts.append(total_errors)

            bars = ax.bar(labels, counts, color='skyblue')

            bars[-1].set_color('salmon')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{int(height)}', ha='center', va='bottom', fontsize=14)  

            ax.set_title(f"{dataset_title} - Most Common Misclassification Targets", fontsize=18) 
            ax.set_xlabel('Predicted Label', fontsize=16)  
            ax.set_ylabel('Number of Misclassifications', fontsize=16) 

            ax.set_xticks(range(len(labels)))  
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)  
            
            ax.tick_params(axis='y', labelsize=16)
  
            y_max = max(counts) * 1.15
            ax.set_ylim(0, y_max)
        

        plt.tight_layout()
        
        output_path = f"{os.path.splitext(save_path_base)[0]}_histogram.png"
        plt.savefig(output_path)
        plt.close()       
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        
    finally:
        plt.close('all')
        gc.collect()

def visualize_misclassification(misclassified_counts, type, save_path_base, title='Misclassification Distribution'):
    """Visualize misclassification distribution with a histogram"""
    matplotlib.use('Agg')  # Non-interactive backend
    plt.rcParams['figure.max_open_warning'] = 1
    plt.rcParams['interactive'] = False
    
    plt.close('all')
    
    dpi = 600
    
    datasets = [type]  # Only one dataset, using the passed 'type' parameter
    dataset_titles = [title]  # Using the title parameter
    
    try:
        fig, ax = plt.subplots(figsize=(20, 8), dpi=dpi)  # Single plot instead of 2x2 grid
        
        dataset = datasets[0]
        dataset_title = dataset_titles[0]
        
        target_counts = misclassified_counts[dataset]['target_counts']
        total_errors = misclassified_counts[dataset]['total_errors']
        
        if not target_counts:
            ax.text(0.5, 0.5, f"No errors in {dataset_title}", 
                   ha='center', va='center', fontsize=18)  
        else:
            # Show top 20 most common misclassification targets
            top_targets = target_counts.most_common(20)
            labels = [f"Label {label}" for label, _ in top_targets]
            counts = [count for _, count in top_targets]

            labels.append("Total Errors")
            counts.append(total_errors)

            bars = ax.bar(labels, counts, color='skyblue')

            bars[-1].set_color('salmon')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,  
                       f'{int(height)}', ha='center', va='bottom', fontsize=14)  

            ax.set_title(f"{dataset_title}", fontsize=18) 
            ax.set_xlabel('Predicted Label', fontsize=16)  
            ax.set_ylabel('Number of Misclassifications', fontsize=16)  

            ax.set_xticks(range(len(labels)))  
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)  
            
            ax.tick_params(axis='y', labelsize=16)
  
            y_max = max(counts) * 1.15
            ax.set_ylim(0, y_max)
        
        plt.tight_layout()
        
        output_path = f"{os.path.splitext(save_path_base)[0]}_histogram.png"
        plt.savefig(output_path)
        plt.close()       
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        
    finally:
        plt.close('all')
        gc.collect()

def count_misclassified_samples(model, loader, criterion, config, device):
    """Count misclassified samples for each dataset and target label"""

    misclassified_counts = {}
    for dataset in ['id_val', 'id_test', 'ood_val', 'ood_test']:
        misclassified_counts[dataset] = {
            'per_true_label': defaultdict(Counter),
            'target_counts': Counter(),
            'total_errors': 0
        }
    
    masks = {'id_val': 'id_val_mask', 'id_test': 'id_test_mask', 
            'ood_val': 'val_mask', 'ood_test': 'test_mask'}

    model.eval()
    
    with torch.no_grad():
        for data in loader['test']:
            data = data.to(device)
            preds = model(data.x, data.edge_index, edge_weight=None)
            preds = preds.argmax(dim=1)
            
            for key, mask_name in masks.items():
                mask = getattr(data, mask_name, None)
                if mask is not None:
                    true_labels = data.y[mask]
                    pred_labels = preds[mask]
                    if true_labels.dim() > 1:
                        true_labels = true_labels.squeeze(-1)
                    if pred_labels.dim() > 1:
                        pred_labels = pred_labels.squeeze(-1)
                    for true, pred in zip(true_labels.tolist(), pred_labels.tolist()):
                        true = int(true)
                        pred = int(pred)
                        if true != pred:
                            misclassified_counts[key]['per_true_label'][true][pred] += 1
                            misclassified_counts[key]['target_counts'][pred] += 1
                            misclassified_counts[key]['total_errors'] += 1
                
    return misclassified_counts

def calculate_validation_loss(model, loader, criterion, config, device, mask_type='id_val'):
    """
    Calculate validation loss on the specified mask type
    
    Args:
        model: The model to evaluate
        loader: Data loader dictionary
        criterion: Loss function
        config: Configuration
        device: Device to use for computation
        mask_type: Type of mask to use ('id_val' or 'val')
    
    Returns:
        Average loss on the specified validation set
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for data in loader[mask_type]:  
            data = data.to(device)
            # Get the appropriate mask and targets
            if mask_type == 'id_val':
                mask = data.id_val_mask
            else:  # 'val' for OOD validation
                mask = data.val_mask
                
            if mask is None or mask.sum() == 0:
                continue
                
            targets = data.y[mask]
            # Forward pass
            preds = model(data.x, data.edge_index, edge_weight=None)
            preds = preds[mask]
            
            # Calculate loss
            loss = criterion(preds, targets)
            # Calculate average loss
            batch_loss = loss.mean().item()
            batch_size = mask.sum().item()
            
            total_loss += batch_loss * batch_size
            total_samples += batch_size
    
    # Return average loss
    return total_loss / max(1, total_samples)

def calculate_test_loss(model, loader, criterion, config, device, mask_type='id_test'):
    """
    Calculate test loss on the specified mask type
    
    Args:
        model: The model to evaluate
        loader: Data loader dictionary
        criterion: Loss function
        config: Configuration
        device: Device to use for computation
        mask_type: Type of mask to use ('id_test' or 'test')
    
    Returns:
        Average loss on the specified test set
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for data in loader[mask_type]:  
            data = data.to(device)
            # Get the appropriate mask and targets
            if mask_type == 'id_test':
                mask = data.id_test_mask
            else:  # 'val' for OOD test
                mask = data.test_mask
                
            if mask is None or mask.sum() == 0:
                continue
                
            targets = data.y[mask]
            # Forward pass
            preds = model(data.x, data.edge_index, edge_weight=None)
            preds = preds[mask]
            
            # Calculate loss
            loss = criterion(preds, targets)
            # Calculate average loss
            batch_loss = loss.mean().item()
            batch_size = mask.sum().item()
            
            total_loss += batch_loss * batch_size
            total_samples += batch_size
    
    # Return average loss
    return total_loss / max(1, total_samples)

def print_label_distribution(dataset):
    masks = ['train_mask', 'id_val_mask', 'id_test_mask', 'val_mask', 'test_mask']
    mask_names = ['Train', 'ID Val', 'ID Test', 'OOD Val', 'OOD Test']
    
    for mask, name in zip(masks, mask_names):
        mask_tensor = getattr(dataset.data, mask, None)
        if mask_tensor is not None:
            labels = dataset.data.y[mask_tensor].tolist()
            label_counts = Counter(labels)
            total_samples = sum(mask_tensor.tolist())  
            print(f"\n{name} Set Label Distribution (Total: {total_samples} samples):")
            print(f"\n{name} Set Label Distribution:")
            for label, count in label_counts.items():
                print(f"  Label {label}: {count} samples")
        else:
            print(f"\n{name} Set: No mask found.")
def train_linear_head(model, config, ood_train=False):
    model.reset_classifier()
    classifier_optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=config.train.linear_head_lr, weight_decay=1e-4)
    for e in track(range(config.train.linear_head_epochs)):
        for data in loader['train']:
            model.classifier.train()
            data = data.to(device)
            labels = data.y if hasattr(data, 'y') else None
            node_norm = data.get('node_norm') if config.model.model_level == 'node' else None
            node_norm = torch.ones(data.x.shape[0], device=data.x.device) if node_norm == None else node_norm
            edge_weight = data.get('edge_norm') if config.model.model_level == 'node' else None
            mask, targets = nan2zero_get_mask(data, 'train', config)
            preds = model(data.x, data.edge_index, edge_weight = None, frozen=True)
            
            loss = criterion(preds, targets) * mask
            loss = loss * node_norm * mask.sum() if config.model.model_level == 'node' else loss 
            loss = loss.mean() / mask.sum()
            loss.backward()
            classifier_optimizer.step()
            classifier_optimizer.zero_grad()      
def pretrain(data, model, config, device):
    model.train()
    data = data.to(device)
    labels = data.y if hasattr(data, 'y') else None
    # if labels is not None:
    #     unique_labels, class_counts = torch.unique(labels, return_counts=True)
    #     print("\n=== Sample distribution in original data ===")
        
    #     # Convert class counts to numpy array for variance calculation
    #     class_counts_np = class_counts.cpu().numpy()
    #     original_variance = np.var(class_counts_np)
        
    #     for label, count in zip(unique_labels, class_counts):
    #         print(f"  Class {label.item()}: {count.item()} samples ({count.item()/len(labels)*100:.2f}%)")
        
    #     print(f"\nVariance of sample counts across classes in original data: {original_variance:.2f}")             
    x1, x2 = targeted_feature_augmentation(data.x, labels, major_class_ratio=config.aug.major_class_ratio,feature_mask_prob=config.aug.mask_feat1), targeted_feature_augmentation(data.x, labels, major_class_ratio=config.aug.major_class_ratio,feature_mask_prob=config.aug.mask_feat2)

    
    #x1, x2 = drop_feature(data.x, config.aug.mask_feat1), drop_feature(data.x, config.aug.mask_feat2)

    # x1, x2 = shuffle_feature(data.x, config.aug.mask_feat1), shuffle_feature(data.x, config.aug.mask_feat2)
    #x1, x2 = mixup_feature(data.x, config.aug.mask_feat1), mixup_feature(data.x, config.aug.mask_feat2)
    # x1, x2 = DropNode(data.x, config.aug.mask_feat1), DropNode(data.x, config.aug.mask_feat2)
    # #Statistical analysis of samples with all features set to zero across classes
    # if labels is not None:
    #     # Store the count of non-zero samples for each class
    #     non_zero_samples_view1 = []
        
    #     # print("\n=== Analysis of completely zeroed samples in View 1 by class ===")
    #     for label in unique_labels:
    #         class_mask = (labels == label)
    #         zero_samples = ((x1[class_mask].sum(dim=1) == 0).sum().item())
    #         total_samples = class_mask.sum().item()
    #         non_zero_count = total_samples - zero_samples
    #         non_zero_samples_view1.append(non_zero_count)
    #         print(f"  Class {label.item()}: {zero_samples}/{total_samples} samples completely zeroed ({zero_samples/total_samples*100:.2f}%)")
        
    #     # Calculate variance of non-zero sample counts after augmentation
    #     non_zero_variance_view1 = np.var(np.array(non_zero_samples_view1))
    #     print(f"\nVariance of non-zero sample counts in View 1 after augmentation: {non_zero_variance_view1:.2f}")
            
    #     # Perform the same calculation for View 2
    #     non_zero_samples_view2 = []
        
    #     # print("\n=== Analysis of completely zeroed samples in View 2 by class ===")
    #     for label in unique_labels:
    #         class_mask = (labels == label)
    #         zero_samples = ((x2[class_mask].sum(dim=1) == 0).sum().item())
    #         total_samples = class_mask.sum().item()
    #         non_zero_count = total_samples - zero_samples
    #         non_zero_samples_view2.append(non_zero_count)
    #         print(f"  Class {label.item()}: {zero_samples}/{total_samples} samples completely zeroed ({zero_samples/total_samples*100:.2f}%)")
        
    #     # Calculate variance of non-zero sample counts after augmentation
    #     non_zero_variance_view2 = np.var(np.array(non_zero_samples_view2))
    #     print(f"\nVariance of non-zero sample counts in View 2 after augmentation: {non_zero_variance_view2:.2f}")

    # Use FLAG for feature enhancement
    # step_size = config.aug.step_size if hasattr(config.aug, 'step_size') else 0.01
    # m = config.aug.m if hasattr(config.aug, 'm') else 3
    # x1, x2 = flag_feature_augmentation(data.x, data.edge_index, model, step_size, m, device),flag_feature_augmentation(data.x, data.edge_index, model, step_size, m, device)
   
    
    # Generate multiple perturbed versions of the edge_index(Lisa)
    # x1, x2 = data.x, data.x
    # K = config.aug.K if hasattr(config.aug, 'K') else 3
    # perturbed_edges = edge_perturbation(data.edge_index, K, device)
    # edge_index1, edge_index2 = perturbed_edges[0], perturbed_edges[1]
    
    
    edge_index1, edge_norm1 = dropout_adj(edge_index=data.edge_index, p=config.aug.mask_edge1)
    edge_index2, edge_norm2 = dropout_adj(edge_index=data.edge_index, p=config.aug.mask_edge2)

    x1, edge_index1, x2, edge_index2 = x1.to(device), edge_index1.to(device), x2.to(device), edge_index2.to(device)
    # original data
    data = data.to(device)
    x, edge_index, edge_weight = data.x, data.edge_index, None

    if config.aug.ad_aug:
        if config.model.model_name in ['GAE', 'VGAE', 'DGI', 'GraphMAE', "MVGRL"]:
            raise NotImplementedError(f'{config.model.model_name} can not use adversarial augmentation now!')
        
        model.update_prototypes(x1=x1, edge_index1=edge_index1, edge_weight1=None, x2=x2, edge_index2=edge_index2, edge_weight2=None)
            
        def node_attack(perturb1, perturb2):
            x1_noise = x1 + perturb1
            x2_noise = x2 + perturb2
            return model.pretrain(x1=x1_noise, edge_index1=edge_index1, edge_weight1=None, 
                                x2=x2_noise, edge_index2=edge_index2, edge_weight2=None)

        loss = adversarial_aug_train(model, node_attack, (x1.shape, x2.shape), 1e-3, 3, device)
    else:
        model.update_prototypes(x1=x1, edge_index1=edge_index1, edge_weight1=None, x2=x2, edge_index2=edge_index2, edge_weight2=None)
    
        loss = model.pretrain(data=data.to(device), x=x, edge_index=edge_index, edge_weight=edge_weight,
                                x1=x1, edge_index1=edge_index1, edge_weight1=None, x2=x2, edge_index2=edge_index2, edge_weight2=None)

    return loss

def targeted_feature_augmentation(x, labels, major_class_ratio=0.6, feature_mask_prob=0.2):

    
    x = x.clone()
    num_nodes = x.size(0)
    if len(labels.shape) > 1:
        labels = labels.flatten() 
    unique_labels, counts = torch.unique(labels, return_counts=True)
    total_nodes = float(num_nodes)
    
    class_ratios = counts.float() / total_nodes
    
    major_classes = unique_labels[class_ratios >= major_class_ratio]
    
    node_mask_probs = torch.ones(num_nodes, device=x.device) * feature_mask_prob 

    for label_idx, label in enumerate(unique_labels):
        ratio = class_ratios[label_idx]
        if label in major_classes:
            adjustment = 1.0 + ratio * 5
            # adjustment = 1.0 + ratio * 10

        else:
            adjustment = 0.0  
        
        class_mask = (labels == label)
        node_mask_probs[class_mask] *= adjustment

    node_mask_probs = torch.clamp(node_mask_probs, 0.0, 0.95)
    
    mask = torch.rand(num_nodes, device=x.device) < node_mask_probs
    true_count = mask.sum().item()
    total_elements = len(mask)

    x[mask] = 0.0
    
    return x

def visualize_misclassification3(misclassified_counts, type, save_path_base, title='Misclassification Distribution'):
    import matplotlib
    matplotlib.use('Agg')  
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    import os
    import gc

    plt.rcParams['figure.max_open_warning'] = 1
    plt.rcParams['interactive'] = False

    plt.close('all')

    dpi = 600

    try:
        dataset = type
        
        classes = set()
        
        for true_label in misclassified_counts[dataset]['per_true_label'].keys():
            classes.add(true_label)
            for pred_label in misclassified_counts[dataset]['per_true_label'][true_label].keys():
                classes.add(pred_label)

        MAX_CLASSES = 20
        if len(classes) > MAX_CLASSES:
            class_errors = defaultdict(int)
            for true_label, pred_dict in misclassified_counts[dataset]['per_true_label'].items():
                for pred_label, count in pred_dict.items():
                    class_errors[true_label] += count
                    class_errors[pred_label] += count
            
            classes = [cls for cls, _ in sorted(
                class_errors.items(), key=lambda x: x[1], reverse=True
            )[:MAX_CLASSES]]
        else:
            classes = sorted(classes)
        
        n_classes = len(classes)
        confusion_matrix = np.zeros((n_classes, n_classes))
        
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        for true_label in misclassified_counts[dataset]['per_true_label']:
            if true_label not in class_to_idx:
                continue
                
            true_idx = class_to_idx[true_label]
            total_errors = sum(misclassified_counts[dataset]['per_true_label'][true_label].values())
            
            if total_errors == 0:
                continue

            for pred_label, count in misclassified_counts[dataset]['per_true_label'][true_label].items():
                if pred_label not in class_to_idx:
                    continue
                    
                if true_label == pred_label:
                    continue
                    
                pred_idx = class_to_idx[pred_label]
                confusion_matrix[true_idx, pred_idx] = count / total_errors
        
        plt.figure(figsize=(10, 8), dpi=dpi)
        
        plt.imshow(confusion_matrix, cmap='Blues', vmin=0, vmax=1)
        
        cbar = plt.colorbar()
        cbar.set_label('Proportion')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Predicted label', fontsize=14)
        plt.ylabel('True label', fontsize=14)
        
        plt.xticks(np.arange(len(classes)), classes, rotation=45, ha='right', fontsize=14)
        plt.yticks(np.arange(len(classes)), classes, fontsize=14)
        
        for i in range(n_classes):
            for j in range(n_classes):
                value = confusion_matrix[i, j]
                if value > 0.0001:  
                    plt.text(j, i, f'{value:.2f}',
                          ha='center', va='center',
                          color='white' if value > 0.5 else 'black',
                          fontsize=14)  
        
        plt.tight_layout()
        output_path = f"{os.path.splitext(save_path_base)[0]}_simplified.png"
        plt.savefig(output_path)
        plt.close()
        
        plt.close('all')
        
        print(f"Saved simplified visualization to {output_path}")
        
    except Exception as e:
        import traceback
        print(f"Error in simplified visualization: {e}")
        traceback.print_exc()
        
    finally:
        plt.close('all')
        gc.collect()
        
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

if __name__ == '__main__':
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = args_parser()
    config = config_summoner(args) 
    reset_random_seed(config)
    dataset = load_dataset(config.dataset.dataset_name, config)
    loader = create_dataloader(dataset, config) 
    train_mask = dataset.data.train_mask
    train_labels = dataset.data.y[train_mask].long().squeeze()
    total_samples = dataset.data.x.shape[0]
    # training process, used for training linear head
    if dataset.num_classes > 2: 
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        # print_label_distribution(dataset)
    else: # binary classification
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    model = load_model(config.model.model_name, config).to(device)
    print(model)
    model_size = count_parameters(model)
    print(f"  Model size (MB): {model_size['total'] * 4 / (1024 * 1024):.2f}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=1e-4)
    if config.model.model_name in ['MARIO']:
        params = []
        for k, v in model.named_parameters():
            if 'projector' in k or 'prototypes' in k:
                continue
            else:
                params.append(v)

        optimizer = torch.optim.AdamW(params, lr=config.train.lr, weight_decay=1e-4)
    
    

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.train.mile_stones,gamma=0.1)
    # load checkpoint if specified
    if config.model.load_checkpoint:
        load_ckpts(model, config)

    ebar = tqdm(range(1, config.train.max_epoch+1))
    best_id_val, best_id_id_test, best_id_ood_test, best_ood_val, best_ood_ood_test = 0, 0, 0, 0, 0
    train_acc, id_val, id_test, ood_val, ood_test = 0, 0, 0, 0, 0
    
    best_id_misclassified = None
    best_ood_misclassified = None
    frame_count = 0  
    train_list, id_val_list, id_test_list, ood_val_list, ood_test_list = [], [], [], [], []
    id_val_loss = 0
    ood_val_loss = 0
    id_test_loss = 0
    ood_test_loss = 0
    train_loss_list = []
    id_val_loss_list = []
    ood_val_loss_list = []
    id_test_loss_list = []
    ood_test_loss_list = []
    cached_optimized_major_ratios = []
    if config.train.best_linear_head:
        print("Note: We will use the best linear head.")
    for e in ebar:
        if config.model.load_checkpoint: # use pre-trained model
            print("Load checkpoint, skip pre-training...")
            break
        epoch_loss = 0
        epoch_node_cnt = 0
        for data in loader['train']:
            #     continue
            optimizer.zero_grad()
            loss = pretrain(data, model, config, device)
            loss.backward()
            optimizer.step()
            # update target network
            lr_scheduler.step()              
            epoch_loss += loss.detach().item() * data.x.shape[0]
            epoch_node_cnt += data.x.shape[0]
        # Calculate average training loss for this epoch
        # avg_train_loss = epoch_loss / epoch_node_cnt
        # print(f"Epoch {e}, Average Training Loss: {avg_train_loss:.4f}")
        # train_loss_list.append(avg_train_loss)

        # train the classifier and eval
        # config.train.eval_step = 1
        if e % config.train.eval_step == 0:
            if e == config.train.max_epoch:
                break # we will evaluate model out of the loop

            train_linear_head(model, config)
            # curr_misclassified = count_misclassified_samples(model, loader, criterion, config, device)
            # eval
            # visualize_misclassification2(curr_misclassified, f'./supervisedcon/tu_data/frame_{frame_count:04d}.png', title=f'Epoch {e} Misclassification Distribution')
            # torch.save(curr_misclassified, f'./supervisedcon/tu_data/{config.model.model_name}_misclassified_{frame_count:04d}.pt')
            frame_count += 1
            train_acc, id_val, id_test, ood_val, ood_test = evaluate_all_with_scores(model, loader, criterion, config, device)
            train_list.append(train_acc), id_val_list.append(id_val), id_test_list.append(id_test), ood_val_list.append(ood_val), ood_test_list.append(ood_test)
            # Add code to calculate and log validation losses

            # id_val_loss = calculate_validation_loss(model, loader, criterion, config, device, 'id_val')
            # ood_val_loss = calculate_validation_loss(model, loader, criterion, config, device, 'val')
            # id_val_loss_list.append(id_val_loss)
            # ood_val_loss_list.append(ood_val_loss)
            # id_test_loss = calculate_test_loss(model, loader, criterion, config, device, 'id_test')
            # ood_test_loss = calculate_test_loss(model, loader, criterion, config, device, 'test')
            # id_test_loss_list.append(id_test_loss)
            # ood_test_loss_list.append(ood_test_loss)
            # id val
            if id_val > best_id_val:
                best_id_val, best_id_id_test, best_id_ood_test = id_val, id_test, ood_test
                # best_id_misclassified = curr_misclassified
            # ood val
            if ood_val > best_ood_val:
                best_ood_val, best_ood_ood_test = ood_val, ood_test
                # best_ood_misclassified = curr_misclassified
        ebar.set_postfix({'Train Loss': epoch_loss/epoch_node_cnt, 'train acc': train_acc,
                            'id val': id_val, 'id test': id_test,
                            'ood val': ood_val, 'ood test': ood_test})
        accs = [train_acc, id_val, id_test, ood_val, ood_test]
        # In your training loop
        # losses = {
        #     'train': avg_train_loss,
        #     'id_val': id_val_loss,
        #     'ood_val': ood_val_loss
        # }
    ##############################################################################
    # evaluate out of the loop
    train_linear_head(model, config)
    # misclassified_counts = count_misclassified_samples(model, loader, criterion, config, device)
#    visualize_misclassification(misclassified_counts, f'./tu/final_misclassification.png', 
#                                title='Final Misclassification Distribution')
    # eval
    train_acc, id_val, id_test, ood_val, ood_test = evaluate_all_with_scores(model, loader, criterion, config, device)
    train_list.append(train_acc), id_val_list.append(id_val), id_test_list.append(id_test), ood_val_list.append(ood_val), ood_test_list.append(ood_test)
    # final_misclassified = misclassified_counts
    # id val
    if id_val > best_id_val:
        best_id_val, best_id_id_test, best_id_ood_test = id_val, id_test, ood_test
        # best_id_misclassified = final_misclassified
#        visualize_misclassification(best_id_misclassified, f'./tu/best_id_misclassification.png', 
#                                    title='Best ID Validation Misclassification Distribution')
    # ood val
    if ood_val > best_ood_val:
        best_ood_val, best_ood_ood_test = ood_val, ood_test
        # best_ood_misclassified = final_misclassified
#        visualize_misclassification(best_ood_misclassified, f'./tu/best_ood_misclassification.png', 
#                                    title='Best OOD Validation Misclassification Distribution')
    # type = 'id_test'
    # visualize_misclassification3(best_id_misclassified, type, f'./supervisedcon/tu_best/final_id_misclassification_GOODWebKB_covariate_{config.model.model_name}.png', title='Final ID Test Misclassification Distribution')
    # visualize_misclassification(best_id_misclassified, type, f'./supervisedcon/tu_best1/{config.model.model_name}_best_id_misclassification.png', 
    #                             title='Final Misclassification Distribution')
    # visualize_misclassification(best_id_misclassified, type, f'./supervisedcon/tu_best/{config.model.model_name}_best_id_misclassification.png', 
    #                             title='Final Misclassification Distribution')
    # torch.save(best_id_misclassified, f'./supervisedcon/tu_data/DropNode_best_id_misclassified_{config.model.model_name}.pt')
    
    # type = 'ood_test'
    # visualize_misclassification3(best_ood_misclassified, type, f'./supervisedcon/tu_best/final_ood_misclassification_GOODWebKB_covariate_{config.model.model_name}.png', title='Final OOD Test Misclassification Distribution')
    # visualize_misclassification(best_ood_misclassified, type, f'./supervisedcon/tu_best1/{config.model.model_name}_best_ood_misclassification.png', 
    #                             title='Final Misclassification Distribution')
    # visualize_misclassification(best_ood_misclassified, type, f'./supervisedcon/tu_best/{config.model.model_name}_best_ood_misclassification.png', 
    #                             title='Final Misclassification Distribution')
    # torch.save(best_ood_misclassified, f'./supervisedcon/tu_data/DropNode_best_ood_misclassified_{config.model.model_name}.pt')
#    create_video_from_frames('./video/tu/', './video/misclassification_visualization.mp4')
    ################################################################################
    # id_val_loss = calculate_validation_loss(model, loader, criterion, config, device, 'id_val')
    # ood_val_loss = calculate_validation_loss(model, loader, criterion, config, device, 'val')
    # id_val_loss_list.append(id_val_loss)
    # ood_val_loss_list.append(ood_val_loss)    
    # id_test_loss = calculate_test_loss(model, loader, criterion, config, device, 'id_test')
    # ood_test_loss = calculate_test_loss(model, loader, criterion, config, device, 'test')
    # id_test_loss_list.append(id_test_loss)
    # ood_test_loss_list.append(ood_test_loss)    
    # save checkpoint
    if config.train.save_checkpoint:
        save_ckpts(model, config)
    print(f"\nFinal results: id-id: {best_id_id_test:.4f}, id-ood: {best_id_ood_test:.4f}, ood-ood: {best_ood_ood_test:.4f}")
    write_res_in_log([best_id_id_test, best_id_ood_test, best_ood_ood_test], config) # write results in /storage/log 
    
    # loss_data = {
    # 'model': config.model.model_name,
    # 'train_loss': train_loss_list,
    # 'id_val_loss': id_val_loss_list,
    # 'ood_val_loss': ood_val_loss_list,
    # 'id_test_loss': id_test_loss_list,
    # 'ood_test_loss': ood_test_loss_list
    # }

    # loss_save_path = f'./loss_data1/DropNode_{config.model.model_name}_loss.json'
    # os.makedirs('./loss_data', exist_ok=True)
    # with open(loss_save_path, 'w') as f:
    #     json.dump(loss_data, f)
    # loss_save_path = f'./loss_data1/{config.model.model_name}_loss.json'
    # os.makedirs('./loss_data', exist_ok=True)
    # with open(loss_save_path, 'w') as f:
    #     json.dump(loss_data, f)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"Total running time: {elapsed_time:.2f} seconds")  # 打印运行时间
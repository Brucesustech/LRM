from easyGOOD.data import create_dataloader, load_dataset
from easyGOOD.utils.initial import reset_random_seed
from easyGOOD.utils.config_reader import config_summoner
from easyGOOD.utils.args import args_parser
from eval_utils import nan2zero_get_mask, evaluate_all_with_scores
from models import GCN, GAT, GCN_Encoder, GRACE, load_sup_model
from writer import write_all_in_pic, write_res_in_log
from collections import defaultdict, Counter
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import matplotlib
import gc
import os
import json
from rich.progress import track
import time
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

    
def train_linear_head(model, config, ood_train=False):
    """Train linear classifier head"""
    model.reset_classifier()
    model.classifier.train()
    for e in track(range(config.train.linear_head_epochs)):
        classifier_optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=config.train.lr, weight_decay=1e-4)
        for data in loader['train']:
            model.classifier.train()
            data = data.to(device)
            node_norm = data.get('node_norm') if config.model.model_level == 'node' else None
            edge_weight = data.get('edge_norm') if config.model.model_level == 'node' else None
            if ood_train:
                assert config.dataset.ood_train_set
                mask, targets = nan2zero_get_mask(data, 'ood_train', config) 
            else:
                mask, targets = nan2zero_get_mask(data, 'train', config)
            # Handle special preprocessing for certain models
            if config.model.model_name in ['Mixup', 'SRGNN']:
                data, targets, mask, node_norm = model.input_preprocess(data, targets,mask,node_norm,training=True,device=device)
            preds = model(data.x, data.edge_index, edge_weight=edge_weight, frozen=True, data=data)
            preds = model.output_postprocess(preds)
            # Calculate loss based on model type
            if config.model.model_name != 'EERM':
                if config.model.model_name == 'Mixup':
                    loss = model.loss_calculate(preds, targets, mask, node_norm, config)
                else:
                    loss = criterion(preds, targets) * mask
                    loss = loss * node_norm * mask.sum() if (config.model.model_level == 'node' and not config.dataset.inductive) else loss
            else:
                loss = preds
            # Apply model-specific post-processing
            if config.model.model_name in ['Coral', 'SRGNN']:
               loss = model.loss_postprocess(loss, data, mask, config, device)
            else:
               loss = model.loss_postprocess(loss, data, mask, config)

            loss.backward()
            classifier_optimizer.step()
            classifier_optimizer.zero_grad()

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
        

        # plt.savefig(save_path_base)
        # plt.close()
        
 
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
    # Setup
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = args_parser()
    config = config_summoner(args)
    print(config)
    reset_random_seed(config)
    # Load dataset
    dataset = load_dataset(config.dataset.dataset_name, config)
    loader = create_dataloader(dataset, config)
   
    train_mask = dataset.data.train_mask
    train_labels = dataset.data.y[train_mask].long().squeeze()  
    num_classes = dataset.num_classes
    total_samples = train_labels.shape[0]
    # training process
    if dataset.num_classes > 2: # multi-label classification
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    else: # binary classification
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    # Initialize model
    model = load_sup_model(config.model.model_name, config).to(device)
    print(model)
    # model_size = count_parameters(model)
    # print(f"  Model size (MB): {model_size['total'] * 4 / (1024 * 1024):.2f}")    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.train.mile_stones,
                                                        gamma=0.1)

    # Training loop
    ebar = tqdm(range(1, config.train.max_epoch+1))
    best_id_val, best_id_id_test, best_id_ood_test, best_ood_val, best_ood_ood_test = 0, 0, 0, 0, 0

    best_id_misclassified = None
    best_ood_misclassified = None
    frame_count = 0  
    train_list, id_val_list, id_test_list, ood_val_list, ood_test_list = [], [], [], [], []
    # id_val_loss = 0
    # ood_val_loss = 0
    # train_loss_list = []
    # id_val_loss_list = []
    # ood_val_loss_list = []
    for e in ebar:
        epoch_loss = 0
        epoch_node_cnt = 0
        for data in loader['train']:
            optimizer.zero_grad()
            model.train()
            data = data.to(device)
            node_norm = data.get('node_norm') if config.model.model_level == 'node' else None
            edge_weight = data.get('edge_norm') if config.model.model_level == 'node' else None

            mask, targets = nan2zero_get_mask(data, 'train', config)
            # Handle special preprocessing for certain models
            if config.model.model_name in ['Mixup', 'SRGNN']:
                data, targets, mask, node_norm = model.input_preprocess(data, targets,mask,node_norm,training=True,device=device)
            preds = model(x=data.x, edge_index=data.edge_index, edge_weight=edge_weight, data=data)
            preds = model.output_postprocess(preds)
            # Calculate loss based on model type
            if config.model.model_name != 'EERM':
                if config.model.model_name == 'Mixup':
                    loss = model.loss_calculate(preds, targets, mask, node_norm, config)
                else:
                    loss = criterion(preds, targets) * mask
                    loss = loss * node_norm * mask.sum() if (config.model.model_level == 'node' and not config.dataset.inductive) else loss
            else:
                loss = preds
            # Apply model-specific post-processing    
            if config.model.model_name in ['Coral', 'SRGNN']:
               loss = model.loss_postprocess(loss, data, mask, config, device)
            else:
               loss = model.loss_postprocess(loss, data, mask, config)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item() * mask.sum().item()
            epoch_node_cnt += mask.sum().item()
        # avg_train_loss = epoch_loss / epoch_node_cnt
        # if e > 1:
        #     train_loss_list.append(avg_train_loss)
        scheduler.step()
        
        # train the linear classifier and eval
        if (e % config.train.eval_step == 0) or (e == config.train.max_epoch):
            train_linear_head(model, config)
            # Calculate misclassification statistics
            # curr_misclassified = count_misclassified_samples(model, loader, criterion, config, device)
            # eval
            # visualize_misclassification2(curr_misclassified, f'./supervised/tu_data/frame_{frame_count:04d}.png', title=f'Epoch {e} Misclassification Distribution')
            # torch.save(curr_misclassified, f'./supervised/tu_data/misclassified_{frame_count:04d}.pt')
            frame_count += 1
            # Evaluate model performance
            train_acc, id_val, id_test, ood_val, ood_test = evaluate_all_with_scores(model, loader, criterion, config, device)
            train_list.append(train_acc)
            id_val_list.append(id_val)
            id_test_list.append(id_test)
            ood_val_list.append(ood_val)
            ood_test_list.append(ood_test)

            # id_val_loss = calculate_validation_loss(model, loader, criterion, config, device, 'id_val')
            # ood_val_loss = calculate_validation_loss(model, loader, criterion, config, device, 'val')
            # id_val_loss_list.append(id_val_loss)
            # ood_val_loss_list.append(ood_val_loss)            
            # id val
            if id_val > best_id_val:
                best_id_val, best_id_id_test, best_id_ood_test = id_val, id_test, ood_test
                # best_id_misclassified = curr_misclassified
            # ood val
            if ood_val > best_ood_val:
                best_ood_val, best_ood_ood_test = ood_val, ood_test
                # best_ood_misclassified = curr_misclassified
            # Update progress bar    
            ebar.set_postfix({'Train Loss': epoch_loss/epoch_node_cnt, 'train acc': train_acc,
                                'id val': id_val, 'id test': id_test,
                                'ood val': ood_val, 'ood test': ood_test})
            accs = [train_acc, id_val, id_test, ood_val, ood_test]
        # write_all_in_pic(current_time, config, accs, e) # the information of tensorboard is recorded in /storage/tensorboard 
    # type = 'id_test'
    # visualize_misclassification3(best_id_misclassified, type, f'./supervised/tu_best/final_id_misclassification.png', title='Final ID Test Misclassification Distribution')
    # visualize_misclassification(best_id_misclassified, type, f'./supervised/tu_best/{config.model.model_name}_best_id_misclassification.png', 
    #                             title='Final Misclassification Distribution')
    # torch.save(best_id_misclassified, './supervised/tu_data/best_id_misclassified.pt')
    
    # type = 'ood_test'
    # visualize_misclassification3(best_ood_misclassified, type, f'./supervised/tu_best/final_ood_misclassification.png', title='Final OOD Test Misclassification Distribution')
    # visualize_misclassification(best_ood_misclassified, type, f'./supervised/tu_best/{config.model.model_name}_best_ood_misclassification.png', 
    #                             title='Final Misclassification Distribution')
    # torch.save(best_ood_misclassified, './supervised/tu_data/best_ood_misclassified.pt') 
    # Report final results 
    print(f"\nFinal results: id-id: {best_id_id_test:.4f}, id-ood: {best_id_ood_test:.4f}, ood-ood: {best_ood_ood_test:.4f}")
    write_res_in_log([best_id_id_test, best_id_ood_test, best_ood_ood_test], config) # write results in /storage/log 
    # loss_data = {
    # 'model': config.model.model_name,
    # 'train_loss': train_loss_list,
    # 'id_val_loss': id_val_loss_list,
    # 'ood_val_loss': ood_val_loss_list
    # }
    # loss_save_path = f'./suploss_data/{config.model.model_name}_loss.json'
    # os.makedirs('./loss_data', exist_ok=True)
    # with open(loss_save_path, 'w') as f:
    #     json.dump(loss_data, f)
    # loss_save_path = f'./suploss_data/{config.model.model_name}_loss.json'
    # os.makedirs('./loss_data', exist_ok=True)
    # with open(loss_save_path, 'w') as f:
    #     json.dump(loss_data, f)
    end_time = time.time() 
    elapsed_time = end_time - start_time  
    print(f"Total running time: {elapsed_time:.2f} seconds")  
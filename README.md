# Beyond Random Masking: Label-Ratio Node Augmentation for Invariant Learning on Out-of-Distribution Graphs

This repository contains the implementation for "Beyond Random Masking: Label-Ratio Node Augmentation for Invariant Learning on Out-of-Distribution Graphs". 

# Setup

You can set up the environment using the following commands:

## Create and activate conda environment
conda create -n graph_ood python=3.9
conda activate graph_ood

## Install PyTorch with CUDA support
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

## Install PyG
pip install torch-geometric==2.0.4
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.2+cu118.html


conda install -c conda-forge rdkit==2022.3.3



# Usage

## Supervised Training
### For covariate shift experiments
bash covariate.sh  # Runs supervised methods (e.g., Mixup)
config_dirs=(
    "configs/GOODWebKB/university/covariate"
    # "configs/GOODCora/degree/covariate"
    # "configs/GOODCora/word/covariate"
)
### For concept shift experiments
bash concept.sh  # Runs supervised methods (e.g., Mixup)
config_dirs=(
    configs/GOODWebKB/university/concept
    # "configs/GOODCora/word/concept"
    # "configs/GOODCora/degree/concept"
)

## supervised contrastive Training

### For covariate shift experiments
bash covariate1.sh  # Runs supervised contrastive methods (MARIO, GRACE, SWAV)

### For concept shift experiments
bash concept1.sh  # Runs supervised contrastive methods (MARIO, GRACE, SWAV)

## Command Line Arguments

`config_path` is an important command line argument which specify the dataset, OOD type and learning algorithm. You can specify it as:

`--config_path={dataset}/{domain}/{shift_type}/{method}.yaml`

These options are available now.
```
dataset: GOODWebKB, GOODCora-degree, GOODCora-word.
domain: according to the selected dataset(GOODWebKB: university, GOODCora: word or degree).
shift_type: concept, covariate.
contrastive method: GRACE，MARIO，SWAV.
data augmentation strategies: Drop-Feature, Shuffle-Feature, Mixup-Feature, DropNode, Ours.
supervised method: ERM, IRM, EERM, Mixup, GroupDRO, DANN, Deep CORAL, and VREx.
```

## Hyperparameters

Key hyperparameters that can be configured in the YAML files:

| Parameter | Description |
|-----------|-------------|
| `major_class_ratio` | Threshold for majority class determination |
| `tau` | Temperature parameter for contrastive learning |
| `Scaling factor` | Scaling factor for label-ratio-based node masking |
| **Training Parameters** | 
| `max_epoch` | Maximum training epochs 
| `lr` | Pretrain Learning rate 
| `linear_head_epochs` | Number of epochs for linear head training 
| `linear_head_lr` | Learning rate for linear head 
| `eval_step` | Evaluation frequency (steps) 

Modify these parameters in the corresponding YAML configuration files to tune model performance.
# Results

The scripts generate comprehensive experimental results in the following format:

## Output Files

Results are saved to the `experiment_results` directory with separate files:
- `covariate_results.txt`: Results for supervised methods on covariate shift
- `covariate1_results.txt`: Results for supervised contrastive methods on covariate shift
- `concept_results.txt`: Results for supervised methods on concept shift
- `concept1_results.txt`: Results for supervised contrastive methods on concept shift

## Result Format

Each result line follows the format:
<config_path> <ID-ID_mean> ± <ID-ID_std>, <ID-OOD_mean> ± <ID-OOD_std>, <OOD-OOD_mean> ± <OOD-OOD_std>

Where:
- **ID-ID**: Accuracy on in-distribution test set using the model checkpoint selected by in-distribution validation set
- **ID-OOD**: Accuracy on out-of-distribution test set using the model checkpoint selected by in-distribution validation set
- **OOD-OOD**: Accuracy on out-of-distribution test set using the model checkpoint selected by out-of-distribution validation set

**Note**: For reporting final performance, we use **ID-ID** as the in-distribution test accuracy and **OOD-OOD** as the out-of-distribution test accuracy.

configs/GOODCora/degree/concept/MARIO.yaml 68.896 ± 0.29753, 60.659 ± 0.0793032, 60.708 ± 0.139198

configs/GOODCora/degree/concept/GRACE.yaml 69.107 ± 0.113053, 60.943 ± 0.079, 60.975 ± 0.0870919

configs/GOODCora/degree/concept/SWAV.yaml 63.6 ± 0.298563, 54.44 ± 0.198595, 54.109 ± 0.170672
## Acknowledgments

We thank the authors of [GOOD](https://github.com/divelab/GOOD) and [MARIO](https://github.com/ZhuYun97/MARIO) for their excellent work and open-source contributions.
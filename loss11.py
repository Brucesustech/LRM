import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_loss(model_name, is_DropNode=False, loss_dir='./loss_data1'):
    prefix = 'DropNode_' if is_DropNode else ''
    file_path = os.path.join(loss_dir, f'{prefix}{model_name}_loss.json')
    if not os.path.exists(file_path):
        print(f"Missing loss file: {file_path}")
        return None
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['ood_val_loss']  # 'id_val_loss', 'ood_val_loss', 'id_test_loss','ood_test_loss'

def plot_loss_comparison(models, save_path='./loss1/loss_comparison_ood_val_loss.pdf'):
    plt.figure(figsize=(8, 6))
    colors = {
        'MARIO': 'red',
        'GRACE': 'purple',
        'SWAV': 'green'
    }

    DropNode_handles = []
    our_handles = []

    epochs = list(range(0, 110, 10))  

    all_losses = []

    for model in models:
        DropNode_loss = load_loss(model, is_DropNode=True)
        our_loss = load_loss(model, is_DropNode=False)

        if DropNode_loss:
            if len(DropNode_loss) != len(epochs) - 1:
                print(f"Warning: {model} + DropNode loss length {len(DropNode_loss)} doesn't match expected {len(epochs) - 1}.")

            DropNode_loss_plot = [None] + DropNode_loss
            line = plt.plot(epochs, DropNode_loss_plot, linestyle='--', color=colors[model], label=f'{model} + DropNode')
            DropNode_handles.append(line[0])
            all_losses += DropNode_loss
        if our_loss:
            if len(our_loss) != len(epochs) - 1:
                print(f"Warning: {model} + Ours loss length {len(our_loss)} doesn't match expected {len(epochs) - 1}.")
  
            our_loss_plot = [None] + our_loss
            line = plt.plot(epochs, our_loss_plot, linestyle='-', color=colors[model], label=f'{model} + Ours')
            our_handles.append(line[0])
            all_losses += our_loss

    plt.xlabel('Epoch', fontsize=20)
    # plt.ylabel('OOD Test Loss', fontsize=20)
    plt.ylabel('OOD Val Loss', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)


    plt.xticks(np.arange(0, 110, 10))


    # if all_losses:
    #     y_max = max(all_losses)
    #     y_limit = int(np.ceil(y_max / 10.0)) * 10
    #     plt.yticks(np.arange(0, y_limit + 10, 10))


    ax = plt.gca()
    leg1 = ax.legend(handles=DropNode_handles, loc='upper left', fontsize=16, frameon=True)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=our_handles, loc='upper right', fontsize=16, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=1200)
    print(f"Saved comparison plot to {save_path}")
    plt.close()

if __name__ == '__main__':
    plot_loss_comparison(['MARIO', 'GRACE', 'SWAV'])
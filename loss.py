import os
import json
import matplotlib.pyplot as plt

def load_loss(model_name, is_DropNode=False, loss_dir='./loss_data1'):
    prefix = 'DropNode_' if is_DropNode else ''
    file_path = os.path.join(loss_dir, f'{prefix}{model_name}_loss.json')
    if not os.path.exists(file_path):
        print(f"Missing loss file: {file_path}")
        return None
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['train_loss']  # 'train_loss'

def plot_loss_comparison(models, save_path='./loss1/loss_comparison_train_loss.pdf'):
    plt.figure(figsize=(8, 6))
    colors = {
        'MARIO': 'red',
        'GRACE': 'purple',
        'SWAV': 'green'
    }


    DropNode_handles = []
    our_handles = []

    for model in models:
        DropNode_loss = load_loss(model, is_DropNode=True)
        our_loss = load_loss(model, is_DropNode=False)

        if DropNode_loss:
            line = plt.plot(DropNode_loss, linestyle='--', color=colors[model], label=f'{model} + DropNode')
            DropNode_handles.append(line[0])
        if our_loss:
            line = plt.plot(our_loss, linestyle='-', color=colors[model], label=f'{model} + Ours')
            our_handles.append(line[0])

    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Pretrain Loss', fontsize=20)
    # plt.title('OOD Validation Loss Comparison', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True)


    plt.legend(handles=DropNode_handles, loc='upper left', fontsize=16, frameon=True)
    plt.legend(handles=our_handles, loc='upper right', fontsize=16, frameon=True)


    ax = plt.gca()
    leg1 = ax.legend(handles=DropNode_handles, loc='upper left', fontsize=16,  frameon=True)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=our_handles, loc='upper right', fontsize=16, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=1200)
    print(f"Saved comparison plot to {save_path}")
    plt.close()

if __name__ == '__main__':
    plot_loss_comparison(['MARIO', 'GRACE', 'SWAV'])
    #plot_loss_comparison(['MARIO'])
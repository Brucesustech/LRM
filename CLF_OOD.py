import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.linewidth'] = 0.8 


labels = ['MARIO', 'GRACE', 'SWAV']
# labels = ['Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Total Errors']
x = np.arange(len(labels))  

# dist1 = [80, 63, 98, 80, 0, 1561]
# dist2 = [87, 85, 64, 65, 55, 1529]
# dist3 = [88, 91, 67, 66, 31, 1487]

# dist2 = [109, 101, 63, 64, 30, 1533]
# dist3 = [97, 91, 56, 63, 30, 1497]
# dist2 = [120, 94, 83, 72, 103, 1863]
# dist3 = [113, 110, 65, 106, 79, 1754]

dist1 = [2034, 2034, 2034] #EERM

dist2 = [1970, 1936, 2173]


dist3 = [1922, 1924, 2206]

bar_width = 0.2
offsets = [-bar_width, 0, bar_width]


colors = ['#66c2a5', '#fc8d62', '#8da0cb']  
total_colors = colors.copy()


fig, ax = plt.subplots(figsize=(14, 12))


bars1 = ax.bar(x + offsets[0], dist1, width=bar_width, label='Mixup',
               color=colors[0], edgecolor='black')
bars2 = ax.bar(x + offsets[1], dist2, width=bar_width, label='DropNode',
               color=colors[1], edgecolor='black')
bars3 = ax.bar(x + offsets[2], dist3, width=bar_width, label='Ours',
               color=colors[2], edgecolor='black')


for i, bars in enumerate([bars1, bars2, bars3]):
    bar = bars[-1] 
    bar.set_color(total_colors[i])
    bar.set_edgecolor('black')
    bar.set_linewidth(1.0)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=24)


ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Number of OOD Test Misclassifications', fontsize=24)
ax.set_xlabel('Contrastive Learning Framework', fontsize=24)



fig.patch.set_facecolor('white')
ax.set_facecolor('white')
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.3)


ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color='gray', alpha=0.6)
ax.xaxis.grid(False)


ax.legend(loc='upper left', fontsize=24)


save_path = './OOD.pdf'
plt.tight_layout()
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.show()
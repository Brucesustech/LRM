import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.linewidth'] = 0.8  


labels = ['Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Total Errors']
x = np.arange(len(labels))  

# dist1 = [5, 2, 15, 11, 2, 35]
# dist2 = [2, 6, 4, 8, 5, 25]
# dist3 = [1, 5, 2, 5, 2, 15]

# dist1 = [5, 2, 15, 11, 2, 35]
# dist2 = [0, 4, 8, 10, 3, 25]
# dist3 = [1, 2, 3, 5, 1, 12]

dist1 = [5, 2, 15, 11, 2, 35]
dist2 = [1, 3, 6, 9, 4, 23]
dist3 = [3, 3, 3, 6, 1, 16]


bar_width = 0.2
offsets = [-bar_width, 0, bar_width]


colors = ['#66c2a5', '#fc8d62', '#8da0cb']  
total_colors = colors.copy()


fig, ax = plt.subplots(figsize=(14, 12))


bars1 = ax.bar(x + offsets[0], dist1, width=bar_width, label='DANN',
               color=colors[0], edgecolor='black')
bars2 = ax.bar(x + offsets[1], dist2, width=bar_width, label='SWAV + DropNode',
               color=colors[1], edgecolor='black')
bars3 = ax.bar(x + offsets[2], dist3, width=bar_width, label='SWAV + Ours',
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
ax.set_ylabel('Number of ID Test Misclassifications', fontsize=24)
ax.set_xlabel('Predicted Label', fontsize=24)



fig.patch.set_facecolor('white')
ax.set_facecolor('white')
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.3)


ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color='gray', alpha=0.6)
ax.xaxis.grid(False)


ax.legend(loc='upper left', fontsize=24)


save_path = './SWAV-ID.pdf'
plt.tight_layout()
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.show()
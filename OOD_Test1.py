import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.linewidth'] = 0.8  


labels = ['Label 19', 'Label 57', 'Label 7', 'Label 51', 'Label 22']
# labels = ['Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Total Errors']
x = np.arange(len(labels))  

# dist1 = [80, 63, 98, 80, 0, 1561]
# dist2 = [87, 85, 64, 65, 55, 1529]
# dist3 = [88, 91, 67, 66, 31, 1487]

# dist2 = [109, 101, 63, 64, 30, 1533]
# dist3 = [97, 91, 56, 63, 30, 1497]
# dist2 = [120, 94, 83, 72, 103, 1863]
# dist3 = [113, 110, 65, 106, 79, 1754]

# dist1 = [214, 84, 92, 69, 102, 2259]
# dist2 = [163, 121, 98, 93, 74, 2313]
# dist3 = [148, 96, 101, 97, 81, 2190]

# dist2 = [149, 97, 102, 85, 81, 2264]
# dist3 = [159, 101, 93, 81, 79, 2200]
# dist2 = [181, 118, 118, 122, 117, 2621]
# dist3 = [175, 104, 111, 118, 129, 2538]

# dist1 = [69, 100, 59, 54, 45, 1368]
# dist2 = [92, 70, 52, 50, 43, 1342]
# dist3 = [89, 73, 58, 47, 36, 1332]

# dist1 = [69, 100, 59, 54, 45, 1368]
# dist2 = [102, 60, 48, 48, 55, 1334]
# dist3 = [99, 68, 47, 47, 33, 1302]

# dist1 = [69, 100, 59, 54, 45, 1368]
# dist2 = [80, 73, 74, 84, 87, 1561]
# dist3 = [127, 83, 69, 69, 52, 1570]

# dist1 = [85, 102, 145, 103, 62, 2034]
# dist2 = [98, 94, 89, 79, 66, 1970]
# dist3 = [87, 89, 103, 71, 67, 1922]

# dist1 = [85, 102, 145, 103, 62, 2034]
# dist2 = [97, 82, 90, 80, 58, 1936]
# dist3 = [104, 105, 86, 70, 74, 1924]

# dist1 = [85, 102, 145, 103, 62, 2034]
# dist2 = [158, 113, 114, 73, 102, 2173]
# dist3 = [127, 120, 139, 82, 126, 2206]


dist1 = [85, 102, 145, 103, 62]


dist2 = [158, 113, 114, 73, 102]



dist3 = [127, 120, 139, 82, 126]

bar_width = 0.2
offsets = [-bar_width, 0, bar_width]


colors = ['#66c2a5', '#fc8d62', '#8da0cb']  
total_colors = colors.copy()


fig, ax = plt.subplots(figsize=(14, 12))


bars1 = ax.bar(x + offsets[0], dist1, width=bar_width, label='Mixup',
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
ax.set_ylabel('Number of OOD Test Misclassifications', fontsize=24)
ax.set_xlabel('Predicted Label', fontsize=24)


fig.patch.set_facecolor('white')
ax.set_facecolor('white')
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.3)

ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color='gray', alpha=0.6)
ax.xaxis.grid(False)


ax.legend(loc='upper right', fontsize=24)


save_path = './SWAV-OOD.pdf'
plt.tight_layout()
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.show()
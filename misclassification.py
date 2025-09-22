import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as patches
import gc


plt.rcParams['figure.max_open_warning'] = 1
plt.rcParams['interactive'] = False
plt.close('all')

dpi = 600  


data = np.array([
    [0.0,  0.17, 0.08, 0.67, 0.08],
    [0.0,  0.0,  0.25, 0.75, 0.0],
    [0.12, 0.06, 0.0,  0.81, 0.0],
    [0.13, 0.13, 0.67, 0.0,  0.07],
    [0.0,  0.08, 0.42, 0.50, 0.0]
])

plt.figure(figsize=(10, 8), dpi=dpi)

ax = sns.heatmap(
    data,
    annot=True,
    cmap="Blues",
    fmt=".2f",
    cbar_kws={'label': 'Proportion'},
    square=True,
    annot_kws={"fontsize": 16}  
)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)              
cbar.ax.set_ylabel("Proportion", fontsize=14)  
rows, cols = data.shape

border = patches.Rectangle(
    (0, 0), cols, rows, 
    linewidth=2,  
    edgecolor='black',  
    facecolor='none'  
)
ax.add_patch(border)


plt.title("Final OOD Test Misclassification Distribution", fontsize=20)  
plt.xlabel("Predicted label", fontsize=16)  
plt.ylabel("True label", fontsize=16)  


plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)  


plt.tight_layout()


plt.savefig("SWAV_misclassification_distribution.pdf", format='pdf', dpi=dpi, bbox_inches='tight')


plt.close('all')
gc.collect()
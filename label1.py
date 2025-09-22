import matplotlib.pyplot as plt
import os


plt.style.use('seaborn-muted') #seaborn-muted,seaborn-darkgrid

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 17,
    'axes.labelsize': 15,
    'legend.fontsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.markersize': 8,
    'lines.linewidth': 2.5
})


x_labels = ['0.05', '0.1', '0.15', '0.25']
x = range(len(x_labels))

set = [58.20, 60.17, 61.20, 62.13]

# set = [67.72, 69.22, 70.99, 73.61]
# id_val_set = [38.01, 39.91, 41.59, 40.37]
# id_test_set = [58.20, 60.17, 61.20, 62.13]
# ood_val_set = [28.99, 31.43, 39.66, 36.43]
# ood_test_set = [81, 87, 71, 64, 68]

# set = [81, 87, 166, 215, 68]
# id_val_set = [81, 87, 149, 190, 68]
# id_test_set = [81, 87, 130, 152, 68]
# ood_val_set = [81, 87, 113, 116, 68]
# ood_test_set = [81, 87, 71, 64, 68]


# colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e']
colors = ['#2ca02c']
# colors = ['#9467bd']

plt.figure(figsize=(10, 6.5))


plt.plot(x, set, marker='o', label='concept', color=colors[0])
# plt.plot(x, id_val_set, marker='o', label='OOD', color=colors[1])
# plt.plot(x, id_test_set, marker='o', label='covariate', color=colors[1])
# plt.plot(x, ood_val_set, marker='o', label='0.15', color=colors[3])
# plt.plot(x, ood_test_set, marker='o', label='0.25', color=colors[4])


for i, data_set in enumerate([set]):
    for j, val in enumerate(data_set):
        plt.annotate(f"{val}%", 
                    (x[j], val), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=15,
                    color=colors[i])
        

plt.xlabel('Mask Probability', fontsize=15, labelpad=10)
plt.ylabel('OOD Test Accuracy', fontsize=15, labelpad=10)


plt.xticks(x, x_labels)

legend = plt.legend(loc='best', frameon=True)
legend.get_frame().set_edgecolor('black')  


plt.grid(True, linestyle='--', alpha=0.6)


plt.tight_layout()


output_path = './covariate-OOD.pdf'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, format='pdf')


# plt.show()
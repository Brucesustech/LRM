import matplotlib.pyplot as plt
import os


plt.style.use('seaborn-muted') #seaborn-muted,seaborn-darkgrid


plt.rcParams.update({
    'font.size': 20,           # 进一步调大到20
    'axes.titlesize': 28,      # 进一步调大到28
    'axes.labelsize': 26,      # 进一步调大到26
    'legend.fontsize': 22,     # 进一步调大到22
    'xtick.labelsize': 20,     # 进一步调大到20
    'ytick.labelsize': 20,     # 进一步调大到20
    'lines.markersize': 16,    # 进一步调大到16
    'lines.linewidth': 5.0     # 进一步调大到5.0
})


x_labels = ['Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4']
x = range(len(x_labels))


set = [81, 87, 166, 215, 68]
# id_val_set = [81, 87, 147, 187, 68]
# id_test_set = [81, 87, 133, 146, 68]
# ood_val_set = [81, 87, 101, 120, 68]
# ood_test_set = [81, 87, 71, 64, 68]

# set = [81, 87, 166, 215, 68]
# id_val_set = [81, 87, 149, 190, 68]
# id_test_set = [81, 87, 130, 152, 68]
# ood_val_set = [81, 87, 113, 116, 68]
# ood_test_set = [81, 87, 71, 64, 68]


# colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e']
colors = ['#ff7f0e']

plt.figure(figsize=(10, 8.5))


plt.plot(x, set, marker='o', label='Node set', color=colors[0])
# plt.plot(x, id_val_set, marker='o', label='0.05', color=colors[1])
# plt.plot(x, id_test_set, marker='o', label='0.1', color=colors[2])
# plt.plot(x, ood_val_set, marker='o', label='0.15', color=colors[3])
# plt.plot(x, ood_test_set, marker='o', label='0.25', color=colors[4])


# for i, data_set in enumerate([set, id_val_set, id_test_set, ood_val_set, ood_test_set]):
#     for j, val in enumerate(data_set):
#         plt.annotate(str(val), 
#                     (x[j], val), 
#                     textcoords="offset points", 
#                     xytext=(0, 10), 
#                     ha='center',
#                     fontsize=15,
#                     color=colors[i])
        

plt.xlabel('Label', labelpad=10)
plt.ylabel('Sample Size', labelpad=10)


plt.xticks(x, x_labels)


legend = plt.legend(loc='best', frameon=True)
legend.get_frame().set_edgecolor('black')  


plt.grid(True, linestyle='--', alpha=0.6)


plt.tight_layout()


output_path = './tu1.pdf'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, format='pdf')


# plt.show()
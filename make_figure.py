import matplotlib.pyplot as plt
import numpy as np

labels = ['ACER', 'Acc.', 'AUC']
paper = [3.6, 96.4, 99.1]
mine = [0.0, 100.0, 100.0]

x = np.arange(len(labels))
w = 0.35
fig, ax = plt.subplots(figsize=(3.5, 2.2))
ax.bar(x - w/2, paper, w, label='Paper', color='#0072B2', hatch='///', edgecolor='black')
ax.bar(x + w/2, mine, w, label='Mine (1 seed, 2 epochs)', color='#E69F00', hatch='...', edgecolor='black')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Value (%)', fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig("fig1.pdf", bbox_inches='tight')
print("saved fig1.pdf")

import os
import sys
sys.path.insert(0, '/Users/saibhuvan/Documents/Research/imple')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

os.chdir('/Users/saibhuvan/Documents/Research/imple')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("Generating additional publication figures...")

# 1. Plasticity vs Stability Trade-off Bar Chart
df_tradeoff = pd.read_csv("results/tables/master_iot_edge_tradeoff_metrics.csv")

# Filter to key comparison strategies
strats = ['Full Retraining / Full Replay', 'Static Replay (100/cls)', 'Dynamic Volume Replay']
strat_labels = ['Full Retraining', 'Static Replay (100/cls)', 'Dynamic Replay (Proposed)']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

attacks = ['mitm', 'ransomware', 'xss', 'backdoor']
attack_names = ['MITM (Low Vol)', 'Ransomware (Med Vol)', 'XSS (High Vol)', 'Backdoor (High Vol)']
x = np.arange(len(attacks))
width = 0.25

colors = ['#E74C3C', '#F39C12', '#2ECC71']

# Ax1: Plasticity (New Attack F1)
for i, s in enumerate(strats):
    sub = df_tradeoff[df_tradeoff['strategy'] == s]
    vals = [sub[sub['unseen_attack'] == a]['new_attack_f1'].values[0] for a in attacks]
    rects = ax1.bar(x + (i - 1) * width, vals, width, label=strat_labels[i], color=colors[i], edgecolor='black')
    for rect in rects:
        h = rect.get_height()
        ax1.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

ax1.set_title("Plasticity: Zero-Day Novel Attack Detection (F1 Score)", fontsize=13, fontweight='bold')
ax1.set_ylabel("Novel Attack F1 Score", fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(attack_names, fontsize=10)
ax1.set_ylim(0, 1.15)
ax1.legend(loc='lower left', frameon=True, fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Ax2: Stability (Known Classes Macro-F1 After Adaptation)
for i, s in enumerate(strats):
    sub = df_tradeoff[df_tradeoff['strategy'] == s]
    vals = [sub[sub['unseen_attack'] == a]['known_macro_f1_after'].values[0] for a in attacks]
    rects = ax2.bar(x + (i - 1) * width, vals, width, label=strat_labels[i], color=colors[i], edgecolor='black')
    for rect in rects:
        h = rect.get_height()
        ax2.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2.set_title("Stability: Historical Knowledge Retention (Macro-F1)", fontsize=13, fontweight='bold')
ax2.set_ylabel("Historical Classes Macro-F1", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(attack_names, fontsize=10)
ax2.set_ylim(0.70, 0.95)
ax2.legend(loc='lower left', frameon=True, fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
out_tradeoff = "results/figures/master/publication_plasticity_vs_stability.png"
plt.savefig(out_tradeoff, dpi=300)
plt.savefig("results/figures/publication/fig_plasticity_vs_stability.png", dpi=300)
plt.close()
print(f"Saved: {out_tradeoff}")

# 2. Replay Memory Footprint & Compression Efficiency
fig, ax = plt.subplots(figsize=(10, 5))
mem_full = [df_tradeoff[(df_tradeoff['strategy']=='Full Retraining / Full Replay') & (df_tradeoff['unseen_attack']==a)]['replay_samples'].values[0] for a in attacks]
mem_static = [df_tradeoff[(df_tradeoff['strategy']=='Static Replay (100/cls)') & (df_tradeoff['unseen_attack']==a)]['replay_samples'].values[0] for a in attacks]
mem_dynamic = [df_tradeoff[(df_tradeoff['strategy']=='Dynamic Volume Replay') & (df_tradeoff['unseen_attack']==a)]['replay_samples'].values[0] for a in attacks]

r1 = ax.bar(x - width, mem_full, width, label='Full Retraining (Entire Corpus)', color='#E74C3C', edgecolor='black')
r2 = ax.bar(x, mem_dynamic, width, label='Dynamic Replay Buffer (Proposed)', color='#2ECC71', edgecolor='black')
r3 = ax.bar(x + width, mem_static, width, label='Static Replay (Fixed 100/cls)', color='#F39C12', edgecolor='black')

ax.set_yscale('log')
ax.set_title("Replay Memory Footprint Comparison (Log Scale)", fontsize=13, fontweight='bold')
ax.set_ylabel("Stored Packets / Samples (Log Scale)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(attack_names, fontsize=10)
ax.legend(frameon=True, fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for rect in r2:
    h = rect.get_height()
    red = 100 * (1 - (h / 120000))
    ax.annotate(f"{int(h):,} slots\n(-{red:.1f}%)", xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1E8449')

plt.tight_layout()
out_mem = "results/figures/master/publication_replay_memory_efficiency.png"
plt.savefig(out_mem, dpi=300)
plt.savefig("results/figures/publication/fig_memory_efficiency.png", dpi=300)
plt.close()
print(f"Saved: {out_mem}")

print("All additional publication figures successfully created!")

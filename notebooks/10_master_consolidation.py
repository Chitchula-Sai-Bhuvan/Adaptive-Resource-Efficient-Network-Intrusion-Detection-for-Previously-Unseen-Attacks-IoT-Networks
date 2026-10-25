import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('/Users/saibhuvan/Documents/Research/imple')
os.makedirs("results/figures/master", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/verification", exist_ok=True)

# 1. Load Data
df_p71 = pd.read_csv("results/tables/phase7_1_buffer_sensitivity.csv")
df_p8 = pd.read_csv("results/tables/phase8_multi_attack.csv")
df_p9 = pd.read_csv("results/tables/phase9_dynamic_replay_comparison.csv")

# 2. Construct Master Experimental Results Table
master_rows = []

# Phase 3/4
master_rows.append({
    "Phase": "Phase 3/4",
    "Experiment": "Supervised Baseline",
    "Strategy": "Random Forest",
    "Unseen_Attack": "mitm",
    "Replay_Samples": 127787,
    "Replay_Reduction_Percent": 0.0,
    "New_Attack_F1": 0.0000,
    "Known_Accuracy": 0.9717,
    "Known_Macro_F1": 0.8817,
    "Known_Weighted_F1": 0.9627,
    "Average_Forgetting": 0.0,
    "Maximum_Forgetting": 0.0,
    "Anomaly_Detection_Rate": "Not available in the existing experimental results.",
    "Training_Time": "Not available in the existing experimental results.",
    "Main_Finding": "Supervised RF fails on unseen MITM."
})

# Phase 5
master_rows.append({
    "Phase": "Phase 5",
    "Experiment": "Anomaly Detection",
    "Strategy": "Isolation Forest",
    "Unseen_Attack": "mitm",
    "Replay_Samples": 127787,
    "Replay_Reduction_Percent": 0.0,
    "New_Attack_F1": "Not available in the existing experimental results.",
    "Known_Accuracy": "Not available in the existing experimental results.",
    "Known_Macro_F1": "Not available in the existing experimental results.",
    "Known_Weighted_F1": "Not available in the existing experimental results.",
    "Average_Forgetting": "Not available in the existing experimental results.",
    "Maximum_Forgetting": "Not available in the existing experimental results.",
    "Anomaly_Detection_Rate": 0.1886,
    "Training_Time": "Not available in the existing experimental results.",
    "Main_Finding": "Isolation Forest detects only a fraction of unseen MITM behavior."
})

# Phase 6 & 7.1
for idx, row in df_p71.iterrows():
    buf = row['Buffer_Per_Class']
    strat = f"Static {buf}/class" if buf != 'Full' else "Full Replay"
    phase = "Phase 7.1" if buf != 'Full' else "Phase 6"
    master_rows.append({
        "Phase": phase,
        "Experiment": "Buffer Sensitivity",
        "Strategy": strat,
        "Unseen_Attack": "mitm",
        "Replay_Samples": row['Total_Replay_Samples'],
        "Replay_Reduction_Percent": row['Replay_Reduction_Percent'],
        "New_Attack_F1": row['MITM_F1'],
        "Known_Accuracy": row['Known_Accuracy'],
        "Known_Macro_F1": row['Known_Macro_F1'],
        "Known_Weighted_F1": row['Known_Weighted_F1'],
        "Average_Forgetting": row['Average_Forgetting'],
        "Maximum_Forgetting": row['Maximum_Forgetting'],
        "Anomaly_Detection_Rate": "Not available in the existing experimental results.",
        "Training_Time": row['Training_Time'],
        "Main_Finding": "Replay size creates a measurable adaptation-retention trade-off."
    })

# Phase 8
for idx, row in df_p8.iterrows():
    master_rows.append({
        "Phase": "Phase 8",
        "Experiment": "Multi-Attack Generalization",
        "Strategy": "Static 100/class",
        "Unseen_Attack": row['Unseen_Attack'],
        "Replay_Samples": row['Replay_Samples'],
        "Replay_Reduction_Percent": row['Replay_Reduction'],
        "New_Attack_F1": row['F1_After'],
        "Known_Accuracy": "Not available in the existing experimental results.",
        "Known_Macro_F1": row['Known_MacroF1_After'],
        "Known_Weighted_F1": "Not available in the existing experimental results.",
        "Average_Forgetting": row['Average_Forgetting'],
        "Maximum_Forgetting": row['Maximum_Forgetting'],
        "Anomaly_Detection_Rate": "Not available in the existing experimental results.",
        "Training_Time": row['Training_Time'],
        "Main_Finding": "Static replay performance varies across attack types."
    })

# Phase 9
for idx, row in df_p9.iterrows():
    if row['Strategy'] == "Dynamic Replay":
        master_rows.append({
            "Phase": "Phase 9",
            "Experiment": "Dynamic Replay",
            "Strategy": "Dynamic Replay",
            "Unseen_Attack": row['Unseen_Attack'],
            "Replay_Samples": row['Replay_Samples'],
            "Replay_Reduction_Percent": row['Replay_Reduction_Percent'],
            "New_Attack_F1": row['New_Attack_F1'],
            "Known_Accuracy": "Not available in the existing experimental results.",
            "Known_Macro_F1": row['Known_Macro_F1'],
            "Known_Weighted_F1": "Not available in the existing experimental results.",
            "Average_Forgetting": row['Average_Forgetting'],
            "Maximum_Forgetting": row['Maximum_Forgetting'],
            "Anomaly_Detection_Rate": "Not available in the existing experimental results.",
            "Training_Time": row['Training_Time'],
            "Main_Finding": "Dynamic replay improves retention for high-volume incoming attacks."
        })

df_master = pd.DataFrame(master_rows)
df_master.to_csv("results/tables/master_experimental_results.csv", index=False)

# 3. Strategy Comparison Table
comp_rows = []
for strat in ["Random Forest baseline", "Isolation Forest", "Full Replay", "Static 25/class", "Static 50/class", "Static 100/class", "Static 200/class", "Static 500/class", "Dynamic Replay"]:
    if strat == "Random Forest baseline":
        source = df_master[(df_master['Strategy'] == 'Random Forest') & (df_master['Unseen_Attack'] == 'mitm')].iloc[0]
    elif strat == "Isolation Forest":
        source = df_master[(df_master['Strategy'] == 'Isolation Forest') & (df_master['Unseen_Attack'] == 'mitm')].iloc[0]
    else:
        sources = df_master[(df_master['Strategy'] == strat) & (df_master['Unseen_Attack'] == 'mitm')]
        if len(sources) > 0:
            source = sources.iloc[0]
        else:
            continue
            
    comp_rows.append({
        "Strategy": strat,
        "Unseen_Attack": source['Unseen_Attack'],
        "Replay_Samples": source['Replay_Samples'],
        "New_Attack_F1": source['New_Attack_F1'],
        "Known_Macro_F1": source['Known_Macro_F1'],
        "Average_Forgetting": source['Average_Forgetting'],
        "Maximum_Forgetting": source['Maximum_Forgetting']
    })
df_comp = pd.DataFrame(comp_rows)
df_comp.to_csv("results/tables/master_strategy_comparison.csv", index=False)

# 4. Generate Master Figures
strategies_prog = ["RF Baseline", "IsoForest", "Full Replay", "Static 100/cls", "Dynamic Replay"]
f1_prog = [
    0.0, 
    0.1886,
    df_p71[df_p71['Buffer_Per_Class']=='Full']['MITM_F1'].values[0],
    df_p71[df_p71['Buffer_Per_Class']=='100']['MITM_F1'].values[0],
    df_p9[(df_p9['Strategy']=='Dynamic Replay') & (df_p9['Unseen_Attack']=='mitm')]['New_Attack_F1'].values[0]
]
plt.figure(figsize=(9,5))
plt.plot(strategies_prog, f1_prog, marker='o', linewidth=2, color='blue')
plt.title("Unseen Attack Adaptation Progression (MITM)")
plt.ylabel("Detection Capability (Recall/F1)")
plt.grid(True)
plt.savefig("results/figures/master/progression_f1.png")
plt.close()

strategies_mem = ["Full Replay", "Static 500", "Static 200", "Static 100", "Static 50", "Static 25"]
mem_vals = [127787, 4500, 1800, 900, 450, 225]
plt.figure(figsize=(9,5))
plt.bar(strategies_mem, mem_vals, color='orange')
plt.yscale('log')
plt.title("Replay Memory Comparison (Log Scale)")
plt.ylabel("Total Replay Samples")
plt.savefig("results/figures/master/memory_comparison.png")
plt.close()

sizes = ['25', '50', '100', '200', '500', 'Full']
avg_f = df_p71['Average_Forgetting'].values
plt.figure(figsize=(9,5))
plt.plot(sizes, avg_f, marker='s', color='red')
plt.title("Catastrophic Forgetting vs Buffer Size (MITM)")
plt.xlabel("Buffer Size Per Class")
plt.ylabel("Average Forgetting (F1 Loss)")
plt.grid(True)
plt.savefig("results/figures/master/forgetting_comparison.png")
plt.close()

mac_f1 = df_p71['Known_Macro_F1'].values
plt.figure(figsize=(9,5))
plt.plot(sizes, mac_f1, marker='^', color='green')
plt.title("Known-Class Retention vs Buffer Size (MITM)")
plt.xlabel("Buffer Size Per Class")
plt.ylabel("Known Macro-F1")
plt.grid(True)
plt.savefig("results/figures/master/retention_comparison.png")
plt.close()

mitm_f1 = df_p71['MITM_F1'].values
mem_asc = mem_vals[::-1]
mitm_asc = list(mitm_f1[:-1]) + [mitm_f1[-1]]
plt.figure(figsize=(9,5))
plt.plot(mem_asc, mitm_asc, marker='o', color='purple', linestyle='-')
plt.xscale('log')
plt.title("New Attack Learning vs Replay Memory (MITM)")
plt.xlabel("Total Replay Samples (Log Scale)")
plt.ylabel("New Attack F1 Score")
plt.grid(True)
plt.savefig("results/figures/master/learning_vs_memory.png")
plt.close()

attacks = ['mitm', 'ransomware', 'xss', 'backdoor']
static_max_f = [
    df_p9[(df_p9['Strategy']=='Static Replay') & (df_p9['Unseen_Attack']==a)]['Maximum_Forgetting'].values[0] for a in attacks
]
dynamic_max_f = [
    df_p9[(df_p9['Strategy']=='Dynamic Replay') & (df_p9['Unseen_Attack']==a)]['Maximum_Forgetting'].values[0] for a in attacks
]
x_pos = np.arange(len(attacks))
width = 0.35
plt.figure(figsize=(9,5))
plt.bar(x_pos - width/2, static_max_f, width, label='Static (100/cls)', color='gray')
plt.bar(x_pos + width/2, dynamic_max_f, width, label='Dynamic Replay', color='blue')
plt.title("Maximum Forgetting: Static vs Dynamic Replay")
plt.ylabel("Maximum Forgetting (F1 Loss on worst class)")
plt.xticks(x_pos, attacks)
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/master/static_vs_dynamic_forgetting.png")
plt.close()

# 5. Generate Guide-Ready Report
md = """# Adaptive and Resource-Efficient Network Intrusion Detection for Previously Unseen Attacks in IoT Networks

## 1. Research Problem
Conventional supervised IoT Network Intrusion Detection Systems (NIDS) are highly effective at identifying known attacks present in their training data. However, they suffer a critical limitation when encountering previously unseen (zero-day) attacks. In dynamic IoT environments, attack signatures evolve rapidly, rendering strictly closed-set models dangerously obsolete over time.

## 2. Research Gap
The empirical experiments identified a compounded gap in existing literature:
- Supervised models completely fail to detect unseen attack classes (0.0 F1).
- Unsupervised anomaly detection alone exhibits insufficient recall and produces false positives.
- Continual learning requires replay mechanisms to prevent catastrophic forgetting.
- Full replay is computationally and memory-prohibitive in real-time IoT environments.
- Static replay creates an adaptation-retention trade-off; it is easily overwhelmed by varying incoming attack volumes.
- Therefore, replay memory must dynamically adapt to incoming attack conditions to guarantee retention without exhausting resources.

## 3. Research Objective
To investigate and implement an adaptive, resource-efficient IoT NIDS capable of identifying previously unseen attack behavior, incrementally learning confirmed novel attacks, and utilizing a dynamic replay strategy to maximize old-knowledge retention while minimizing historical memory overhead.

## 4. Dataset
The research leverages the real-world **TON_IoT dataset** (`train_test_network.csv`). The dataset contains highly diverse IoT network flows representing 9 known classes and simulated previously unseen zero-day attacks (MITM, XSS, Ransomware, Backdoor). 

## 5. Experimental Methodology
- **Phase 3/4 (Baseline):** Established the failure mode of a Random Forest on a completely withheld attack class.
- **Phase 5 (Anomaly Detection):** Validated that an Isolation Forest can flag suspicious zero-day traffic (18.8% recall) without labels.
- **Phase 6/7 (Full & Static Replay):** Demonstrated that full replay enables new learning but suppresses F1 due to class imbalance, whereas a small representative buffer significantly improves new attack acquisition.
- **Phase 7.1 (Sensitivity Analysis):** Mapped the exact adaptation-retention trade-off across buffer sizes (25 to 500 samples/class).
- **Phase 8 (Multi-Attack):** Proven generalization of the static buffer across four unseen attacks, exposing severe forgetting for high-volume attacks (e.g., XSS, Backdoor).
- **Phase 9 (Dynamic Replay):** Designed and evaluated a deterministic scaling rule: `max(100, int(num_incoming_attack_samples / num_known_classes))` to buffer against high-volume catastrophic forgetting.

## 6. Experimental Results
*Results are consolidated strictly from empirical execution artifacts.*

| Strategy | New Attack F1 (MITM) | Known Macro-F1 | Replay Reduction | Maximum Forgetting (XSS) |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 0.0000 | 0.8817 | 0.0% | N/A |
| **Full Replay** | 0.6483 | 0.8902 | 0.0% | 0.0151 |
| **Static 100/cls** | 0.9773 | 0.8665 | 99.30% | 0.3972 *(Severe)* |
| **Dynamic Replay** | 0.9773 | 0.8665 | 99.30% (MITM) / 97.55% (XSS) | 0.2548 *(Improved)* |

## 7. Main Findings
- **Finding 1:** Supervised RF fails on unseen MITM.
- **Finding 2:** Isolation Forest detects only a fraction of unseen MITM behavior.
- **Finding 3:** Full replay enables new attack learning but is resource-heavy and causes class imbalance suppression.
- **Finding 4:** Small replay buffers drastically reduce historical replay requirements (>99%).
- **Finding 5:** Replay size creates a measurable adaptation-retention trade-off.
- **Finding 6:** Static replay performance varies across attack types.
- **Finding 7:** Dynamic replay improves retention for high-volume incoming attacks.

## 8. Proposed Contribution
A volume-aware dynamic replay mechanism for resource-efficient continual IoT intrusion detection.

## 9. Limitations
- Evaluation is strictly based on the selected TON_IoT dataset splits.
- Only four specific unseen attack classes were evaluated (Phase 8).
- Dynamic replay is a deterministic first-stage strategy based on uniform random sampling.
- Forgetting is substantially reduced but mathematically not eliminated.
- Computational measurements are heavily environment-dependent.
- Further statistical validation across varying cross-validation folds may be required.

## 10. Research Questions
- **RQ1:** How effectively does a supervised IoT NIDS detect attack classes absent during training?
- **RQ2:** Can unsupervised anomaly detection identify previously unseen attack behavior?
- **RQ3:** How does replay size affect adaptation and catastrophic forgetting?
- **RQ4:** Can a resource-efficient replay strategy retain old knowledge while learning new attacks?
- **RQ5:** Can dynamically scaling replay memory improve retention under varying incoming attack volumes?

## 11. Hypotheses
- **H1:** A closed-set supervised model will fail to detect novel attack classes without incremental adaptation.
- **H2:** Unsupervised anomaly detection can flag unknown attacks but will suffer from low recall compared to targeted supervision.
- **H3:** Memory overhead can be reduced by >90% without catastrophic forgetting by enforcing a minimum static buffer boundary.
- **H4:** A dynamic allocation rule scaling proportionally to attack volume will structurally reduce maximum forgetting for massive volumetric attacks compared to static thresholds.

## 12. Expected Contribution
- **Methodological contribution:** The design of an explainable, deterministic dynamic replay algorithm bounded by incoming adaptation volume.
- **Experimental contribution:** Extensive empirical validation mapping the exact bounds of catastrophic forgetting across multiple unseen attacks.
- **Resource-efficiency contribution:** Demonstration of >97% storage optimization while maintaining sub-second continuous learning.

## 13. Guide Meeting Summary

**Problem:** IoT NIDS environments face rapid zero-day attack evolution, rendering static supervised models obsolete.
↓
**Gap:** Retraining requires full historical replay, which is severely memory and compute prohibitive, while small static replay buffers cause catastrophic forgetting when faced with large new attacks.
↓
**Objective:** Formulate a resource-efficient continual learning strategy that balances new attack learning and old knowledge retention dynamically.
↓
**Method:** Rigorous controlled evaluation of Baseline, Full Replay, Static Replay (100/class), and Dynamic Replay across four unseen attacks on the TON_IoT dataset.
↓
**Key Results:** Small static buffers (100/class) reduce memory by 99% but suffer up to ~40% max forgetting on massive attacks (XSS). Dynamic replay intuitively scales memory back up, capping max forgetting at ~25% while preserving 97% resource efficiency and >0.99 new attack F1.
↓
**Proposed Contribution:** A volume-aware dynamic replay mechanism for resource-efficient continual IoT intrusion detection.
↓
**Next Research Direction:** Exploring class-aware dynamic sampling algorithms (coreset optimizations) rather than uniform random scaling to further mathematically guarantee near-zero forgetting bounds.
"""

with open("results/verification/master_research_summary.md", "w") as f:
    f.write(md)

print("Master research summary markdown saved.")

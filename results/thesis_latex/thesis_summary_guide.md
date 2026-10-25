# M.Tech Thesis Integration Guide: Dynamic Continual Learning for IoT NIDS

This guide provides instructions for integrating the newly generated chapters, tables, and publication-quality figures directly into your M.Tech Thesis document (`MTech_thesis.tex`).

---

## 📁 Generated Thesis Artifacts Directory
All files are ready in:
`/Users/saibhuvan/Documents/Research/imple/results/thesis_latex/`

| File | Content |
| :--- | :--- |
| `chapter4_proposed_framework.tex` | Complete mathematical formulation, memory allocation rule, algorithmic workflow, and continual learning metrics. |
| `chapter5_results_and_discussion.tex` | Complete empirical evaluation, multi-attack comparison tables, figures inclusion code, and discussion. |

---

## 📊 Publication-Quality Figures for the Thesis
The following 300-DPI high-resolution figures have been created in `results/figures/master/` and `results/figures/publication/`:

1. **Master Executive Dashboard (Figure 1):**  
   Path: `results/figures/master/master_framework_executive_dashboard.png`  
   *A 6-panel summary displaying Novel Attack Plasticity (F1), Knowledge Retention (Macro-F1), Catastrophic Forgetting Mitigation, Replay Memory Footprint (Log Scale), Composite Edge IoT Viability, and the Pareto Frontier.*

2. **IoT Edge Architectural Viability Radar Chart (Figure 2):**  
   Path: `results/figures/master/publication_iot_edge_viability_radar.png`  
   *A 5-axis radar chart showing why Dynamic Replay outclasses Full Retraining and Static Replay in real IoT edge environments.*

3. **Volumetric Catastrophic Forgetting Mitigation (Figure 3):**  
   Path: `results/figures/master/publication_catastrophic_forgetting_mitigation.png`  
   *Grouped bar chart showing how Dynamic Replay caps maximum legacy degradation across attack volumes.*

4. **Plasticity vs Stability Dilemma Solved (Figure 4):**  
   Path: `results/figures/master/publication_plasticity_vs_stability.png`  
   *Side-by-side comparison of novel attack learning vs historical knowledge retention.*

5. **Replay Memory Footprint & Efficiency (Figure 5):**  
   Path: `results/figures/master/publication_replay_memory_efficiency.png`  
   *Log-scale comparison proving 97.5% - 99.3% memory reduction.*

---

## 🛠️ How to Include in `MTech_thesis.tex`

In your main `MTech_thesis.tex` file, simply add:

```latex
% Ensure the graphicx and booktabs packages are in your preamble
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{algorithm2e}

% In the body of your thesis:
\input{results/thesis_latex/chapter4_proposed_framework.tex}
\input{results/thesis_latex/chapter5_results_and_discussion.tex}
```

Or copy-paste the sections directly into your respective chapter files!

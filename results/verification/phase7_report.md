# Phase 7 Report: Resource-Efficient Replay Experiment

==================================================
1. Research question
==================================================
Can a small memory/replay buffer achieve effective adaptation to a previously unseen attack while substantially reducing the amount of historical data required for retraining? This evaluates the fundamental trade-offs between New-attack learning, Old-knowledge retention, Memory/replay size, and Computational resource cost.

==================================================
2. Experimental setup
==================================================
Three adaptation strategies were evaluated across a static Random Forest model architecture (n_estimators=100, max_depth=20, random_state=42). All strategies used the strictly controlled 20% MITM subset (207 samples) for adaptation, and evaluated against the identical untouched 80% MITM set (832 samples) and original Known test set.

==================================================
3. Full replay baseline
==================================================
Strategy A utilizes 100% of the old training data (127787 samples). It achieves maximal retention (nearly zero forgetting) but bounds MITM adaptation and suffers the longest training latency.

==================================================
4. Small replay strategy
==================================================
Strategy B implements a balanced, minimal memory buffer sampled uniformly (100 samples per known class, yielding 900 total old samples). This represents a drastically reduced memory footprint (>99.30% reduction).

==================================================
5. New-only strategy
==================================================
Strategy C utilizes precisely 0 old samples. This simulates catastrophic forgetting extremes when attempting continuous adaptation without memory.

==================================================
6. MITM performance
==================================================
- Full Replay MITM F1: 0.6483
- Small Replay MITM F1: 0.9773
- New-Only MITM F1: 1.0000

==================================================
7. Old-class retention
==================================================
- Full Replay Known Macro-F1: 0.8902
- Small Replay Known Macro-F1: 0.8665
- New-Only Known Macro-F1: 0.0000

==================================================
8. Forgetting analysis
==================================================
- Full Replay Avg Forgetting: -0.0002 (Max: 0.0003)
- Small Replay Avg Forgetting: 0.0262 (Max: 0.0604)
- New-Only Avg Forgetting: 0.9890 (Max: 0.9996)

==================================================
9. Memory reduction
==================================================
- Strategy B (Small Replay) achieved a 99.30% reduction in historical data memory overhead compared to full replay.

==================================================
10. Training-time comparison
==================================================
- Full Replay Time: 0.76 seconds
- Small Replay Time: 0.06 seconds
- New-Only Time: 0.05 seconds

==================================================
11. Trade-off analysis
==================================================
The Small Replay Buffer (Strategy B) presents a compelling trade-off. By retaining merely ~0.7% of the historical data, it massively reduces training time and memory, while mitigating catastrophic forgetting far better than Strategy C. Interestingly, it also alleviates the extreme class imbalance present in Strategy A, allowing the model to adapt much more strongly to the new MITM class (higher MITM F1). The penalty is a moderate degree of average forgetting on old classes compared to Full Replay.

==================================================
12. Limitations
==================================================
The 100-samples-per-class buffer is static and sampled universally via a flat random function. It does not intelligently capture decision boundaries or represent distributional shifts beyond uniform chance. More sophisticated reservoir sampling or coreset optimization could potentially minimize forgetting even further.

==================================================
13. Conclusion
==================================================
A small representative replay buffer successfully approximates effective continuous adaptation. Strategy B achieved superior MITM learning performance compared to Full Replay by circumventing severe historical bias, reduced training latency significantly, and prevented total catastrophic forgetting. This affirms that computationally lightweight, memory-efficient replay mechanisms are highly viable for adaptive NIDS.
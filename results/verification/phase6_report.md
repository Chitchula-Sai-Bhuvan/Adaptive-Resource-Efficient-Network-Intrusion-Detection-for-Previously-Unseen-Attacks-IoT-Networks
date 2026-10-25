# Phase 6 Report: Adaptive / Continual Learning of a Confirmed Unseen Attack

==================================================
1. Research objective
==================================================
To determine whether the NIDS can incrementally learn a newly confirmed MITM attack class while retaining its previously learned knowledge of known attacks. This Phase evaluates a baseline naive-replay adaptation strategy.

==================================================
2. Adaptation scenario
==================================================
Detection (Phase 5) flagged suspicious traffic via Isolation Forest. We assume a subset of that traffic has been externally confirmed by an analyst as MITM. The adaptation process incorporates these confirmed samples into the classifier, which is then tested for new learning and old knowledge retention.

==================================================
3. Old knowledge dataset
==================================================
The original known-class training data: 127787 samples across 9 known attack and normal classes.

==================================================
4. New MITM adaptation dataset
==================================================
A controlled 20% subset of the MITM samples was used for adaptation: 207 samples.

==================================================
5. Untouched MITM evaluation dataset
==================================================
The remaining 80% of MITM samples were held in a strictly untouched evaluation set: 832 samples. They were never used for training or hyperparameter selection.

==================================================
6. Model configuration
==================================================
Random Forest (n_estimators=100, max_depth=20, random_state=42) was utilized for both the original Model A and adapted Model B.

==================================================
7. Adaptation strategy
==================================================
A full replay-based incremental baseline was used. The new adaptation dataset consisted of a direct concatenation of the entire old training set and the new confirmed MITM samples. This represents the simplest non-optimized continuous learning approach. No class-balancing techniques were applied, meaning the old training set heavily dominates the adaptation samples.

==================================================
8. Before-adaptation results
==================================================
- Known Accuracy: 0.9902
- Known Macro-F1: 0.9890
- Known Weighted-F1: 0.9902
- MITM F1: 0.0000

==================================================
9. After-adaptation results
==================================================
- Known Accuracy: 0.9900
- Known Macro-F1: 0.8902
- Known Weighted-F1: 0.9903
- MITM F1: 0.6483

==================================================
10. MITM learning performance
==================================================
On the untouched 80% evaluation set:
- MITM Precision: 1.0000
- MITM Recall: 0.4796
- MITM F1: 0.6483

==================================================
11. Per-class forgetting
==================================================
- backdoor: 0.0000
- ddos: -0.0002
- dos: 0.0003
- injection: 0.0003
- normal: -0.0002
- password: -0.0002
- ransomware: 0.0000
- scanning: -0.0001
- xss: -0.0014

==================================================
12. Average forgetting
==================================================
Mean F1 loss across known classes: -0.0002

==================================================
13. Maximum forgetting
==================================================
Maximum F1 loss for a single class: 0.0003

==================================================
14. Limitations
==================================================
This is a naive replay baseline. Training on the entire massive old dataset heavily biases the model against the small MITM adaptation set (127,000 old samples vs ~200 new samples). Full replay is computationally prohibitive in true continual learning scenarios and serves only to anchor the evaluation framework. True continual learning techniques (EWC, rehearsal buffers) are not yet employed.

==================================================
15. Research interpretation
==================================================
The baseline replay strategy successfully retains prior knowledge, as evidenced by near-zero forgetting across known classes. However, learning performance for the new MITM class is potentially stifled due to severe class imbalance during naive replay adaptation, highlighting the need for specialized continuous learning mechanics beyond simple dataset concatenation.

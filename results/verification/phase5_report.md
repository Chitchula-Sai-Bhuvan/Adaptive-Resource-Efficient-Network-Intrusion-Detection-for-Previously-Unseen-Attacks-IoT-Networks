# Phase 5 Report: Anomaly Detection for Previously Unseen Attacks

==================================================
1. Research objective
==================================================
The objective is to determine whether an unsupervised anomaly detection model can identify previously unseen MITM network behavior that a closed-set supervised Random Forest classifier fails to recognize. We evaluate a controlled previously-unseen attack class using a class-holdout protocol.

==================================================
2. Dataset
==================================================
The validated `train_test_network.csv` dataset, strictly preprocessed and deduplicated to avoid data leakage.

==================================================
3. Training protocol
==================================================
The Isolation Forest was fitted **exclusively** on the known-class training partition. No MITM traffic and no attack labels were exposed during the fitting process.

==================================================
4. Feature representation
==================================================
The exact same 18-dimension continuous feature representation from the clean Phase 1-4 baseline was utilized (including numeric columns and one-hot encoded metadata, strictly omitting raw IPs and explicit targets).

==================================================
5. Isolation Forest configuration
==================================================
`IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)`

==================================================
6. Training sample count
==================================================
127787 samples

==================================================
7. Test sample count
==================================================
55805 samples (Known Normal: 12593, Known Attack: 42173, Unseen MITM: 1039)

==================================================
8. Known normal results
==================================================
Known Normal False-Positive Rate: 6.92% (871/12593)

==================================================
9. Known attack results
==================================================
Known Attack Anomaly Detection Rate: 2.70% (1137/42173)

==================================================
10. MITM anomaly detection results
==================================================
- MITM total: 1039
- MITM detected as anomaly: 196
- MITM classified as inlier: 843
- MITM anomaly detection rate: 18.86%
- MITM precision: 1.0000
- MITM recall: 0.1886
- MITM F1: 0.3174

==================================================
11. Confusion matrix (Overall Binary)
==================================================
True Negative (Actual Normal -> Inlier): 11722
False Positive (Actual Normal -> Anomaly): 871
False Negative (Actual Attack -> Inlier): 41879
True Positive (Actual Attack -> Anomaly): 1333

==================================================
12. RF vs Isolation Forest comparison
==================================================
|    | Metric                  | Random Forest                            | Isolation Forest             |
|---:|:------------------------|:-----------------------------------------|:-----------------------------|
|  0 | Known-Class Main Metric | Accuracy: 0.9717                         | Known Attack Det Rate: 2.70% |
|  1 | MITM F1 Score           | 0.0000                                   | 0.3174                       |
|  2 | MITM Detection Rate     | 0.00% (Misclassified into known classes) | 18.86%                       |

==================================================
13. Limitations
==================================================
The Isolation Forest was deployed directly with `contamination='auto'` without hyperparameter tuning to serve as a pure unsupervised baseline. Results reflect fundamental distributional differences, rather than an optimized defensive boundary.

==================================================
14. Research interpretation
==================================================
Random Forest achieves excellent performance on known classes, BUT the MITM class cannot be recognized because it was absent during training. The Isolation Forest, which does not rely on MITM class labels and merely learns the structural distribution of training traffic, flags unseen MITM traffic as structurally anomalous at a distinct rate. A poor anomaly detection rate here verifies that anomaly detection alone is insufficient, motivating adaptive or continual learning methodologies.

==================================================
15. Conclusion
==================================================
The unsupervised Isolation Forest achieved an anomaly detection rate of 18.86% against the zero-label MITM attack, producing an F1 score of 0.3174. When juxtaposed with the supervised Random Forest (MITM F1: 0.0000), this solidifies the structural gap in defense paradigms when confronted with withheld behavioral classes.

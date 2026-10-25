# Phase 5 Metric Validation Report

==================================================
1. EXPLAIN MITM SAMPLE COUNT
==================================================
- Original MITM count: 1043
- Number removed by duplicate handling: 4
- Number removed by invalid/missing-value handling: 0
- Final MITM count used in Phase 5: 1039
Explanation: By dropping high-cardinality metadata (IP addresses), 4 MITM flows became perfectly identical numerically to other MITM flows. They were removed by the explicit deduplication rule.

==================================================
2. VALIDATE ANOMALY CONFUSION MATRIX
==================================================
Binary Anomaly Confusion Matrix:
- True Positive (Actual Attack, Predicted Anomaly): 1333
- True Negative (Actual Normal, Predicted Inlier): 11722
- False Positive (Actual Normal, Predicted Anomaly): 871
- False Negative (Actual Attack, Predicted Inlier): 41879

- Accuracy:  0.2339
- Precision: 0.6048
- Recall:    0.0308
- F1 Score:  0.0587
- FPR:       0.0692
- FNR:       0.9692

==================================================
3. MITM-SPECIFIC RESULTS
==================================================
- Total MITM: 1039
- MITM predicted anomaly: 196
- MITM predicted inlier: 843
- MITM detection rate / recall: 0.1886 (18.86%)

==================================================
4. KNOWN NORMAL FALSE POSITIVE
==================================================
FPR_normal = 871 (FP) / 12593 (Total Normal)
Confirmed FPR_normal: 0.0692 (6.92%)

==================================================
5. KNOWN ATTACK ANOMALY RATE
==================================================
Known attack anomaly rate = 1137 / 42173
Confirmed Known Attack Detection Rate: 0.0270 (2.70%)

==================================================
6. ISOLATION FOREST TRAINING
==================================================
- MITM was not used during fitting: Confirmed. The training set strictly excluded the `mitm` class prior to isolation forest initialization.
- Test labels were not used: Confirmed. IsolationForest uses `.fit(X_train)` and ignores `y_train` entirely.
- MITM labels were not used to select contamination: Confirmed.
- contamination remained 'auto': Confirmed (Actual: auto)
- random_state = 42: Confirmed (Actual: 42)
- n_estimators = 100: Confirmed (Actual: 100)

==================================================
7. PRECISION CORRECTION
==================================================
The previously reported `MITM precision = 1.0000` is mathematically meaningless. It was calculated purely on the slice of MITM samples (`y_true` contained only 1s). Since there are no negative samples in that slice, a False Positive is mathematically impossible, artificially resulting in Precision = TP / (TP + 0) = 1.0.
This is an invalid, circular metric for a single class slice in anomaly detection. It is explicitly removed from the final report.
The proper binary precision for the entire test set (predicting anomalies correctly across all traffic) was calculated above as: 0.6048 (60.48094373865699%)
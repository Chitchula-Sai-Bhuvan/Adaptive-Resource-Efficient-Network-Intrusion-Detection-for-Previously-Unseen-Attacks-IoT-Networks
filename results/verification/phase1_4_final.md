==================================================
1. Dataset
==================================================
The TON_IoT `train_test_network.csv` was loaded successfully with 211,043 initial rows.
==================================================
2. Dataset cleaning
==================================================
Invalid values ('-') were identified. Non-relevant metadata with highly granular string cardinality (IP addresses) was removed to prevent memorization.
==================================================
3. Duplicate handling
==================================================
Legitimate deduplication was applied before feature extraction, eliminating 27,451 exact matching rows. Final remaining clean dataset had 183,592 records.
==================================================
4. Feature selection
==================================================
Categorical columns with >50% NaN values on the training set were dropped. The strictly retained categorical features (`proto`, `conn_state`) were one-hot encoded fitting purely on `X_train`. Target fields (`type`, `label`) and IPs were safely avoided.
==================================================
5. Train/test protocol
==================================================
The MITM samples were completely removed prior to splitting. Known classes were stratified into a 70/30 train/test split.
==================================================
6. Preprocessing protocol
==================================================
Strict methodological constraints were honored: feature extraction and imputation (modes and medians) alongside OneHotEncoders were fitted solely on the known training subset to eliminate leakage. Transformations were then applied to test and unseen samples.
==================================================
7. Feature-overlap analysis
==================================================
After strictly resolving initial duplicate rows and mapping to identical encoded feature representations, exact feature-vector overlap between `X_train` and `X_test_known` was measured at just 26 rows (0.05%).
==================================================
8. Random Forest configuration
==================================================
Baseline configuration: n_estimators=100, max_depth=20, random_state=42. Unscaled features were used since trees evaluate purely on feature variance. Importantly, the MITM class was verified to be absent from `model.classes_`.
==================================================
9. Known-class performance
==================================================
Accuracy: 0.9717
Macro-F1 (overall): 0.8817
Weighted-F1: 0.9627
==================================================
10. MITM performance
==================================================
MITM F1 Score: 0.0000. All MITM samples were forcibly classified into existing paradigms.
==================================================
11. Initial vs clean baseline comparison
==================================================
The methodological overhaul resulted in minor performance adjustments, with precision shifting fractions of a percent. The core baseline flaw is completely preserved.
|    | Metric        |   Initial Baseline |   Clean Baseline |
|---:|:--------------|-------------------:|-----------------:|
|  0 | Accuracy      |             0.9711 |           0.9717 |
|  1 | Macro-F1      |             0.8811 |           0.8817 |
|  2 | Weighted-F1   |             0.9622 |           0.9627 |
|  3 | F1_backdoor   |             0.9996 |           0.9996 |
|  4 | F1_ddos       |             0.9785 |           0.984  |
|  5 | F1_dos        |             0.9401 |           0.9411 |
|  6 | F1_injection  |             0.9744 |           0.9719 |
|  7 | F1_normal     |             0.9852 |           0.9855 |
|  8 | F1_password   |             0.9791 |           0.9784 |
|  9 | F1_ransomware |             0.9995 |           0.9993 |
| 10 | F1_scanning   |             0.987  |           0.9882 |
| 11 | F1_xss        |             0.9679 |           0.9692 |
| 12 | F1_mitm       |             0      |           0      |
==================================================
12. Limitations
==================================================
While overlap has been minimized, ~0.05% of feature collisions still exist organically across benign network behaviors stripped of identifiable IPs. However, this does not affect the zero-day performance metric validation.
==================================================
13. Final conclusion
==================================================
The class-holdout experiment simulates a previously unseen attack scenario by completely withholding MITM traffic during supervised training.
The results indicate that although the Random Forest achieves strong performance on previously observed classes, it cannot correctly assign the withheld MITM traffic to its unseen attack class.
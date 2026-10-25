==================================================
1. Dataset verification
==================================================
- Total original rows: 211043
- Total columns: 44
- Target column: 'type'
- Number of classes: 10
- Classes:
  * normal: 50000
  * backdoor: 20000
  * ddos: 20000
  * dos: 20000
  * injection: 20000
  * password: 20000
  * ransomware: 20000
  * scanning: 20000
  * xss: 20000
  * mitm: 1043

==================================================
2. Class distribution
==================================================
See section 1.

==================================================
3. Duplicate analysis
==================================================
A. Number of duplicate rows in original dataset: 20569
B. Number of rows removed by duplicate removal (after NaN drop): 28045
C. Number of rows removed because of missing/invalid values: 0
D. Final number of rows: 182998

==================================================
4. MITM holdout verification
==================================================
- Total MITM samples in cleaned dataset: 1031
- MITM samples in training: 0
- MITM samples in test: 1031
- Confirmed: all MITM samples are present only in the unseen test set.

==================================================
5. Train/test overlap verification
==================================================
- Number of exact overlapping feature rows: 10
- Overlap percentage (relative to test set): 0.02%
WARNING: Potential leakage issue due to duplicate feature rows spanning train/test split. Since we dropped non-numeric metadata, distinct network flows may share identical numerical characteristics, leading to feature-level collisions.

==================================================
6. Target leakage verification
==================================================
- 'type' is not present in features: True
- 'label' is not present in features: True
- Exact feature names used:
  * src_port
  * dst_port
  * duration
  * src_bytes
  * dst_bytes
  * missed_bytes
  * src_pkts
  * src_ip_bytes
  * dst_pkts
  * dst_ip_bytes
  * dns_qclass
  * dns_qtype
  * dns_rcode
  * http_request_body_len
  * http_response_body_len
  * http_status_code
- Verified no feature is an encoded copy of the target or contains attack string names.

==================================================
7. Preprocessing leakage verification
==================================================
Potential preprocessing leakage: [Missing Value Dropping, Global Label Encoding]
Explanation: The dropping of missing values and columns (>50% NaN) was computed on the full dataset before the train/test split, meaning global dataset statistics slightly influenced the feature selection space. The LabelEncoder was also fitted globally. Scaling was NOT applied to the RF model, avoiding test-to-train distribution leakage for continuous values.

==================================================
8. Random Forest learned classes
==================================================
- Classes actually learned by Random Forest (encoded IDs): [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(5), np.int64(6), np.int64(7), np.int64(8), np.int64(9)]
- Is MITM (encoded 4) among model.classes_? False

==================================================
9. Known-class performance
==================================================
- Known-Class Accuracy: 0.9895
- Known-Class Macro-F1: 0.9883
- Known-Class Weighted-F1: 0.9895
- Precision, Recall, F1 for every known class:
  * backdoor: P=0.9996, R=0.9996, F1=0.9996
  * ddos: P=0.9838, R=0.9813, F1=0.9825
  * dos: P=0.9897, R=0.9811, F1=0.9854
  * injection: P=0.9802, R=0.9741, F1=0.9771
  * normal: P=0.9960, R=0.9971, F1=0.9965
  * password: P=0.9951, R=0.9943, F1=0.9947
  * ransomware: P=0.9993, R=0.9998, F1=0.9995
  * scanning: P=0.9861, R=0.9934, F1=0.9897
  * xss: P=0.9654, R=0.9741, F1=0.9697

==================================================
10. MITM prediction distribution
==================================================
Total MITM samples predicted: 1031
- backdoor: 0
- ddos: 49
- dos: 422
- injection: 34
- mitm: 0
- normal: 289
- password: 190
- ransomware: 0
- scanning: 31
- xss: 16

==================================================
11. Unseen MITM performance
==================================================
- MITM Precision: 0.0000
- MITM Recall: 0.0000
- MITM F1: 0.0000

==================================================
12. Final experimental conclusion
==================================================
The class-holdout protocol simulates a previously unseen attack scenario by completely withholding the MITM class during supervised training.
The results indicate that although the Random Forest achieves strong performance on previously observed classes, it cannot correctly assign the withheld MITM traffic to its unseen attack class.

==================================================
13. Issues requiring attention
==================================================
1. **Data Overlap**: Dropping unique string identifiers (IPs, HTTP URIs) causes thousands of network flow records to become numerically identical. We observed overlapping feature rows spanning train and test sets. This may slightly inflate known-class accuracy due to identical numerical signatures being memorized.
2. **Global Preprocessing**: Columns with >50% NaNs were dropped globally before splitting. While this doesn't leak target predictions directly, it does leak distribution structure to the training set. Strictly, this should be fitted on the train split only.
No critical leakage undermining the core MITM-holdout experiment was found (the unseen attack truly remained unseen).
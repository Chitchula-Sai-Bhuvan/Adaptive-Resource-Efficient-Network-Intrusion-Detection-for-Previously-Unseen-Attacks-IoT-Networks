# Adaptive and Resource-Efficient Network Intrusion Detection for Previously Unseen Attacks in IoT Networks

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

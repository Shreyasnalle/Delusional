# Mastercard AI Defense Lab

This repository contains the prototype for the Mastercard AI Defense Lab, built for the Mastercard Innovation Challenge at the Global Fintech Fest (GFF) 2026. The project is an end-to-end, closed-loop Red Team / Blue Team AI system that generates, detects, and dynamically adapts to emerging GenAI-powered payment fraud vectors.

---

## 1. The Frauds We Are Covering (Red Team)

We focus on three specific, high-impact payment fraud vectors that are simulatable at the transaction level and closely mirror the modern financial crime landscape.

### 1.1 Synthetic Identity Fraud → Mule Account Bootstrapping
Fraudsters combine stolen PII (e.g., SSNs) with fabricated data and AI-generated headshots to open synthetic bank accounts. These accounts are "aged" with normal, low-value behavior for months before suddenly receiving large inbound transfers and rapidly dispersing funds across a network of other synthetic accounts (money mules).

### 1.2 AI-Orchestrated Card Testing / BIN Attack
Using stolen payment card data (PANs) purchased on the dark web, automated botnets fire rapid, micro-transactions ($0.50–$1.00) across hundreds of distinct merchants in minutes to validate which cards are active. Validated cards are immediately hit with large fraudulent purchases.

### 1.3 APP-Style Scam (Beneficiary-Risk Pattern)
In an Authorized Push Payment (APP) scam, the victim is psychologically manipulated (e.g., a fake bank fraud department call) into willingly transferring funds to a "safe" account controlled by fraudsters. Because the sender's credentials and biometrics are perfectly legitimate, the transaction cannot be flagged by analyzing the sender alone.

---

## 2. How We Fight Back (Blue Team)

Our defense architecture utilizes modern, AI-driven approaches inspired directly by Mastercard's enterprise security suite to intercept these advanced attacks.

### 2.1 Defending Mule Account Bootstrapping (Inspired by TRACE & DI Pro)
- **Behavioral Discontinuity:** We detect sudden jumps from dormant or small-purchase patterns to large-inflow-then-rapid-outflow activity.
- **Graph Linkage:** We build a multi-institution account graph mapping shared devices, IPs, and beneficiary links. When one account is flagged, we inspect its graph neighborhood to uncover the entire mule ring.
- **Model:** XGBoost using account-level features combined with graph-derived features (shared-attribute clusters, degree centrality).

### 2.2 Defending Card Testing / BIN Attacks (Inspired by Safety Net)
- **Streaming Velocity Tracking:** We monitor rolling-window features per card, including transactions per minute, distinct merchants hit, and uniform micro-charge patterns.
- **Low-Latency Real-Time Scoring:** Because this is a speed problem, we utilize lightweight classifiers (e.g., Logistic Regression) benchmarking decision latency (milliseconds) alongside standard accuracy metrics to block attacks in real-time.

### 2.3 Defending APP-Style Scams (Inspired by Consumer Fraud Risk - CFR)
- **Inverted Recipient Focus:** Instead of evaluating the sender, our AI scores the *recipient* account.
- **Key Signals:** Is the beneficiary account brand new? Is it their first interaction? Is the recipient seeing sudden inbound spikes from multiple unrelated senders? Does the recipient link to other flagged nodes in the graph?

---

## 3. Closed-Loop Workflow (The AI Arms Race)

The core innovation of this platform is the adversarial feedback loop. The system does not just detect static fraud; it learns and evolves. 

For each of the three fraud vectors, the system runs an independent workflow:

1. **IDENTIFY:** Define the attack vector and mechanism.
2. **GENERATE:** Simulate the attack accurately against a baseline dataset (PaySim).
3. **DEFEND:** Train the specialized Blue Team classifier and score the transactions.
4. **EVALUATE:** Measure Precision, Recall, F1, AUC, and Decision Latency.
5. **FALSE NEGATIVES:** Identify the specific attacks that successfully evaded detection.
6. **REGENERATE:** Use the attack generator to create harder, perturbed attack variants specifically targeting the blind spots discovered in step 5.
7. **RETRAIN:** Combine the original and hardened attack batches, retrain the classifier, and repeat the loop to continuously harden the defense.

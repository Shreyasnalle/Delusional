# Mastercard AI Defense Lab

This repository contains the prototype for the Mastercard AI Defense Lab, built for the Mastercard Innovation Challenge at the Global Fintech Fest (GFF) 2026. The project is an end-to-end, closed-loop Red Team / Blue Team AI system that generates, detects, and dynamically adapts to emerging GenAI-powered payment fraud vectors.

---

## 1. The Fraud We Are Covering (Red Team)

We focus on a high-impact payment fraud vector that is simulatable at the transaction level and closely mirrors the modern financial crime landscape.

### Synthetic Identity Fraud -> Mule Account Bootstrapping
Fraudsters combine stolen PII (e.g., SSNs) with fabricated data and AI-generated headshots to open synthetic bank accounts. These accounts are aged with normal, low-value behavior for months before suddenly receiving large inbound transfers and rapidly dispersing funds across a network of other synthetic accounts (money mules).

---

## 2. How We Fight Back (Blue Team)

Our defense architecture utilizes modern, AI-driven approaches inspired directly by Mastercard's enterprise security suite to intercept these advanced attacks.

### Defending Mule Account Bootstrapping (Inspired by TRACE & DI Pro)
- Behavioral Discontinuity: We detect sudden jumps from dormant or small-purchase patterns to large-inflow-then-rapid-outflow activity.
- Graph Linkage: We build a multi-institution account graph mapping shared devices, IPs, and beneficiary links. When one account is flagged, we inspect its graph neighborhood to uncover the entire mule ring.
- Model: Graph Neural Networks (Multi-GNN) & XGBoost using account-level features combined with graph-derived features (shared-attribute clusters, degree centrality, temporal ports, and time deltas).

---

## 3. Closed-Loop Workflow (The AI Arms Race)

The core innovation of this platform is the adversarial feedback loop. The system does not just detect static fraud; it learns and evolves dynamically.

Workflow Pipeline:
1. IDENTIFY -> 2. GENERATE -> 3. DEFEND -> 4. EVALUATE -> 5. FALSE NEGATIVES -> 6. REGENERATE -> 7. RETRAIN

1. IDENTIFY: Define the attack vector and mechanism.
2. GENERATE: Simulate the attack accurately against a baseline dataset.
3. DEFEND: Train the specialized Blue Team classifier and score the transactions.
4. EVALUATE: Measure Precision, Recall, F1, AUC, and Decision Latency.
5. FALSE NEGATIVES: Identify the specific attacks that successfully evaded detection.
6. REGENERATE: Use the attack generator to create harder, perturbed attack variants specifically targeting the blind spots discovered in step 5.
7. RETRAIN: Combine the original and hardened attack batches, retrain the classifier, and repeat the loop to continuously harden the defense.

---

## 4. Running Kaggle Remote Session

To run the training notebook on Kaggle via cloudflared tunnel:

```python
import os, time

os.system("fuser -k 9999/tcp 2>/dev/null")
time.sleep(2)
print("Port 9999 cleared.")

TOKEN = "delusional123"
PORT = 9999
os.system(f"jupyter notebook --ip=0.0.0.0 --port={PORT} --no-browser --allow-root --NotebookApp.token={TOKEN} --NotebookApp.disable_check_xsrf=True &")
time.sleep(5)
print("Jupyter started on port 9999.")

os.system("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared")
os.system("chmod +x cloudflared")
print("=" * 60)
print("YOUR IDE URL = https://<xxx>.trycloudflare.com/?token=delusional123")
print("Look for the trycloudflare.com link below:")
print("=" * 60)

os.system(f"./cloudflared tunnel --url http://localhost:{PORT}")
```

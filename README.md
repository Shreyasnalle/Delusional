# GENAI FRAUD DEFENSE
### Self-Improving Multi-GNN Model for GenAI-Powered Payment Frauds

**GenAI Fraud Defense** is an autonomous, closed-loop Red Team vs. Blue Team platform designed to generate, evaluate, and dynamically adapt to modern AI-driven financial crimes. The system pairs an LLM-driven Red Team Adversarial Architect with a Heterogeneous Graph Neural Network (Multi-GNN) Blue Team Defense Model in a continuous active learning loop.

---

## 1. Fraud Vectors Covered (Identify Phase)

The system focuses on 5 high-impact, GenAI-amplified financial crimes that challenge static rule engines and traditional machine learning models:

1. **Synthetic Identity & Mule Network Bootstrapping**  
   Stolen credentials and synthetic identities are combined to open bank accounts. These accounts remain dormant or low-activity before suddenly executing rapid fund transfers.
2. **Automated Smurfing & Micro-Splitting**  
   Large illicit financial targets are broken down into hundreds of randomized micro-transactions below regulatory reporting thresholds, keeping the stolen amounts untracked.
3. **Temporal Poisson Smoothing & Time-Delta Masking**  
   Transaction time intervals are sampled using stochastic Poisson process distributions to eliminate periodic burst signatures that trigger static anomaly rules, thus bypassing the trained model's detection capabilities mathematically. 
4. **Indian Business Hours Masking (IST Realism)**  
   Timestamps are automatically aligned with commercial banking windows (09:30 AM – 06:30 PM), blending malicious edges seamlessly into high-volume domestic payments, which keeps the fraud out of sight.
5. **Graph Topology Evasion & Noise Injection**  
   Legitimate-looking "noise" transactions are strategically injected between malicious nodes to artificially lower node degree centrality, alter in/out port ratios, and pass unnoticed. 

---

## 2. Red Team Attack Generation (Generative Red Teaming)

The **Red Team Engine** simulates evasive payment fraud through a two-stage process:

1. **LLM Adversarial Architect (Groq AI)**  
   The Red Team queries the LLM using transaction schemas and missed fraud patterns (`unnoticed_frauds.csv`). The LLM analyzes GNN blind spots and outputs raw JSON attack parameters specifying hub sizes, micro-amount boundaries, Poisson time-delta rates, and noise ratios.
2. **Procedural Execution Engine**  
   The execution engine scales the LLM parameters into 100,000 synthetic transaction datasets with dynamic fraud ratios (10,000 to 60,000 frauds). Transaction timestamps, amounts, payment channels, and bank routing are synthesized in real time.

---

## 3. Blue Team Active Defense & Self-Improvement Loop

The **Blue Team Engine** intercepts and adapts to evasive attacks through an active learning feedback loop:

1. **Heterogeneous Graph Neural Network (Multi-GNN)**  
   The defense pipeline constructs a multi-relational graph (`Account` $\rightarrow$ `To` $\rightarrow$ `Account`) using PyTorch Geometric GINe layers with edge attribute updates. The baseline model evaluates the full incoming dataset to flag suspicious edges.
2. **Hard-Example Extraction & False Negative Mining**  
   Transactions missed by the baseline model (False Negatives) and false alarms (False Positives) are extracted to form a specialized hard-example training pool.
3. **Autonomous GNN Fine-Tuning**  
   The GNN model undergoes 3 epochs of active fine-tuning using AdamW optimization and class-weighted cross-entropy loss.
4. **Active Adaptation & Evasion Memory**  
   The fine-tuned model thus broadens its boundaries of fraud detection. Remaining missed frauds are exported to `unnoticed_frauds.csv` so the Red Team LLM can target new model blind spots in subsequent cycles, thereby making the Red Team stronger in parallel as well. 

---

## 4. End-to-End System Workflow

```text
Start Simulation ➔ Red Team (generates attack) ➔ Blue Team (Multi-Hetero GNN model) ➔ Extract frauds which passed 
       ▲                                                                                           │                         
       │                                                                                  Active GNN Fine-Tuning
       │                                                                                           │
Use them to strengthen next attack generation  Export Unnoticed Frauds  Shows Precision Report  
```

---

## 5. Developer & Code Inspection Guide

### Local Setup & Execution
To run the full stack locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Shreyasnalle/Delusional.git
   cd Delusional
   ```

2. **Backend Setup (FastAPI + PyTorch GNN)**:
   ```bash
   conda create -n mastermoney python=3.9 -y
   conda activate mastermoney
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --port 8000 --reload
   ```

3. **Frontend Setup (Next.js)**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Open `http://localhost:3000` in your browser.

---

### Kaggle GPU Cloud Server Code
For training large graph datasets on GPU, use **Kaggle GPU Instances** (Recommended GPU: **Kaggle T4 x2 GPU**).

Run the following code block in a Kaggle Notebook cell to launch a remote Jupyter server via Cloudflare Tunnel:

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

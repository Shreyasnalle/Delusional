import os
import json
import random
import glob
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from groq import Groq
except ImportError:
    print("Warning: groq library not installed. Please run `pip install groq`.")

class AttackGenerator:
    def __init__(self, raw_data_path, output_dir=None):
        self.raw_data_path = raw_data_path
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attacks")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.client = None
        if os.getenv("GROQ_API_KEY"):
            try:
                self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            except Exception as e:
                print(f"Error initializing Groq client: {e}")
        else:
            print("Warning: GROQ_API_KEY not found in environment.")

    def _get_next_filename(self):
        """Finds the next attack_N.csv filename."""
        existing_files = glob.glob(os.path.join(self.output_dir, "attack_*.csv"))
        max_idx = 0
        for f in existing_files:
            try:
                idx = int(os.path.basename(f).split('_')[1].split('.')[0])
                max_idx = max(max_idx, idx)
            except ValueError:
                continue
        return os.path.join(self.output_dir, f"attack_{max_idx + 1}.csv")

    def load_and_slice_data(self, skip_rows=5_000_000, target_rows=10_000, fraud_ratio=0.4):
        """
        Loads a slice of the dataset ensuring max 40% fraud, leaving the rest as real background noise.
        """
        print(f"Loading data from {self.raw_data_path} (skipping first {skip_rows:,} rows)...")
        try:
            chunk_size = 500_000
            df = pd.read_csv(self.raw_data_path, skiprows=range(1, skip_rows), nrows=chunk_size)
            
            if 'Is Laundering' not in df.columns:
                print("Error: 'Is Laundering' column not found in dataset.")
                return None

            fraud_df = df[df['Is Laundering'] == 1]
            clean_df = df[df['Is Laundering'] == 0]

            num_fraud_needed = int(target_rows * fraud_ratio)
            num_clean_needed = target_rows - num_fraud_needed

            if len(fraud_df) < num_fraud_needed:
                num_fraud_needed = len(fraud_df)
                num_clean_needed = target_rows - num_fraud_needed

            sampled_fraud = fraud_df.sample(n=num_fraud_needed, replace=True)
            sampled_clean = clean_df.sample(n=num_clean_needed, replace=True)

            sampled_df = pd.concat([sampled_clean, sampled_fraud]).sample(frac=1).reset_index(drop=True)
            print(f"Sampled {len(sampled_df)} rows as baseline. Fraud: {len(sampled_fraud)} ({len(sampled_fraud)/len(sampled_df):.1%}), Clean: {len(sampled_clean)}")
            return sampled_df

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def prompt_llm_architect(self, sample_data):
        """
        Feeds a JSON sample to the LLM (Groq API) to orchestrate advanced attack vectors.
        """
        if not self.client:
            print("No Groq API key found. Using default procedural parameters.")
            return self._default_attack_params()

        essential_cols = ['Timestamp', 'Account', 'Account.1', 'Amount Received', 'Payment Format', 'Is Laundering']
        available_cols = [c for c in essential_cols if c in sample_data.columns]
        sample_json = sample_data[available_cols].head(10).to_json(orient="records")
        
        system_prompt = """
You are an elite Adversarial ML Red-Teamer and Financial Crimes Architect.
Your objective is to design synthetic transaction parameters that can successfully evade a highly tuned Heterogeneous Graph Neural Network (Multi-GNN). 
The defense model analyzes Port Numberings (in/out counts) and Time-Deltas to detect structured money laundering.

Design parameters for 3 evasion-resistant attack strategies:
1. Mule Bootstrapping: Complex hub-and-spoke networks with varying spoke sizes to avoid static topological signatures.
2. Smurfing / Micro-Splitting: Breaking large laundered targets into highly randomized micro-transactions carefully below regulatory thresholds (e.g. $10k), with unpredictable time-deltas.
3. Evasion Perturbation: Injecting dummy, legitimate-looking "noise" transactions between malicious edges to artificially inflate time-deltas and break the GNN's temporal continuity algorithms.

Respond ONLY with a valid raw JSON object. Do not wrap it in markdown blockquotes like ```json.
The structure MUST match this exactly:
{
    "mule_network": {
        "num_hubs": 25,
        "spokes_per_hub": [10, 50],
        "time_delta_seconds_range": [3, 120]
    },
    "smurfing": {
        "target_amount_range": [150000, 800000],
        "split_count_range": [30, 90],
        "max_micro_amount": 9950
    },
    "evasion": {
        "noise_ratio": 0.35,
        "noise_amount_range": [5, 100]
    }
}
"""
        user_prompt = f"Here is the sample of the target transaction network. Design the attack parameters based on this data format:\n{sample_json}"
        
        print("Querying Groq (openai/gpt-oss-20b) Architect for highly evasive attack parameters...")
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                model="openai/gpt-oss-20b",
                temperature=0.7
            )
            response_text = chat_completion.choices[0].message.content
            import re
            # Strip <think> reasoning blocks if model output included them
            clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            clean_text = clean_text.replace("```json", "").replace("```", "").strip()
            params = json.loads(clean_text)
            print("Groq LLM Evasive Parameters received successfully.")
            return params
        except Exception as e:
            print(f"Groq API Error: {e}. Falling back to default parameters.")
            return self._default_attack_params()

    def _default_attack_params(self):
        return {
            "mule_network": {"num_hubs": 50, "spokes_per_hub": [3, 10], "time_delta_seconds_range": [5, 30]},
            "smurfing": {"target_amount_range": [50000, 200000], "split_count_range": [10, 50], "max_micro_amount": 9900},
            "evasion": {"noise_ratio": 0.15, "noise_amount_range": [5, 200]}
        }

    def generate_attacks(self, base_df, params, total_target_rows=100_000, max_fraud_ratio=0.4):
        """
        Procedural engine that scales the LLM's attack parameters to exactly 100,000 transactions.
        Forces the generated dataset to never exceed the max_fraud_ratio.
        """
        print(f"Scaling evasive attacks to {total_target_rows:,} transactions (Max Fraud limit: {max_fraud_ratio:.1%})...")
        
        accounts = list(set(base_df['Account'].tolist() + base_df['Account.1'].tolist())) if 'Account' in base_df.columns else [f"ACC_{i}" for i in range(10000)]
        currencies = base_df['Receiving Currency'].unique().tolist() if 'Receiving Currency' in base_df.columns else ["USD", "EUR", "GBP"]
        payment_formats = base_df['Payment Format'].unique().tolist() if 'Payment Format' in base_df.columns else ["Wire", "Credit Card", "ACH"]
        
        generated_rows = []
        base_time = datetime.now()
        
        target_fraud_rows = int(total_target_rows * max_fraud_ratio)
        current_fraud_count = 0
        
        # --- Strategy 1: Mule Bootstrapping ---
        mule_p = params.get("mule_network", {})
        num_hubs = mule_p.get("num_hubs", 20)
        
        for _ in range(num_hubs):
            if current_fraud_count >= target_fraud_rows: break
            
            hub_acc = f"MULE_HUB_{random.randint(10000, 99999)}"
            spokes_range = mule_p.get("spokes_per_hub", [5, 10])
            num_spokes = random.randint(spokes_range[0], max(spokes_range[0], spokes_range[1]))
            time_delta_range = mule_p.get("time_delta_seconds_range", [10, 60])
            
            for _ in range(num_spokes):
                if current_fraud_count >= target_fraud_rows: break
                spoke_acc = random.choice(accounts)
                amount = random.uniform(1000, 50000)
                t_delta = random.randint(time_delta_range[0], max(time_delta_range[0], time_delta_range[1]))
                base_time += timedelta(seconds=t_delta)
                
                generated_rows.append(self._create_row(
                    timestamp=base_time.strftime("%Y/%m/%d %H:%M"),
                    from_acc=hub_acc, to_acc=spoke_acc, amount=amount,
                    currency=random.choice(currencies), p_format=random.choice(payment_formats),
                    is_laundering=1
                ))
                current_fraud_count += 1
                
        # --- Strategy 2: Smurfing / Micro-Splitting ---
        smurf_p = params.get("smurfing", {})
        amount_range = smurf_p.get("target_amount_range", [100000, 500000])
        split_range = smurf_p.get("split_count_range", [20, 60])
        max_micro = smurf_p.get("max_micro_amount", 9500)
        
        # Distribute the remaining fraud quota into smurfing clusters
        num_smurf_attacks = (target_fraud_rows - current_fraud_count) // max(1, (split_range[0] + split_range[1]) // 2)
        
        for _ in range(max(1, num_smurf_attacks)):
            if current_fraud_count >= target_fraud_rows: break
            
            target_amount = random.uniform(amount_range[0], max(amount_range[0], amount_range[1]))
            num_splits = random.randint(split_range[0], max(split_range[0], split_range[1]))
            source_acc = random.choice(accounts)
            dest_acc = f"OFFSHORE_{random.randint(1000, 9999)}"
            
            for _ in range(num_splits):
                if current_fraud_count >= target_fraud_rows: break
                micro_amount = min(target_amount / num_splits, max_micro) * random.uniform(0.85, 0.99) # Sub-threshold jitter
                base_time += timedelta(seconds=random.randint(1, 45))
                
                generated_rows.append(self._create_row(
                    timestamp=base_time.strftime("%Y/%m/%d %H:%M"),
                    from_acc=source_acc, to_acc=dest_acc, amount=micro_amount,
                    currency=random.choice(currencies), p_format=random.choice(payment_formats),
                    is_laundering=1
                ))
                current_fraud_count += 1

        # --- Strategy 3: Real Background & Evasion Perturbation ---
        evasion_p = params.get("evasion", {})
        noise_amount_range = evasion_p.get("noise_amount_range", [10, 500])
        noise_ratio = evasion_p.get("noise_ratio", 0.2)
        
        clean_rows_needed = total_target_rows - len(generated_rows)
        print(f"Injecting {clean_rows_needed:,} real/noise transactions to mask fraud...")
        
        for _ in range(clean_rows_needed):
            base_time += timedelta(seconds=random.randint(1, 100))
            is_noise = random.random() < noise_ratio
            
            # If it's an evasion noise transaction, use tiny amounts
            # If it's standard real background, use a wider random distribution
            amount = random.uniform(noise_amount_range[0], max(noise_amount_range[0], noise_amount_range[1])) if is_noise else random.uniform(10, 8000)
            
            generated_rows.append(self._create_row(
                timestamp=base_time.strftime("%Y/%m/%d %H:%M"),
                from_acc=random.choice(accounts), to_acc=random.choice(accounts), amount=amount,
                currency=random.choice(currencies), p_format=random.choice(payment_formats),
                is_laundering=0
            ))
            
        # Randomly shuffle everything to break sequential blocks and scatter the fraud
        random.shuffle(generated_rows)
        final_df = pd.DataFrame(generated_rows)
        
        output_file = self._get_next_filename()
        final_df.to_csv(output_file, index=False)
        
        actual_fraud = final_df['Is Laundering'].sum()
        actual_clean = len(final_df) - actual_fraud
        print(f"\n✅ Attack Generator Complete!")
        print(f"Output File: {output_file}")
        print(f"Total Rows : {len(final_df):,}")
        print(f"Distribution -> Fraud: {actual_fraud:,} ({actual_fraud/len(final_df):.1%}), Clean: {actual_clean:,} ({actual_clean/len(final_df):.1%})")
        
        return final_df

    def _create_row(self, timestamp, from_acc, to_acc, amount, currency, p_format, is_laundering):
        return {
            "Timestamp": timestamp,
            "From Bank": random.randint(10, 99),
            "Account": from_acc,
            "To Bank": random.randint(10, 99),
            "Account.1": to_acc,
            "Amount Received": round(amount, 2),
            "Receiving Currency": currency,
            "Amount Paid": round(amount, 2),
            "Payment Currency": currency,
            "Payment Format": p_format,
            "Is Laundering": is_laundering
        }

if __name__ == "__main__":
    # Check Kaggle cache first, fallback to local model directory
    RAW_CSV_PATH = os.path.expanduser("~/.cache/kagglehub/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/versions/8/HI-Large_Trans.csv")
    if not os.path.exists(RAW_CSV_PATH):
        RAW_CSV_PATH = "/home/shreyas-nalle/Desktop/Delusional/model/formatted_transactions.csv"
        
    generator = AttackGenerator(raw_data_path=RAW_CSV_PATH)
    
    # 1. Load baseline (slice 5M+, get 10k sample of <= 40% fraud)
    sample_df = generator.load_and_slice_data(skip_rows=5_000_000, target_rows=10_000, fraud_ratio=0.4)
    
    if sample_df is not None:
        # 2. Get LLM Architect Parameters
        attack_params = generator.prompt_llm_architect(sample_df)
        
        # 3. Procedurally scale to exactly 100k
        generator.generate_attacks(sample_df, attack_params, total_target_rows=100_000, max_fraud_ratio=0.4)

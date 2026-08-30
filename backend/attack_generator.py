import os
import json
import random
import glob
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except ImportError:
    print("Warning: groq library not installed")

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
            print("Warning: GROQ_API_KEY not found in environment")

    def _get_next_filename(self):
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
        print(f"Loading data from {self.raw_data_path}")
        try:
            chunk_size = 500_000
            df = pd.read_csv(self.raw_data_path, skiprows=range(1, skip_rows), nrows=chunk_size)
            
            if 'Is Laundering' not in df.columns:
                print("Error: 'Is Laundering' column not found in dataset")
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
            print(f"Sampled {len(sampled_df)} rows as baseline")
            return sampled_df

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def prompt_llm_architect(self, sample_data):
        if not self.client:
            print("No Groq API key found. Using default procedural parameters")
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
2. Smurfing / Micro-Splitting: Breaking large laundered targets into micro-transactions using Temporal Poisson Smoothing to mimic organic human time-deltas.
3. Commercial Business Hours Evasion: Aligning transaction timestamps with commercial banking hours (09:30 AM to 06:30 PM) to blend seamlessly into active banking traffic.

Respond ONLY with a valid raw JSON object. Do not wrap it in markdown blockquotes like ```json.
The structure MUST match this exactly:
{
    "mule_network": {
        "num_hubs": 25,
        "spokes_per_hub": [10, 50],
        "time_delta_seconds_range": [15, 300]
    },
    "smurfing": {
        "target_amount_range": [150000, 800000],
        "split_count_range": [30, 90],
        "max_micro_amount": 9950,
        "poisson_lambda_sec": 45
    },
    "evasion": {
        "noise_ratio": 0.35,
        "noise_amount_range": [5, 100],
        "business_hours_only": true
    }
}
"""
        user_prompt = f"Here is the sample of the target transaction network. Design the attack parameters based on this data format:\n{sample_json}"
        
        print("Querying Groq LLM Architect for evasive attack parameters")
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
            clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            clean_text = clean_text.replace("```json", "").replace("```", "").strip()
            params = json.loads(clean_text)
            print("LLM Evasive Parameters received successfully")
            return params
        except Exception as e:
            print(f"Groq API Error: {e}. Falling back to default parameters")
            return self._default_attack_params()

    def _default_attack_params(self):
        return {
            "mule_network": {"num_hubs": 50, "spokes_per_hub": [3, 10], "time_delta_seconds_range": [15, 180]},
            "smurfing": {"target_amount_range": [50000, 200000], "split_count_range": [10, 50], "max_micro_amount": 9900, "poisson_lambda_sec": 40},
            "evasion": {"noise_ratio": 0.25, "noise_amount_range": [5, 200], "business_hours_only": True}
        }

    def _get_next_business_timestamp(self, current_dt, seconds_delta):
        import numpy as np
        current_dt += timedelta(seconds=seconds_delta)

        if current_dt.hour < 9 or (current_dt.hour == 9 and current_dt.minute < 30) or current_dt.hour >= 18:
            if current_dt.hour >= 18:
                current_dt += timedelta(days=1)
            jitter_mins = random.randint(0, 45)
            jitter_secs = random.randint(0, 59)
            current_dt = current_dt.replace(hour=9, minute=30, second=0) + timedelta(minutes=jitter_mins, seconds=jitter_secs)
        return current_dt

    def generate_attacks(self, base_df, params, total_target_rows=100_000, max_fraud_ratio=0.4, unnoticed_frauds_df=None):
        import numpy as np
        print(f"Scaling evasive attacks to {total_target_rows:,} transactions (Max Fraud limit: {max_fraud_ratio:.1%})")
        
        accounts = list(set(base_df['Account'].tolist() + base_df['Account.1'].tolist())) if 'Account' in base_df.columns else [f"ACC_{i}" for i in range(10000)]
        currencies = base_df['Receiving Currency'].unique().tolist() if 'Receiving Currency' in base_df.columns else ["USD", "EUR", "GBP"]
        payment_formats = base_df['Payment Format'].unique().tolist() if 'Payment Format' in base_df.columns else ["Wire", "Credit Card", "ACH"]
        
        generated_rows = []
        base_time = datetime.now().replace(hour=10, minute=0, second=0) 
        
        target_fraud_rows = int(total_target_rows * max_fraud_ratio)
        current_fraud_count = 0
        
        mule_p = params.get("mule_network", {})
        num_hubs = mule_p.get("num_hubs", 20)
        
        for _ in range(num_hubs):
            if current_fraud_count >= target_fraud_rows: break
            
            hub_acc = f"MULE_HUB_{random.randint(10000, 99999)}"
            spokes_range = mule_p.get("spokes_per_hub", [5, 10])
            num_spokes = random.randint(spokes_range[0], max(spokes_range[0], spokes_range[1]))
            time_delta_range = mule_p.get("time_delta_seconds_range", [15, 180])
            
            for _ in range(num_spokes):
                if current_fraud_count >= target_fraud_rows: break
                spoke_acc = random.choice(accounts)
                amount = random.uniform(1000, 50000)
                
                poisson_delta = int(np.random.poisson(lam=mule_p.get("poisson_lambda_sec", 45)))
                t_delta = max(time_delta_range[0], min(poisson_delta, time_delta_range[1] * 2))
                base_time = self._get_next_business_timestamp(base_time, t_delta)
                
                generated_rows.append(self._create_row(
                    timestamp=base_time.strftime("%Y/%m/%d %H:%M"),
                    from_acc=hub_acc, to_acc=spoke_acc, amount=amount,
                    currency=random.choice(currencies), p_format=random.choice(payment_formats),
                    is_laundering=1
                ))
                current_fraud_count += 1
                
        smurf_p = params.get("smurfing", {})
        amount_range = smurf_p.get("target_amount_range", [100000, 500000])
        split_range = smurf_p.get("split_count_range", [20, 60])
        max_micro = smurf_p.get("max_micro_amount", 9500)
        poisson_lambda = smurf_p.get("poisson_lambda_sec", 40)
        
        num_smurf_attacks = (target_fraud_rows - current_fraud_count) // max(1, (split_range[0] + split_range[1]) // 2)
        
        for _ in range(max(1, num_smurf_attacks)):
            if current_fraud_count >= target_fraud_rows: break
            
            target_amount = random.uniform(amount_range[0], max(amount_range[0], amount_range[1]))
            num_splits = random.randint(split_range[0], max(split_range[0], split_range[1]))
            source_acc = random.choice(accounts)
            dest_acc = f"OFFSHORE_{random.randint(1000, 9999)}"
            
            for _ in range(num_splits):
                if current_fraud_count >= target_fraud_rows: break
                micro_amount = min(target_amount / num_splits, max_micro) * random.uniform(0.85, 0.99)
                
                poisson_delta = int(np.random.poisson(lam=poisson_lambda))
                base_time = self._get_next_business_timestamp(base_time, max(5, poisson_delta))
                
                generated_rows.append(self._create_row(
                    timestamp=base_time.strftime("%Y/%m/%d %H:%M"),
                    from_acc=source_acc, to_acc=dest_acc, amount=micro_amount,
                    currency=random.choice(currencies), p_format=random.choice(payment_formats),
                    is_laundering=1
                ))
                current_fraud_count += 1

        evasion_p = params.get("evasion", {})
        noise_amount_range = evasion_p.get("noise_amount_range", [10, 500])
        noise_ratio = evasion_p.get("noise_ratio", 0.2)
        
        clean_rows_needed = total_target_rows - len(generated_rows)
        print(f"Injecting {clean_rows_needed:,} real/noise transactions to mask fraud")
        
        for _ in range(clean_rows_needed):
            clean_poisson_delta = int(np.random.poisson(lam=30))
            base_time = self._get_next_business_timestamp(base_time, max(2, clean_poisson_delta))
            is_noise = random.random() < noise_ratio
            
            amount = random.uniform(noise_amount_range[0], max(noise_amount_range[0], noise_amount_range[1])) if is_noise else random.uniform(10, 8000)
            
            generated_rows.append(self._create_row(
                timestamp=base_time.strftime("%Y/%m/%d %H:%M"),
                from_acc=random.choice(accounts), to_acc=random.choice(accounts), amount=amount,
                currency=random.choice(currencies), p_format=random.choice(payment_formats),
                is_laundering=0
            ))
            
        random.shuffle(generated_rows)
        final_df = pd.DataFrame(generated_rows)
        
        output_file = self._get_next_filename()
        final_df.to_csv(output_file, index=False)
        
        actual_fraud = final_df['Is Laundering'].sum()
        actual_clean = len(final_df) - actual_fraud
        print("Attack Generator Complete")
        print(f"Output File: {output_file}")
        print(f"Total Rows : {len(final_df):,}")
        print(f"Distribution -> Fraud: {actual_fraud:,} ({actual_fraud/len(final_df):.1%}), Clean: {actual_clean:,} ({actual_clean/len(final_df):.1%})")
        
        return output_file

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

def run_attack_generator(unnoticed_frauds_df=None):
    RAW_CSV_PATH = os.path.expanduser("~/.cache/kagglehub/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/versions/8/HI-Large_Trans.csv")
    if not os.path.exists(RAW_CSV_PATH):
        RAW_CSV_PATH = "/home/shreyas-nalle/Desktop/Delusional/model/formatted_transactions.csv"
        
    generator = AttackGenerator(raw_data_path=RAW_CSV_PATH)
    
    dynamic_fraud_ratio = round(random.uniform(0.12, 0.58), 2)
    print(f"\n[ATTACK GENERATOR] Dynamic target fraud ratio for this cycle: {dynamic_fraud_ratio * 100:.1f}%")
    
    if unnoticed_frauds_df is not None and not unnoticed_frauds_df.empty:
        print("Using unnoticed frauds to build LLM prompt sample")
        target_rows = 10_000
        num_fraud_needed = int(target_rows * dynamic_fraud_ratio)
        num_clean_needed = target_rows - num_fraud_needed
        
        try:
            chunk_size = 500_000
            raw_df_chunk = pd.read_csv(RAW_CSV_PATH, skiprows=range(1, 5_000_000), nrows=chunk_size)
            if 'Is Laundering' in raw_df_chunk.columns:
                clean_df = raw_df_chunk[raw_df_chunk['Is Laundering'] == 0]
                sampled_fraud = unnoticed_frauds_df.sample(n=num_fraud_needed, replace=True)
                sampled_clean = clean_df.sample(n=num_clean_needed, replace=True)
                sample_df = pd.concat([sampled_clean, sampled_fraud]).sample(frac=1).reset_index(drop=True)
            else:
                sample_df = None
        except Exception as e:
            print(f"Error loading clean data for sample: {e}")
            sample_df = None
    else:
        sample_df = generator.load_and_slice_data(skip_rows=5_000_000, target_rows=10_000, fraud_ratio=dynamic_fraud_ratio)
    
    if sample_df is None:
        print("Using dummy base data since source dataset is missing.")
        sample_df = pd.DataFrame({
            'Account': [f"ACC_{i}" for i in range(100)],
            'Account.1': [f"ACC_{i}" for i in range(100, 200)],
            'Amount Received': [1000.0] * 100,
            'Receiving Currency': ['USD'] * 100,
            'Amount Paid': [1000.0] * 100,
            'Payment Currency': ['USD'] * 100,
            'Payment Format': ['Wire'] * 100,
            'Is Laundering': [0] * 100
        })

    attack_params = generator.prompt_llm_architect(sample_df)
    # Reduce rows from 100k to 15k to prevent OOM (Status 137) on Render Free Tier
    output_file = generator.generate_attacks(sample_df, attack_params, total_target_rows=15000, max_fraud_ratio=dynamic_fraud_ratio)
    return output_file

if __name__ == "__main__":
    run_attack_generator()

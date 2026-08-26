import os
import pandas as pd
import kagglehub

DATASET_HANDLE = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"

def load_aml_dataset(file_name = "HI-Medium_Trans.csv"):
    dataset_dir = kagglehub.dataset_download(DATASET_HANDLE)
    
    csv_file_path = os.path.join(dataset_dir, file_name)
    if not os.path.exists(csv_file_path):
        for root, dirs, files in os.walk(dataset_dir):
            if file_name in files:
                csv_file_path = os.path.join(root, file_name)
                break
                
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file '{file_name}' not found in {dataset_dir}")
        
    print(f"[*] Reading '{csv_file_path}' into pandas DataFrame...")
    df = pd.read_csv(csv_file_path)
    print(f"[+] Successfully loaded DataFrame shape: {df.shape}")
    return df
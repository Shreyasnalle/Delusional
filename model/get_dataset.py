import os
import argparse
import pandas as pd
import kagglehub

DATASET_HANDLE = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"

def load_aml_dataset(file_name = "HI-Medium_Trans.csv") :
    dataset_path = kagglehub.dataset_download(DATASET_HANDLE)
    csv_file_path = os.path.join(dataset_path, file_name)
    if not os.path.exists(csv_file_path) :
        for root, dirs, files in os.walk(dataset_path):
            for f in files:
                if f.endswith('.csv'):
                    print(f"{os.path.join(root, f)}")
        raise FileNotFoundError(f"CSV file not found in the dataset path")
    df = pd.read_csv(csv_file_path)
    return df

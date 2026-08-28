import os
import pandas as pd
import kagglehub

DATASET_HANDLE = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
KAGGLE_CACHE_DIR = os.path.expanduser("~/.cache/kagglehub/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/versions/8")
SAFE_NROWS = 22_000_000

def load_aml_dataset(file_name = "HI-Large_Trans.csv") :
    csv_file_path = os.path.join(KAGGLE_CACHE_DIR, file_name)
    if not os.path.exists(csv_file_path) :
        dataset_dir = kagglehub.dataset_download(DATASET_HANDLE, path = file_name)
        if os.path.isfile(dataset_dir) :
            csv_file_path = dataset_dir 
        else :
            csv_file_path = os.path.join(dataset_dir, file_name) 

    df = pd.read_csv(csv_file_path, nrows = SAFE_NROWS)
    return df
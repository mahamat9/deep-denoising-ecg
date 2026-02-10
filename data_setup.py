import os
import sys
from utils.data_utils import get_optimized_dataset

def setup_data(data_path="./data/", sampling_rate=100):
    """Downloads PTB-XL and creates optimized .npy cache."""
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(os.path.join(data_path, 'ptbxl_database.csv')):
        print("Initiating PTB-XL data download...")
        os.system(f"wget -O {data_path}ptb-xl.zip https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")
        os.system(f"unzip -q {data_path}ptb-xl.zip -d {data_path}")
        os.system(f"mv {data_path}ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/* {data_path}")
        os.system(f"rm -rf {data_path}ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 {data_path}ptb-xl.zip")
        print("Data acquisition complete.")
    else:
        print("PTB-XL data detected.")

    print("Creating optimized .npy cache...")
    _ = get_optimized_dataset(data_path, sampling_rate=sampling_rate, preprocess=False, n_jobs=-1)
    print("Data setup complete.")

if __name__ == "__main__":
    setup_data()

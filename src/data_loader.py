import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.wf_config import CONFIG

def str_to_sinste(fname):
    """Convert filename to (site, instance) tuple"""
    fname = os.path.basename(fname).split('.')[0]
    if '-' in fname:
        site, inst = map(int, fname.split('-'))
        return (site, inst)
    else:
        return (-1, int(fname))  # -1 indicates non-monitored

def load_packet_sequence(file_path):
    """Load a single packet sequence file"""
    try:
        return pd.read_csv(file_path, sep='\t', header=None, 
                         names=['timestamp', 'direction'])
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def load_dataset():
    """Load the entire dataset"""
    X, y, site_ids = [], [], []
    data_path = CONFIG["DATA_PATH"]
    
    # Load monitored sites (0-0 to 99-89)
    for site in tqdm(range(CONFIG["MONITORED_SITES"]), desc="Loading monitored sites"):
        for inst in range(CONFIG["INSTANCES_PER_SITE"]):
            file_path = os.path.join(data_path, f"{site}-{inst}")
            seq = load_packet_sequence(file_path)
            if seq is not None:
                X.append(seq)
                y.append(1)  # 1 = monitored
                site_ids.append(site)
    
    # Load non-monitored sites
    for inst in tqdm(range(CONFIG["NON_MONITORED_SITES"]), desc="Loading non-monitored sites"):
        file_path = os.path.join(data_path, f"{site}")
        seq = load_packet_sequence(file_path)
        if seq is not None:
            X.append(seq)
            y.append(0)  # 0 = non-monitored
            site_ids.append(-1)
    
    return X, np.array(y), np.array(site_ids)
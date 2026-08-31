import os
import shutil
import pandas as pd
import numpy as np
import json

def process_demand():
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Move Delivery_Logistics.csv to data/raw/kaggle_delivery.csv if it's in root
    root_csv = "Delivery_Logistics.csv"
    raw_csv_path = os.path.join(raw_dir, "kaggle_delivery.csv")

    if os.path.exists(root_csv) and not os.path.exists(raw_csv_path):
        print(f"Moving '{root_csv}' to '{raw_csv_path}'...")
        shutil.move(root_csv, raw_csv_path)
    elif os.path.exists(root_csv) and os.path.exists(raw_csv_path):
        print(f"File already exists at '{raw_csv_path}'. Removing duplicate root file.")
        os.remove(root_csv)

    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Missing Kaggle delivery logistics dataset at '{raw_csv_path}'!")

    # Load delivery data
    print(f"Loading delivery logistics dataset from '{raw_csv_path}'...")
    df = pd.read_csv(raw_csv_path)

    # Calculate demand from package_weight_kg (taking ceiling, ensuring min 1)
    df['demand'] = np.ceil(df['package_weight_kg']).astype(int)
    df['demand'] = df['demand'].clip(lower=1)

    # Load node ID map to map random demand records to actual network nodes
    node_map_path = os.path.join(processed_dir, "node_id_map.json")
    if not os.path.exists(node_map_path):
        raise FileNotFoundError(f"Cannot map demands: node_id_map.json not found! Run fetch_osm.py first.")

    with open(node_map_path, "r") as f:
        node_id_map = json.load(f)
    
    node_ids = list(node_id_map.keys())

    # Shuffle node IDs to randomly assign demand to nodes (excluding first node for potential depot)
    np.random.seed(42)
    shuffled_nodes = np.random.permutation(node_ids)
    
    # We sample as many delivery records as we have nodes, or cycle through them if nodes > records
    n_samples = min(len(shuffled_nodes), len(df))
    sampled_df = df.sample(n=n_samples, random_state=42).copy()
    
    # Assign a unique node ID to each sampled record
    sampled_df['node_id'] = shuffled_nodes[:n_samples]

    # Select relevant columns
    demand_df = sampled_df[['node_id', 'demand', 'vehicle_type']].rename(columns={'vehicle_type': 'vehicle_type_hint'})

    demand_csv_path = os.path.join(processed_dir, "demand_vector.csv")
    demand_df.to_csv(demand_csv_path, index=False)
    print(f"Processed demand vector. Sampled {len(demand_df)} records.")
    print(f"Saved demand vector to '{demand_csv_path}'.")

if __name__ == "__main__":
    process_demand()

import os
import requests
import pandas as pd
import numpy as np
import json
import osmnx as ox

def download_pincode_data(pincode_csv_path: str):
    url_options = [
        "https://raw.githubusercontent.com/dropdevrahul/pincodes-india/master/pincode.csv",
        "https://raw.githubusercontent.com/dropdevrahul/pincodes-india/main/pincode.csv"
    ]
    
    for url in url_options:
        try:
            print(f"Attempting to download India Pincode CSV from: {url}...")
            response = requests.get(url, stream=True, timeout=15)
            if response.status_code == 200:
                with open(pincode_csv_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                print("Download complete.")
                return True
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            
    return False

def generate_fallback_pincodes(pincode_csv_path: str):
    print("Generating local fallback pincodes for Kharagpur / Paschim Medinipur...")
    # Creating local mock dataset for Paschim Medinipur, WB
    data = {
        "Circle Name": ["West Bengal Circle"] * 5,
        "Region Name": ["South Bengal Region"] * 5,
        "Division Name": ["Midnapore Division"] * 5,
        "Office Name": ["Kharagpur Technology B.O", "Kharagpur Town S.O", "Nimpura S.O", "Salua B.O", "Inda S.O"],
        "Pincode": [721302, 721301, 721304, 721145, 721305],
        "Office Type": ["Branch Office", "Sub Office", "Sub Office", "Branch Office", "Sub Office"],
        "Delivery status": ["Delivery", "Delivery", "Delivery", "Delivery", "Delivery"],
        "District": ["Paschim Medinipur", "Paschim Medinipur", "Paschim Medinipur", "Paschim Medinipur", "Paschim Medinipur"],
        "State Name": ["WEST BENGAL", "WEST BENGAL", "WEST BENGAL", "WEST BENGAL", "WEST BENGAL"],
        "Latitude": [22.3168, 22.3385, 22.3480, 22.2858, 22.3551],
        "Longitude": [87.3000, 87.3235, 87.2917, 87.2798, 87.3377]
    }
    df = pd.DataFrame(data)
    df.to_csv(pincode_csv_path, index=False)
    print(f"Local fallback pincodes generated at '{pincode_csv_path}'.")

def fetch_and_snap_depots(city_name: str = "Kharagpur, West Bengal, India"):
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    pincode_csv_path = os.path.join(raw_dir, "india_pincode.csv")
    
    # 1. Download if missing
    if not os.path.exists(pincode_csv_path):
        success = download_pincode_data(pincode_csv_path)
        if not success:
            generate_fallback_pincodes(pincode_csv_path)

    # 2. Load and Filter
    print(f"Loading pincode data from '{pincode_csv_path}'...")
    try:
        df = pd.read_csv(pincode_csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}. Re-generating fallback data...")
        generate_fallback_pincodes(pincode_csv_path)
        df = pd.read_csv(pincode_csv_path)

    # Clean column names in case they have spaces or quotes
    df.columns = [col.replace('"', '').strip() for col in df.columns]

    # Map potential column names dynamically (case-insensitive and space-insensitive)
    col_mapping = {}
    for col in df.columns:
        clean_col = col.replace(' ', '').lower()
        if clean_col == 'officename':
            col_mapping[col] = 'OfficeName'
        elif clean_col == 'pincode':
            col_mapping[col] = 'Pincode'
        elif clean_col == 'district':
            col_mapping[col] = 'District'
        elif clean_col == 'statename':
            col_mapping[col] = 'StateName'
        elif clean_col == 'latitude':
            col_mapping[col] = 'Latitude'
        elif clean_col == 'longitude':
            col_mapping[col] = 'Longitude'

    df = df.rename(columns=col_mapping)

    # Clean string data
    for col in ['District', 'StateName', 'OfficeName']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # Filter to Paschim Medinipur / West Midnapore (district for Kharagpur)
    print("Filtering rows for Paschim Medinipur / Kharagpur...")
    mask = df['District'].str.contains('MEDINIPUR|MIDNAPORE', na=False, case=False) | df['OfficeName'].str.contains('KHARAGPUR', na=False, case=False)
    filtered_df = df[mask].copy()

    if len(filtered_df) == 0:
        print("No matches found in Paschim Medinipur. Creating fallback matching records...")
        generate_fallback_pincodes(pincode_csv_path)
        df = pd.read_csv(pincode_csv_path)
        df.columns = [col.replace('"', '').strip() for col in df.columns]
        # Re-apply mapping
        df = df.rename(columns=col_mapping)
        filtered_df = df.copy()

    # 3. Geocode/validate lat-long coordinates
    # Drop rows without latitude/longitude
    filtered_df = filtered_df.dropna(subset=['Latitude', 'Longitude'])

    # Convert coordinates to floats (in case they are read as strings due to quotes)
    filtered_df['Latitude'] = pd.to_numeric(filtered_df['Latitude'], errors='coerce')
    filtered_df['Longitude'] = pd.to_numeric(filtered_df['Longitude'], errors='coerce')
    filtered_df = filtered_df.dropna(subset=['Latitude', 'Longitude'])

    # India Bounding Box: Lat [6, 38], Lon [68, 98]
    print("Validating coordinate bounds...")
    valid_coords_mask = (
        (filtered_df['Latitude'] >= 6.0) & (filtered_df['Latitude'] <= 38.0) &
        (filtered_df['Longitude'] >= 68.0) & (filtered_df['Longitude'] <= 98.0)
    )
    filtered_df = filtered_df[valid_coords_mask].copy()
    print(f"Found {len(filtered_df)} valid geocoded office locations.")

    # 4. Snap each to the nearest node in the strongly connected component road graph
    node_map_path = os.path.join(processed_dir, "node_id_map.json")
    if not os.path.exists(node_map_path):
        raise FileNotFoundError(f"Missing node_id_map.json. Run fetch_osm.py first.")

    with open(node_map_path, "r") as f:
        node_id_map = json.load(f)
    
    # Load cached graphml to snap
    raw_osm_dir = os.path.join("data", "raw", "osm")
    safe_name = "".join(c if c.isalnum() else "_" for c in city_name.lower())
    cache_file = os.path.join(raw_osm_dir, f"{safe_name}.graphml")
    
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Missing graph cache at '{cache_file}'. Run fetch_osm.py first.")
    
    print(f"Loading graph from '{cache_file}' for snapping...")
    G = ox.load_graphml(cache_file)
    
    # Create the subgraph containing only the SCC nodes
    import networkx as nx
    scc_nodes = [int(n) for n in node_id_map.keys()]
    G_scc = G.subgraph(scc_nodes)

    print("Vectorized snapping of depot coordinates to road network...")
    lats = filtered_df['Latitude'].tolist()
    lons = filtered_df['Longitude'].tolist()
    nearest_nodes = ox.nearest_nodes(G_scc, X=lons, Y=lats)

    depot_nodes = []
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        depot_nodes.append({
            "node_id": int(nearest_nodes[i]),
            "pincode": int(row['Pincode']),
            "office_name": row['OfficeName'],
            "latitude": float(row['Latitude']),
            "longitude": float(row['Longitude'])
        })

    # Save to depot_nodes.csv
    depot_df = pd.DataFrame(depot_nodes)
    # Deduplicate by node_id
    depot_df = depot_df.drop_duplicates(subset=['node_id'])
    
    depot_csv_path = os.path.join(processed_dir, "depot_nodes.csv")
    depot_df.to_csv(depot_csv_path, index=False)
    print(f"Successfully snapped {len(depot_df)} unique depot nodes and saved to '{depot_csv_path}'.")

if __name__ == "__main__":
    fetch_and_snap_depots()

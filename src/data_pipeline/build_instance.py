import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

def build_instances(seed: int = 42):
    processed_dir = os.path.join("data", "processed")
    instances_dir = os.path.join("data", "instances")
    os.makedirs(instances_dir, exist_ok=True)

    # 1. Load processed data files
    depot_nodes_path = os.path.join(processed_dir, "depot_nodes.csv")
    demand_vector_path = os.path.join(processed_dir, "demand_vector.csv")
    node_id_map_path = os.path.join(processed_dir, "node_id_map.json")
    nodes_metadata_path = os.path.join(processed_dir, "nodes_metadata.json")
    cost_params_path = os.path.join(processed_dir, "cost_params.json")
    demand_nodes_path = os.path.join(processed_dir, "demand_nodes.json")

    for path in [depot_nodes_path, demand_vector_path, node_id_map_path, nodes_metadata_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing processed file: {path}. Run earlier pipeline scripts first.")

    depot_df = pd.read_csv(depot_nodes_path)
    demand_df = pd.read_csv(demand_vector_path)
    with open(node_id_map_path, "r") as f:
        node_id_map = json.load(f)
    with open(nodes_metadata_path, "r") as f:
        nodes_metadata = json.load(f)

    # Optional rich metadata
    cost_params = {}
    if os.path.exists(cost_params_path):
        with open(cost_params_path, "r") as f:
            cost_params = json.load(f)

    demand_json_customers = {}
    if os.path.exists(demand_nodes_path):
        with open(demand_nodes_path, "r") as f:
            d_data = json.load(f)
            for c in d_data.get("customers", []):
                demand_json_customers[c["graph_node"]] = c

    # 2. Select the primary depot
    depot_row = depot_df.iloc[0]
    depot_node_id = int(depot_row['node_id'])
    depot_lat = float(depot_row['latitude'])
    depot_lon = float(depot_row['longitude'])
    depot_name = str(depot_row.get('name', f"Depot_{depot_node_id}"))
    depot_cat = str(depot_row.get('category', 'building:warehouse'))
    depot_cap = float(depot_row.get('capacity_kg', 5000))

    if str(depot_node_id) not in node_id_map:
        depot_node_id = int(list(node_id_map.keys())[0])
        depot_lat = float(nodes_metadata[str(depot_node_id)]['lat'])
        depot_lon = float(nodes_metadata[str(depot_node_id)]['lon'])
        print(f"Warning: Depot node was not in SCC. Falling back to node {depot_node_id}")

    print(f"Selected depot node ID: {depot_node_id} ({depot_name}) (lat: {depot_lat:.4f}, lon: {depot_lon:.4f})")

    # 3. Filter customer candidates
    demand_df['node_id_str'] = demand_df['node_id'].astype(str)
    customer_candidates = demand_df[
        (demand_df['node_id_str'].isin(node_id_map.keys())) & 
        (demand_df['node_id'] != depot_node_id)
    ].copy()

    print(f"Number of valid customer candidates in SCC: {len(customer_candidates)}")

    # 4. Generate instances for N = 20, 50, 100
    np.random.seed(seed)
    sizes = [20, 50, 100]

    for size in sizes:
        if len(customer_candidates) < size:
            print(f"Warning: Not enough customer candidates for N={size}. Sampling extra graph nodes...")
            scc_nodes = [int(nid) for nid in node_id_map.keys() if int(nid) != depot_node_id]
            extra_needed = size - len(customer_candidates)
            extra_nodes = np.random.choice(scc_nodes, size=extra_needed, replace=False)
            extra_records = []
            for enode in extra_nodes:
                extra_records.append({
                    "node_id": enode,
                    "demand": int(np.random.randint(5, 30)),
                    "demand_units": int(np.random.randint(1, 5)),
                    "time_window_start": int(np.random.randint(360, 600)),
                    "time_window_end": int(np.random.randint(660, 1200)),
                    "service_time_min": int(np.random.randint(3, 10)),
                    "vehicle_type_hint": "two_wheeler",
                    "poi_category": "amenity:commercial_stop",
                    "customer_name": f"Customer_{enode}",
                    "node_id_str": str(enode)
                })
            extra_df = pd.DataFrame(extra_records)
            customer_candidates = pd.concat([customer_candidates, extra_df], ignore_index=True)

        sampled_customers = customer_candidates.sample(n=size, random_state=seed).copy()

        customers_list = []
        for _, row in sampled_customers.iterrows():
            cid = int(row['node_id'])
            demand = int(row['demand'])
            coords = nodes_metadata[str(cid)]

            # Check if we have rich details in demand_nodes.json
            rich = demand_json_customers.get(cid, {})
            tw_start = int(row.get('time_window_start', rich.get('time_window', [480, 720])[0]))
            tw_end = int(row.get('time_window_end', rich.get('time_window', [480, 720])[1]))
            serv_time = int(row.get('service_time_min', rich.get('service_time_min', 5)))
            units = int(row.get('demand_units', rich.get('demand_units', 1)))
            cat = str(row.get('poi_category', rich.get('category', 'amenity:customer_poi')))
            cname = str(row.get('customer_name', rich.get('name', f"Stop_{cid}")))

            customers_list.append({
                "node_id": cid,
                "demand": demand,
                "demand_kg": float(demand),
                "demand_units": units,
                "lat": float(coords['lat']),
                "lon": float(coords['lon']),
                "time_window": [tw_start, tw_end],
                "service_time_min": serv_time,
                "category": cat,
                "name": cname
            })

        # Set VRP configuration
        if size == 20:
            num_vehicles = 5
            vehicle_capacity = 100
        elif size == 50:
            num_vehicles = 8
            vehicle_capacity = 120
        else:
            num_vehicles = 15
            vehicle_capacity = 150

        instance_data = {
            "instance_id": f"instance_n{size}",
            "depot": {
                "node_id": depot_node_id,
                "lat": depot_lat,
                "lon": depot_lon,
                "name": depot_name,
                "category": depot_cat,
                "capacity_kg": depot_cap
            },
            "customers": customers_list,
            "vehicle_capacity": vehicle_capacity,
            "num_vehicles": num_vehicles,
            "distance_matrix": "ref:data/processed/distance_matrix.npy",
            "travel_time_matrix": "ref:data/processed/travel_time_matrix.npy",
            "node_id_map": "ref:data/processed/node_id_map.json",
            "fleet": cost_params.get("fleet", []),
            "cost_weights": cost_params.get("cost_weights", {}),
            "time_aware_weights": "ref:data/processed/time_aware_weights.json"
        }

        instance_file_path = os.path.join(instances_dir, f"instance_n{size}.json")
        with open(instance_file_path, "w") as f:
            json.dump(instance_data, f, indent=4)
        print(f"Generated and saved instance {size} to '{instance_file_path}' (depot: {depot_node_id}, customers: {size}, vehicles: {num_vehicles}, capacity: {vehicle_capacity})")

if __name__ == "__main__":
    build_instances()

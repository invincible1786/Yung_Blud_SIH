import os
import json
import numpy as np
import pandas as pd

def build_instances(seed: int = 42):
    processed_dir = os.path.join("data", "processed")
    instances_dir = os.path.join("data", "instances")
    os.makedirs(instances_dir, exist_ok=True)

    # 1. Load processed data files
    depot_nodes_path = os.path.join(processed_dir, "depot_nodes.csv")
    demand_vector_path = os.path.join(processed_dir, "demand_vector.csv")
    node_id_map_path = os.path.join(processed_dir, "node_id_map.json")
    nodes_metadata_path = os.path.join(processed_dir, "nodes_metadata.json")

    for path in [depot_nodes_path, demand_vector_path, node_id_map_path, nodes_metadata_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing processed file: {path}. Run earlier pipeline scripts first.")

    depot_df = pd.read_csv(depot_nodes_path)
    demand_df = pd.read_csv(demand_vector_path)
    with open(node_id_map_path, "r") as f:
        node_id_map = json.load(f)
    with open(nodes_metadata_path, "r") as f:
        nodes_metadata = json.load(f)

    # 2. Select the depot
    # We will pick the first depot in the list as the primary depot
    depot_node_id = int(depot_df.iloc[0]['node_id'])
    depot_lat = float(depot_df.iloc[0]['latitude'])
    depot_lon = float(depot_df.iloc[0]['longitude'])

    # Ensure depot node ID is valid and exists in our SCC node mapping
    if str(depot_node_id) not in node_id_map:
        # Fallback: pick any node from node_id_map as depot
        depot_node_id = int(list(node_id_map.keys())[0])
        depot_lat = float(nodes_metadata[str(depot_node_id)]['lat'])
        depot_lon = float(nodes_metadata[str(depot_node_id)]['lon'])
        print(f"Warning: Depot node was not in SCC. Falling back to node {depot_node_id}")

    print(f"Selected depot node ID: {depot_node_id} (lat: {depot_lat:.4f}, lon: {depot_lon:.4f})")

    # 3. Filter customer candidates
    # Customers must have demand, exist in SCC (node_id_map), and not be the depot
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
            print(f"Warning: Not enough customer candidates for N={size}. We will sample with replacement or assign random demands to other graph nodes.")
            # Fallback: sample other graph nodes and assign random demand
            scc_nodes = [int(nid) for nid in node_id_map.keys() if int(nid) != depot_node_id]
            extra_needed = size - len(customer_candidates)
            extra_nodes = np.random.choice(scc_nodes, size=extra_needed, replace=False)
            extra_records = []
            for enode in extra_nodes:
                # Random demand between 5 and 30 kg
                extra_records.append({
                    "node_id": enode,
                    "demand": int(np.random.randint(5, 30)),
                    "node_id_str": str(enode)
                })
            extra_df = pd.DataFrame(extra_records)
            customer_candidates = pd.concat([customer_candidates, extra_df], ignore_index=True)

        # Sample customer nodes
        sampled_customers = customer_candidates.sample(n=size, random_state=seed).copy()

        customers_list = []
        for _, row in sampled_customers.iterrows():
            cid = int(row['node_id'])
            demand = int(row['demand'])
            coords = nodes_metadata[str(cid)]
            customers_list.append({
                "node_id": cid,
                "demand": demand,
                "lat": float(coords['lat']),
                "lon": float(coords['lon'])
            })

        # Set VRP configuration
        # For N=20: 5 vehicles, capacity 100
        # For N=50: 8 vehicles, capacity 120
        # For N=100: 15 vehicles, capacity 150
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
                "lon": depot_lon
            },
            "customers": customers_list,
            "vehicle_capacity": vehicle_capacity,
            "num_vehicles": num_vehicles,
            "distance_matrix": "ref:data/processed/distance_matrix.npy",
            "node_id_map": "ref:data/processed/node_id_map.json"
        }

        instance_file_path = os.path.join(instances_dir, f"instance_n{size}.json")
        with open(instance_file_path, "w") as f:
            json.dump(instance_data, f, indent=4)
        print(f"Generated and saved instance {size} to '{instance_file_path}' (depot: {depot_node_id}, customers: {size}, vehicles: {num_vehicles}, capacity: {vehicle_capacity})")

if __name__ == "__main__":
    build_instances()

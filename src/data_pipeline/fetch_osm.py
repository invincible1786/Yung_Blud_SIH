import os
import json
import numpy as np
import networkx as nx
import osmnx as ox
from tqdm import tqdm

def get_cache_path(city_name: str, cache_dir: str) -> str:
    safe_name = "".join(c if c.isalnum() else "_" for c in city_name.lower())
    return os.path.join(cache_dir, f"{safe_name}.graphml")

def fetch_and_process_road_network(city_name: str = "Kharagpur, West Bengal, India", radius: float = 3000):
    raw_osm_dir = os.path.join("data", "raw", "osm")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(raw_osm_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    cache_file = get_cache_path(city_name, raw_osm_dir)

    # 1. Fetch or Load Graph
    if os.path.exists(cache_file):
        print(f"Loading cached graph from '{cache_file}'...")
        G = ox.load_graphml(cache_file)
    else:
        print(f"Fetching road network within {radius}m of '{city_name}'...")
        G = ox.graph_from_address(city_name, dist=radius, network_type="drive")
        # Add edge speeds and travel times
        G = ox.add_edge_speeds(G, fallback=30)  # default speed limit of 30 km/h
        G = ox.add_edge_travel_times(G)
        ox.save_graphml(G, cache_file)
        print(f"Saved graph cache to '{cache_file}'.")

    # 2. Extract strongly connected component (SCC) to guarantee connectivity
    print("Extracting the largest strongly connected component...")
    scc_nodes = max(nx.strongly_connected_components(G), key=len)
    G_scc = G.subgraph(scc_nodes).copy()
    print(f"Graph nodes reduced from {G.number_of_nodes()} to {G_scc.number_of_nodes()} (SCC size)")

    nodes = list(G_scc.nodes())
    node_id_map = {int(node_id): idx for idx, node_id in enumerate(nodes)}

    # Save node ID mapping
    node_map_path = os.path.join(processed_dir, "node_id_map.json")
    with open(node_map_path, "w") as f:
        json.dump(node_id_map, f, indent=4)
    print(f"Saved node ID mapping to '{node_map_path}'.")

    # 3. Compute Distance and Travel-Time Matrices
    n_nodes = len(nodes)
    print(f"Computing shortest path matrices ({n_nodes} x {n_nodes})...")
    dist_matrix = np.full((n_nodes, n_nodes), np.inf, dtype=np.float32)
    time_matrix = np.full((n_nodes, n_nodes), np.inf, dtype=np.float32)

    # Set diagonals to 0
    np.fill_diagonal(dist_matrix, 0)
    np.fill_diagonal(time_matrix, 0)

    # We use Dijkstra for all-pairs shortest paths
    for node in tqdm(nodes, desc="Dijkstra sweeps"):
        u_idx = node_id_map[int(node)]
        
        # Distances
        lengths = nx.single_source_dijkstra_path_length(G_scc, node, weight='length')
        for target, length in lengths.items():
            v_idx = node_id_map[int(target)]
            dist_matrix[u_idx, v_idx] = length

        # Travel times
        times = nx.single_source_dijkstra_path_length(G_scc, node, weight='travel_time')
        for target, time_val in times.items():
            v_idx = node_id_map[int(target)]
            time_matrix[u_idx, v_idx] = time_val

    # 4. Sanity checks
    print("Performing sanity checks on matrices...")
    assert not np.isnan(dist_matrix).any(), "Distance matrix contains NaNs!"
    assert not np.isinf(dist_matrix).any(), "Distance matrix contains Infs!"
    assert not np.isnan(time_matrix).any(), "Travel time matrix contains NaNs!"
    assert not np.isinf(time_matrix).any(), "Travel time matrix contains Infs!"

    # Check symmetry and log difference
    sym_diff = np.abs(dist_matrix - dist_matrix.T)
    max_sym_diff = np.max(sym_diff)
    print(f"Max asymmetry in distance matrix (due to one-way streets): {max_sym_diff:.2f} meters")

    # Save matrices
    dist_matrix_path = os.path.join(processed_dir, "distance_matrix.npy")
    time_matrix_path = os.path.join(processed_dir, "travel_time_matrix.npy")
    np.save(dist_matrix_path, dist_matrix)
    np.save(time_matrix_path, time_matrix)
    print(f"Saved distance matrix to '{dist_matrix_path}'.")
    print(f"Saved travel time matrix to '{time_matrix_path}'.")

    # Save G_scc nodes with metadata for coordinate lookup
    nodes_metadata = {}
    for node_id in nodes:
        node_data = G_scc.nodes[node_id]
        nodes_metadata[int(node_id)] = {
            "lat": node_data.get("y"),
            "lon": node_data.get("x")
        }
    metadata_path = os.path.join(processed_dir, "nodes_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(nodes_metadata, f, indent=4)
    print(f"Saved nodes metadata coordinates to '{metadata_path}'.")

if __name__ == "__main__":
    fetch_and_process_road_network()

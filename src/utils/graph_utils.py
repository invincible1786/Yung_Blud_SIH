import os
import json
import numpy as np

def load_instance_resources(instance: dict):
    """Loads the distance matrix and node_id_map referenced in the VRP instance."""
    dist_ref = instance["distance_matrix"]
    map_ref = instance["node_id_map"]
    
    dist_path = dist_ref.replace("ref:", "")
    map_path = map_ref.replace("ref:", "")
    
    # Ensure paths are correct
    if not os.path.exists(dist_path):
        # Fallback to absolute or relative search
        dist_path = os.path.join(os.getcwd(), dist_path)
        map_path = os.path.join(os.getcwd(), map_path)
        
    with open(map_path, "r") as f:
        node_id_map = json.load(f)
        
    # Convert keys to integers in node_id_map since json saves keys as strings
    node_id_map = {int(k): int(v) for k, v in node_id_map.items()}
    
    distance_matrix = np.load(dist_path)
    
    return distance_matrix, node_id_map

def calculate_route_distance(route: list, distance_matrix, node_id_map) -> float:
    """Calculates the total distance of a route (list of node IDs starting and ending at depot)."""
    if len(route) <= 1:
        return 0.0
    
    dist = 0.0
    for i in range(len(route) - 1):
        u = node_id_map[route[i]]
        v = node_id_map[route[i+1]]
        dist += distance_matrix[u, v]
    return float(dist)

def calculate_total_distance(routes: list, distance_matrix, node_id_map) -> float:
    """Calculates the total distance across all routes."""
    return sum(calculate_route_distance(r, distance_matrix, node_id_map) for r in routes)

"""
Static OSM Route Map Visualizer for VRP Solutions.
Renders VRP vehicle routes directly onto real OpenStreetMap road networks using OSMnx.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.algorithms.base import SolutionResult
from src.algorithms.nn_clarke_wright import NearestNeighborClarkeWright
from src.algorithms.standard_pso import StandardPSO
from src.algorithms.qpso import QPSO
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.aco_mmas import ACOMMAS
from src.algorithms.qpso_optimized import QPSOOptimized
from src.benchmark.runner import get_algo_key, ALGO_KEY_MAP

# Graph default search paths
DEFAULT_GRAPH_PATHS = [
    os.path.join("data", "raw", "osm", "kharagpur_west_bengal_india.graphml"),
    os.path.join("data", "raw", "osm", "kharagpur__west_bengal__india.graphml"),
]

ALGORITHM_CLASSES = {
    "Nearest Neighbor": (NearestNeighborClarkeWright, {"method": "nn", "max_iter": 100}),
    "Clarke-Wright": (NearestNeighborClarkeWright, {"method": "clarke_wright", "max_iter": 100}),
    "Standard PSO": (StandardPSO, {"swarm_size": 30, "max_iter": 100, "w": 0.7, "c1": 1.5, "c2": 1.5}),
    "QPSO": (QPSO, {"swarm_size": 30, "max_iter": 100, "beta_start": 1.0, "beta_end": 0.5}),
    "Genetic Algorithm": (GeneticAlgorithm, {"population_size": 30, "max_generations": 100, "crossover_rate": 0.8, "mutation_rate": 0.2, "apply_2opt": True}),
    "Ant Colony (MMAS)": (ACOMMAS, {"num_ants": 15, "max_iter": 100, "alpha": 1.0, "beta": 3.0, "evaporation_rate": 0.1}),
    "QPSO-Optimized": (QPSOOptimized, {"swarm_size": 30, "max_iter": 100, "inter_route_2opt_star": True})
}


def load_default_graph(graph_path: Optional[str] = None):
    """
    Loads the cached OSM road network graphml file.
    
    Args:
        graph_path: Optional custom path to graphml file.
        
    Returns:
        nx.MultiDiGraph: OSMnx road network graph.
    """
    if graph_path and os.path.exists(graph_path):
        return ox.load_graphml(graph_path)

    for path in DEFAULT_GRAPH_PATHS:
        if os.path.exists(path):
            return ox.load_graphml(path)

    raise FileNotFoundError(
        f"OSM road network graphml not found. Checked: {DEFAULT_GRAPH_PATHS}"
    )


def convert_routes_to_osmnx_paths(graph, routes: List[List[int]]) -> List[List[int]]:
    """
    Converts a list of VRP routes (sequences of stop node IDs) into
    contiguous OSMnx node paths by calculating shortest road network paths
    between consecutive stops.
    
    Args:
        graph: OSMnx MultiDiGraph.
        routes: List of routes, where each route is [depot, stop1, stop2, ..., depot].
        
    Returns:
        List of detailed node ID lists traversing intermediate road graph junctions.
    """
    detailed_routes = []
    for r in routes:
        if not r or len(r) <= 1:
            continue
            
        d_path = []
        for i in range(len(r) - 1):
            u = int(r[i])
            v = int(r[i + 1])
            if u == v:
                continue
                
            try:
                sub_path = nx.shortest_path(graph, u, v, weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                try:
                    sub_path = nx.shortest_path(graph, u, v, weight="travel_time")
                except Exception:
                    sub_path = [u, v]
                    
            if not d_path:
                d_path.extend(sub_path)
            else:
                d_path.extend(sub_path[1:])
                
        if len(d_path) > 1:
            detailed_routes.append(d_path)

    return detailed_routes


def load_solution_for_pair(
    instance_id: str,
    algorithm_name: str,
    instance: Optional[Dict[str, Any]] = None
) -> SolutionResult:
    """
    Loads SolutionResult for a given (instance_id, algorithm_name) pair.
    Prefers results/logs/best_routes.json; falls back to results_raw.csv and re-solving.
    """
    algo_slug = get_algo_key(algorithm_name)
    best_routes_path = os.path.join("results", "logs", "best_routes.json")

    # 1. Attempt load from best_routes.json
    if os.path.exists(best_routes_path):
        try:
            with open(best_routes_path, "r") as f:
                data = json.load(f)
            if instance_id in data:
                # Check slug and display name
                entry = data[instance_id].get(algo_slug) or data[instance_id].get(algorithm_name)
                if entry and "routes" in entry:
                    return SolutionResult(
                        routes=entry["routes"],
                        total_cost=float(entry.get("cost", entry.get("penalized_cost", entry.get("distance", 0.0)))),
                        convergence_history=[],
                        runtime_seconds=0.0
                    )
        except Exception as e:
            print(f"Warning: Could not read from best_routes.json ({e}). Falling back to re-solve.")

    # 1b. Fallback to results/qpso_study/best_routes.json (e.g. for QPSO-Optimized)
    qpso_routes_path = os.path.join("results", "qpso_study", "best_routes.json")
    if os.path.exists(qpso_routes_path):
        try:
            with open(qpso_routes_path, "r") as f:
                qdata = json.load(f)
            if instance_id in qdata:
                entry = (
                    qdata[instance_id].get(algo_slug)
                    or qdata[instance_id].get(algorithm_name)
                    or qdata[instance_id].get("qpso_optimized")
                )
                if entry and "routes" in entry:
                    return SolutionResult(
                        routes=entry["routes"],
                        total_cost=float(entry.get("cost", entry.get("penalized_cost", entry.get("distance", 0.0)))),
                        convergence_history=[],
                        runtime_seconds=0.0
                    )
        except Exception as e:
            print(f"Warning: Could not read from qpso_study best_routes.json ({e}).")

    # 2. Re-run best seed using results_raw.csv or seed 1
    raw_csv_path = os.path.join("results", "logs", "results_raw.csv")
    best_seed = 1

    if os.path.exists(raw_csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(raw_csv_path)
            matches = df[(df["instance_id"] == instance_id) & (df["algorithm"].str.lower() == algorithm_name.lower())]
            if not matches.empty:
                best_seed = int(matches.loc[matches["total_cost"].idxmin()]["seed"])
        except Exception:
            best_seed = 1

    if instance is None:
        inst_path = os.path.join("data", "instances", f"{instance_id}.json")
        with open(inst_path, "r") as f:
            instance = json.load(f)

    # Find matching solver class
    matched_cls = None
    matched_cfg = None
    for name, (cls_ref, cfg) in ALGORITHM_CLASSES.items():
        if name.lower() == algorithm_name.lower() or get_algo_key(name) == algo_slug:
            matched_cls = cls_ref
            matched_cfg = cfg
            break

    if matched_cls is None:
        raise ValueError(f"Unknown algorithm name: '{algorithm_name}'")

    print(f"Re-solving {algorithm_name} on {instance_id} with seed {best_seed}...")
    np.random.seed(best_seed)
    solver = matched_cls(instance, matched_cfg)
    return solver.solve()


def plot_algorithm_routes(
    graph,
    instance: dict,
    solution: SolutionResult,
    algorithm_name: str,
    save_path: str
) -> None:
    """
    Renders vehicle routing solution on the road network using OSMnx.

    Args:
        graph: OSMnx MultiDiGraph (or None to load default).
        instance: VRP instance dictionary with depot and customer metadata.
        solution: SolutionResult containing .routes (and .total_cost).
        algorithm_name: Name of the algorithm (e.g., 'QPSO', 'Genetic Algorithm').
        save_path: Output file path for the generated static image.
    """
    if graph is None:
        graph = load_default_graph()

    if isinstance(instance, str):
        # Allow passing instance path or ID
        if os.path.exists(instance):
            with open(instance, "r") as f:
                instance = json.load(f)
        else:
            inst_path = os.path.join("data", "instances", f"{instance}.json")
            with open(inst_path, "r") as f:
                instance = json.load(f)

    # Convert routes to OSMnx continuous node paths
    routes = solution.routes if hasattr(solution, "routes") else solution.get("routes", [])
    total_cost = solution.total_cost if hasattr(solution, "total_cost") else solution.get("cost", 0.0)
    detailed_routes = convert_routes_to_osmnx_paths(graph, routes)

    num_routes = len(detailed_routes)
    if num_routes == 0:
        print(f"Warning: No valid vehicle routes found to plot for {algorithm_name}.")
        return

    # Qualitative colormap for vehicle routes (tab10 or Set2 or tab20)
    if num_routes <= 10:
        cmap = plt.get_cmap("tab10")
        route_colors = [matplotlib.colors.to_hex(cmap(i)) for i in range(num_routes)]
    elif num_routes <= 20:
        cmap = plt.get_cmap("tab20")
        route_colors = [matplotlib.colors.to_hex(cmap(i)) for i in range(num_routes)]
    else:
        cmap = plt.get_cmap("gist_rainbow")
        route_colors = [matplotlib.colors.to_hex(cmap(i / num_routes)) for i in range(num_routes)]

    # Draw all vehicle routes on the road network using OSMnx
    fig, ax = ox.plot_graph_routes(
        graph,
        detailed_routes,
        route_colors=route_colors,
        route_linewidths=3.0,
        orig_dest_size=0,         # Suppress default start/end points to draw custom styled markers
        node_size=0,              # Hide regular road intersection dots for visual clarity
        edge_color="#b8b8b8",     # Light gray road background
        edge_linewidth=0.55,
        bgcolor="#ffffff",
        show=False,
        close=False
    )

    # 1. Plot Depot with distinct large marker (red star with black edge)
    depot_info = instance["depot"]
    depot_nid = int(depot_info["node_id"])
    if depot_nid in graph.nodes:
        depot_x = graph.nodes[depot_nid]["x"]
        depot_y = graph.nodes[depot_nid]["y"]
    else:
        depot_x = depot_info.get("lon")
        depot_y = depot_info.get("lat")

    ax.scatter(
        depot_x,
        depot_y,
        marker="*",
        s=480,
        color="#e63946",
        edgecolors="black",
        linewidths=1.6,
        zorder=15,
        label="Depot"
    )

    # 2. Plot Customer nodes marked with small circles, sized & colored by demand
    customers = instance.get("customers", [])
    cust_x, cust_y, demands = [], [], []
    for c in customers:
        nid = int(c["node_id"])
        if nid in graph.nodes:
            x = graph.nodes[nid]["x"]
            y = graph.nodes[nid]["y"]
        else:
            x = c.get("lon")
            y = c.get("lat")
        cust_x.append(x)
        cust_y.append(y)
        demands.append(float(c.get("demand", 10.0)))

    if cust_x:
        demands_arr = np.array(demands)
        min_d, max_d = float(np.min(demands_arr)), float(np.max(demands_arr))
        range_d = max_d - min_d if max_d > min_d else 1.0
        # Scaled circle sizes (from 45 to 160)
        sizes = 45.0 + ((demands_arr - min_d) / range_d) * 115.0

        sc = ax.scatter(
            cust_x,
            cust_y,
            s=sizes,
            c=demands_arr,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.85,
            alpha=0.92,
            zorder=12,
            label="Customers"
        )

        # Add colorbar for customer demand
        cbar = fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.55, pad=0.02)
        cbar.set_label("Customer Demand (units / kg)", fontsize=9.5, fontweight="bold")
        cbar.ax.tick_params(labelsize=8.5)

    # 3. Add Legend for Depot and Vehicle Routes
    legend_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#e63946", markeredgecolor="k", markersize=16, label="Depot")
    ]
    for idx, col in enumerate(route_colors):
        legend_handles.append(
            Line2D([0], [0], color=col, lw=3.2, label=f"Vehicle {idx + 1}")
        )

    # Determine legend column count
    ncol = 1
    if len(legend_handles) > 12:
        ncol = 3
    elif len(legend_handles) > 6:
        ncol = 2

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=ncol,
        frameon=True,
        facecolor="#ffffff",
        framealpha=0.92,
        fontsize=8.5
    )

    # 4. Title with algorithm name, instance size, total cost, and vehicle count
    instance_id = instance.get("instance_id", "Instance")
    num_customers = len(customers)
    title_text = (
        f"{algorithm_name} — {instance_id} (N={num_customers})\n"
        f"Total Cost: {total_cost:,.2f} m | Vehicles Used: {num_routes}"
    )
    ax.set_title(title_text, fontsize=13, fontweight="bold", pad=12)

    # Ensure output directory exists and save
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved route map to: '{save_path}'")


def generate_all_static_maps(
    instance_ids: Optional[List[str]] = None,
    algorithm_names: Optional[List[str]] = None,
    output_dir: str = os.path.join("results", "route_maps", "static")
):
    """
    Batch generates static OSM route maps for all combinations of instances and algorithms.
    """
    graph = load_default_graph()

    instances_dir = os.path.join("data", "instances")
    if instance_ids is None:
        inst_files = [f.replace(".json", "") for f in os.listdir(instances_dir) if f.endswith(".json")]
        inst_files.sort()
        instance_ids = inst_files

    if algorithm_names is None:
        algorithm_names = list(ALGORITHM_CLASSES.keys())

    os.makedirs(output_dir, exist_ok=True)

    for inst_id in instance_ids:
        inst_path = os.path.join(instances_dir, f"{inst_id}.json")
        if not os.path.exists(inst_path):
            print(f"Skipping unknown instance: {inst_id}")
            continue

        with open(inst_path, "r") as f:
            instance = json.load(f)

        for algo in algorithm_names:
            algo_slug = get_algo_key(algo)
            save_name = f"{inst_id}_{algo_slug}.png"
            save_path = os.path.join(output_dir, save_name)

            try:
                solution = load_solution_for_pair(inst_id, algo, instance=instance)
                plot_algorithm_routes(
                    graph=graph,
                    instance=instance,
                    solution=solution,
                    algorithm_name=algo,
                    save_path=save_path
                )
            except Exception as e:
                print(f"Error plotting {inst_id} - {algo}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate static OSM route map visualizations for VRP solutions.")
    parser.add_argument("--instance", type=str, default="instance_n20", help="Instance ID (e.g. instance_n20)")
    parser.add_argument("--algorithm", type=str, default="QPSO", help="Algorithm name (e.g. QPSO, 'Genetic Algorithm')")
    parser.add_argument("--all", action="store_true", help="Generate route maps for all instances and algorithms")
    parser.add_argument("--output_dir", type=str, default=os.path.join("results", "route_maps", "static"), help="Output directory")

    args = parser.parse_args()

    if args.all:
        generate_all_static_maps(output_dir=args.output_dir)
    else:
        inst_id = args.instance
        algo = args.algorithm
        algo_slug = get_algo_key(algo)
        out_file = os.path.join(args.output_dir, f"{inst_id}_{algo_slug}.png")
        
        G = load_default_graph()
        inst_path = os.path.join("data", "instances", f"{inst_id}.json")
        with open(inst_path, "r") as f:
            inst = json.load(f)
            
        sol = load_solution_for_pair(inst_id, algo, instance=inst)
        plot_algorithm_routes(G, inst, sol, algo, out_file)

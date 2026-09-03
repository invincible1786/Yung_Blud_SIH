"""
Side-by-Side Comparison Grid Visualizer for VRP Solutions.
Generates a 2x3 Matplotlib subplot grid comparing all 6 VRP algorithms
on real OpenStreetMap road networks with shared styling, scales, and metrics.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import networkx as nx
import osmnx as ox
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.algorithms.base import SolutionResult
from src.benchmark.runner import get_algo_key
from src.visualization.static_map import (
    load_default_graph,
    load_solution_for_pair,
    convert_routes_to_osmnx_paths,
)
from src.visualization.interactive_map import (
    load_nodes_metadata,
    get_node_lat_lon,
    PALETTE,
    KHARAGPUR_CENTER,
)

# Ordered layout so Genetic Algorithm is at bottom-left (row 1, col 0)
GRID_ALGORITHMS = [
    ("Nearest Neighbor", "Nearest Neighbor"),
    ("Clarke-Wright", "Clarke-Wright"),
    ("Standard PSO", "Standard PSO"),
    ("Genetic Algorithm", "Genetic Algorithm"),  # Bottom-Left (row 1, col 0)
    ("QPSO", "QPSO"),
    ("Ant Colony (MMAS)", "Ant Colony (MMAS)"),
]


def load_optimality_gaps(summary_csv_path: str = os.path.join("results", "logs", "results_summary.csv")) -> Dict[Tuple[str, str], float]:
    """
    Loads optimality gap percentages from results_summary.csv.
    Returns a dict mapping (instance_id, algorithm_name) -> gap_percent.
    """
    gaps = {}
    if os.path.exists(summary_csv_path):
        try:
            df = pd.read_csv(summary_csv_path)
            for _, row in df.iterrows():
                inst = str(row["instance_id"]).strip()
                algo = str(row["algorithm"]).strip()
                gap = float(row.get("optimality_gap_percent", 0.0))
                gaps[(inst, algo)] = gap
                # Also map slug
                gaps[(inst, get_algo_key(algo))] = gap
        except Exception as e:
            print(f"Warning: Could not read optimality gaps from {summary_csv_path} ({e})")
    return gaps


def plot_comparison_grid(
    instance_id: str,
    graph: Optional[nx.MultiDiGraph] = None,
    instance: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    summary_gaps: Optional[Dict[Tuple[str, str], float]] = None
) -> str:
    """
    Creates a 2x3 Matplotlib subplot grid comparing all 6 VRP algorithms for a given instance.

    Args:
        instance_id: Identifier of the instance (e.g., 'instance_n20', 'instance_n50').
        graph: Pre-loaded OSMnx graph (or None to load default).
        instance: Pre-loaded instance dict (or None to load from data/instances/).
        save_path: Destination PNG path (default: results/route_maps/comparison_grids/comparison_grid_{instance}.png).
        summary_gaps: Pre-loaded optimality gaps dict.

    Returns:
        The save_path string.
    """
    if graph is None:
        graph = load_default_graph()

    if instance is None:
        inst_file = os.path.join("data", "instances", f"{instance_id}.json")
        with open(inst_file, "r") as f:
            instance = json.load(f)

    if summary_gaps is None:
        summary_gaps = load_optimality_gaps()

    if save_path is None:
        save_path = os.path.join(
            "results", "route_maps", "comparison_grids", f"comparison_grid_{instance_id}.png"
        )

    nodes_metadata = load_nodes_metadata()

    # Parse customer demand bounds for shared color and size scale
    customers = instance.get("customers", [])
    cust_demands = [float(c.get("demand", 10.0)) for c in customers]
    min_demand = min(cust_demands) if cust_demands else 1.0
    max_demand = max(cust_demands) if cust_demands else 50.0
    demand_range = max_demand - min_demand if max_demand > min_demand else 1.0

    cust_x, cust_y = [], []
    for c in customers:
        nid = int(c["node_id"])
        lat, lon = get_node_lat_lon(nid, graph, nodes_metadata)
        cust_x.append(lon)
        cust_y.append(lat)

    # Customer circle sizes (shared across all 6 subplots)
    cust_sizes = 20.0 + ((np.array(cust_demands) - min_demand) / demand_range) * 65.0

    # Depot coordinates
    depot_nid = int(instance["depot"]["node_id"])
    depot_lat, depot_lon = get_node_lat_lon(depot_nid, graph, nodes_metadata)

    # Create 2x3 subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 13.5), facecolor="#ffffff")
    plt.subplots_adjust(wspace=0.08, hspace=0.18, left=0.03, right=0.92, top=0.91, bottom=0.06)

    # Global tracking for legend
    max_vehicles_seen = 0
    scatter_handle = None

    # Determine reference cost across the 6 algorithms for gap calculation
    solutions = {}
    costs = {}
    for display_name, internal_name in GRID_ALGORITHMS:
        try:
            sol = load_solution_for_pair(instance_id, internal_name, instance=instance)
            solutions[display_name] = sol
            costs[display_name] = float(sol.total_cost)
        except Exception as e:
            print(f"Warning: could not load solution for {display_name} on {instance_id}: {e}")

    min_instance_cost = min(costs.values()) if costs else 1.0

    # Render each algorithm in its respective subplot
    for idx, (display_name, internal_name) in enumerate(GRID_ALGORITHMS):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        sol = solutions.get(display_name)
        if sol is None:
            ax.text(0.5, 0.5, f"Solution Not Found\n{display_name}", ha="center", va="center", transform=ax.transAxes)
            continue

        routes = sol.routes
        total_cost = sol.total_cost
        valid_routes = [r for r in routes if len(r) > 1]
        num_veh = len(valid_routes)
        max_vehicles_seen = max(max_vehicles_seen, num_veh)

        # 1. Background Road Network
        ox.plot_graph(
            graph,
            ax=ax,
            node_size=0,
            edge_color="#d2d2d2",
            edge_linewidth=0.45,
            bgcolor="#ffffff",
            show=False,
            close=False
        )

        # 2. Convert and draw actual road-following vehicle routes
        detailed_routes = convert_routes_to_osmnx_paths(graph, valid_routes)
        route_colors = [PALETTE[i % len(PALETTE)] for i in range(len(detailed_routes))]

        if detailed_routes:
            ox.plot_graph_routes(
                graph,
                detailed_routes,
                ax=ax,
                route_colors=route_colors,
                route_linewidths=2.5,
                orig_dest_size=0,
                node_size=0,
                show=False,
                close=False
            )

        # 3. Customer Stop Markers (shared size and colormap scale)
        if cust_x:
            scatter_handle = ax.scatter(
                cust_x,
                cust_y,
                s=cust_sizes,
                c=cust_demands,
                cmap="viridis",
                vmin=min_demand,
                vmax=max_demand,
                edgecolors="#111111",
                linewidths=0.5,
                alpha=0.92,
                zorder=10
            )

        # 4. Depot Marker (consistent red star)
        ax.scatter(
            depot_lon,
            depot_lat,
            marker="*",
            s=280,
            color="#e63946",
            edgecolors="#111111",
            linewidths=1.2,
            zorder=15
        )

        # 5. Determine Optimality Gap %
        # Check summary gaps first, fallback to min cost comparison
        gap = summary_gaps.get((instance_id, display_name))
        if gap is None:
            gap = summary_gaps.get((instance_id, get_algo_key(display_name)))
        if gap is None:
            gap = ((total_cost - min_instance_cost) / min_instance_cost) * 100.0

        gap_badge = f"{gap:.2f}%"
        if gap <= 0.001 or total_cost == min_instance_cost:
            gap_badge = "0.00% (Best)"

        # 6. Subplot Title Box
        title_text = (
            f"{display_name}\n"
            f"Cost: {total_cost:,.1f} m  |  Vehicles: {num_veh}  |  Gap: {gap_badge}"
        )
        ax.set_title(
            title_text,
            fontsize=11.5,
            fontweight="bold",
            color="#0f172a",
            pad=8
        )

    # Shared Colorbar for Customer Demand (Right edge)
    if scatter_handle is not None:
        cbar_ax = fig.add_axes([0.935, 0.22, 0.015, 0.55])
        cbar = fig.colorbar(scatter_handle, cax=cbar_ax)
        cbar.set_label("Customer Demand (units / kg)", fontsize=11, fontweight="bold", labelpad=8)
        cbar.ax.tick_params(labelsize=9.5)

    # Shared Legend (Top center)
    legend_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#e63946", markeredgecolor="k", markersize=14, label="Depot")
    ]
    # Add representative vehicle lines
    display_veh_count = min(max_vehicles_seen, 10)
    for v_idx in range(display_veh_count):
        legend_handles.append(
            Line2D([0], [0], color=PALETTE[v_idx % len(PALETTE)], lw=2.8, label=f"Vehicle {v_idx + 1}")
        )
    if max_vehicles_seen > 10:
        legend_handles.append(
            Line2D([0], [0], color="#666666", lw=2, linestyle="--", label=f"+{max_vehicles_seen - 10} more")
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.47, 0.965),
        ncol=len(legend_handles),
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        framealpha=0.95,
        fontsize=9.5
    )

    # Global Figure Super Title
    N = len(customers)
    fig.suptitle(
        f"VRP Algorithm Benchmark Comparison Grid — {instance_id.upper()} (N={N} Customers, Kharagpur Road Network)",
        fontsize=16,
        fontweight="heavy",
        color="#0f172a",
        y=0.99
    )

    # Save figure
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison grid to: '{save_path}'")

    return save_path


def generate_all_comparison_grids(
    instance_ids: Optional[List[str]] = None,
    output_dir: str = os.path.join("results", "route_maps", "comparison_grids")
):
    """
    Batch generates comparison grids for all benchmark instances.
    """
    graph = load_default_graph()
    summary_gaps = load_optimality_gaps()

    instances_dir = os.path.join("data", "instances")
    if instance_ids is None:
        inst_files = [f.replace(".json", "") for f in os.listdir(instances_dir) if f.endswith(".json")]
        inst_files.sort()
        instance_ids = inst_files

    os.makedirs(output_dir, exist_ok=True)

    for inst_id in instance_ids:
        inst_path = os.path.join(instances_dir, f"{inst_id}.json")
        if not os.path.exists(inst_path):
            continue

        with open(inst_path, "r") as f:
            instance = json.load(f)

        save_path = os.path.join(output_dir, f"comparison_grid_{inst_id}.png")
        try:
            plot_comparison_grid(
                instance_id=inst_id,
                graph=graph,
                instance=instance,
                save_path=save_path,
                summary_gaps=summary_gaps
            )
        except Exception as e:
            print(f"Error creating comparison grid for {inst_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2x3 Side-by-Side VRP Algorithm Comparison Grids.")
    parser.add_argument("--instance", type=str, default="instance_n50", help="Instance ID (e.g. instance_n20, instance_n50, instance_n100)")
    parser.add_argument("--all", action="store_true", help="Generate comparison grids for all instances")
    parser.add_argument("--output_dir", type=str, default=os.path.join("results", "route_maps", "comparison_grids"), help="Output directory")

    args = parser.parse_args()

    if args.all:
        generate_all_comparison_grids(output_dir=args.output_dir)
    else:
        out_file = os.path.join(args.output_dir, f"comparison_grid_{args.instance}.png")
        plot_comparison_grid(instance_id=args.instance, save_path=out_file)

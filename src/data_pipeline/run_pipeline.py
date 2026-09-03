"""
Master Data & Traffic Pipeline Orchestrator.
Executes the full end-to-end data processing, simulation, calibration, and instance generation:
1. OSMnx Road Graph & Matrices (fetch_osm)
2. Kaggle Demand Calibration & Customer POIs (fetch_demand)
3. Depot Candidates & Postal Hubs (fetch_depots)
4. Traffic Simulation & Congestion Weights (traffic_simulation)
5. Problem Instances Generation (build_instance)
6. Data & Traffic Visualizations (visualization suite)
"""

import os
import sys
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_pipeline.fetch_osm import fetch_and_process_road_network
from src.data_pipeline.fetch_demand import process_demand
from src.data_pipeline.fetch_depots import fetch_and_snap_depots
from src.data_pipeline.traffic_simulation import run_traffic_pipeline
from src.data_pipeline.build_instance import build_instances


def run_full_pipeline(
    city_name: str = "Kharagpur, West Bengal, India",
    radius: float = 3000,
    generate_visuals: bool = True
):
    print("=" * 70)
    print(f"SIH QPSO-VRP: INTEGRATED DATA & TRAFFIC PIPELINE")
    print(f"Target City: '{city_name}' (radius: {radius}m)")
    print("=" * 70)
    start_time = time.time()

    # Step 1: Road Graph & Dijkstra Matrices
    print("\n[Step 1/5] Fetching Road Network & Computing Topological Matrices...")
    G, G_scc, osm_xml_path = fetch_and_process_road_network(city_name=city_name, radius=radius)

    # Step 2: Demand Calibration & Customer Extraction
    print("\n[Step 2/5] Calibrating Kaggle Demands & Extracting Customer POIs...")
    process_demand(city_name=city_name, osm_xml_path=osm_xml_path)

    # Step 3: Depots & Facilities
    print("\n[Step 3/5] Resolving Warehouses (2-Pass Ways) & Post Office Depots...")
    fetch_and_snap_depots(city_name=city_name, osm_xml_path=osm_xml_path)

    # Step 4: Traffic Simulation & Congestion Weights
    print("\n[Step 4/5] Running Traffic Simulation & Multi-Window Congestion Weighting...")
    run_traffic_pipeline(G)

    # Step 5: Compose VRP Instances
    print("\n[Step 5/5] Composing Benchmark Problem Instances (N=20, 50, 100)...")
    build_instances()

    # Optional Step 6: Generate Visualizations
    if generate_visuals:
        print("\n[Step 6/5] Generating Data & Traffic Visualizations...")
        try:
            from src.visualization.data_visualizer import generate_all_data_plots
            from src.visualization.traffic_visualizer import generate_all_traffic_plots
            generate_all_data_plots()
            generate_all_traffic_plots(G)
        except Exception as e:
            print(f"  Note on visualizer generation: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE EXECUTION COMPLETE (Total time: {elapsed:.2f}s)")
    print("=" * 70)
    print("Persistent Artifacts Ready:")
    print("  - data/raw/osm/                 Road networks (.graphml) & raw XML (.osm.xml)")
    print("  - data/processed/distance_matrix.npy    (N x N) shortest road distance matrix")
    print("  - data/processed/travel_time_matrix.npy (N x N) shortest travel time matrix")
    print("  - data/processed/demand_nodes.json      Customer POIs, time windows & service times")
    print("  - data/processed/depot_nodes.json       Warehouse & Postal depot facilities")
    print("  - data/processed/kaggle_calibration.json Fitted package weight lognormal & cost/km")
    print("  - data/processed/fleet_calibration.json Fitted vehicle speed factors")
    print("  - data/processed/time_aware_weights.json Congestion multipliers per window & 15m bucket")
    print("  - data/processed/cost_params.json       Fleet definitions & cost weights")
    print("  - data/instances/instance_n*.json       Self-contained benchmark instances")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run full SIH data & traffic pipeline")
    parser.add_argument("--city", type=str, default="Kharagpur, West Bengal, India",
                        help="Target city or location name")
    parser.add_argument("--radius", type=float, default=3000,
                        help="Search radius in meters")
    parser.add_argument("--skip-visuals", action="store_true",
                        help="Skip generating static and interactive visualization reports")
    args = parser.parse_args()

    run_full_pipeline(
        city_name=args.city,
        radius=args.radius,
        generate_visuals=not args.skip_visuals
    )


if __name__ == "__main__":
    main()

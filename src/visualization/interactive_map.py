"""
Interactive Folium Route Map Visualizer for VRP Solutions.
Renders interactive zoomable maps tracing real OpenStreetMap road geometry,
with individual vehicle route layers, custom depot and customer markers, popups,
and layer toggling controls.
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
import folium
from folium.plugins import Fullscreen

from src.algorithms.base import SolutionResult
from src.benchmark.runner import get_algo_key
from src.visualization.static_map import (
    load_default_graph,
    load_solution_for_pair,
    ALGORITHM_CLASSES,
    DEFAULT_GRAPH_PATHS,
)

# Standard default Kharagpur center coordinates
KHARAGPUR_CENTER = [22.340006, 87.304860]

# Vibrant qualitative palette for vehicle route tracing
PALETTE = [
    "#0072B2",  # Deep Blue
    "#D55E00",  # Vermilion / Red-Orange
    "#009E73",  # Bluish Green
    "#CC79A7",  # Reddish Purple
    "#E69F00",  # Amber / Warm Yellow
    "#56B4E9",  # Sky Blue
    "#F0E442",  # Yellow
    "#332288",  # Indigo
    "#88CCEE",  # Cyan
    "#44AA99",  # Teal
    "#117733",  # Deep Green
    "#999933",  # Olive
    "#DDCC77",  # Sand
    "#CC6677",  # Rose
    "#882255",  # Wine
    "#AA4499",  # Magenta
    "#4dac26",  # Lime
    "#7570b3",  # Slate Purple
    "#e7298a",  # Vibrant Pink
    "#e6ab02",  # Goldenrod
]

# Preset colors for folium.Icon markers
FOLIUM_ICON_COLORS = [
    "blue", "green", "purple", "orange", "darkred", "lightred",
    "darkblue", "darkgreen", "cadetblue", "darkpurple", "pink", "lightblue", "lightgreen"
]


def load_nodes_metadata(meta_path: str = os.path.join("data", "processed", "nodes_metadata.json")) -> Dict[str, Dict[str, float]]:
    """
    Loads nodes metadata dictionary containing {'lat': ..., 'lon': ...} per node ID.
    """
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


def get_node_lat_lon(
    node_id: int,
    graph: nx.MultiDiGraph,
    nodes_metadata: Dict[str, Dict[str, float]]
) -> Tuple[float, float]:
    """
    Looks up (lat, lon) coordinates for a given road node ID.
    Prefers nodes_metadata, falls back to graph.nodes attributes.
    """
    str_id = str(node_id)
    if str_id in nodes_metadata:
        m = nodes_metadata[str_id]
        if m.get("lat") is not None and m.get("lon") is not None:
            return float(m["lat"]), float(m["lon"])

    int_id = int(node_id)
    if int_id in graph.nodes:
        node_data = graph.nodes[int_id]
        return float(node_data.get("y", 22.34)), float(node_data.get("x", 87.30))

    return KHARAGPUR_CENTER[0], KHARAGPUR_CENTER[1]


def extract_road_geometry_coords(
    graph: nx.MultiDiGraph,
    route: List[int],
    nodes_metadata: Dict[str, Dict[str, float]]
) -> List[List[float]]:
    """
    Extracts high-resolution (lat, lon) coordinates along actual road geometry
    between consecutive stops in a vehicle route.

    Uses shortest path Dijkstra traversals, tracing curved OSM edge geometries
    (shapely LineString) so lines follow real physical streets instead of straight lines.
    """
    if not route or len(route) <= 1:
        return []

    # 1. Expand discrete stops into a sequence of contiguous road junction nodes
    detailed_path = []
    for i in range(len(route) - 1):
        u = int(route[i])
        v = int(route[i + 1])
        if u == v:
            continue

        try:
            sp = nx.shortest_path(graph, u, v, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            try:
                sp = nx.shortest_path(graph, u, v, weight="travel_time")
            except Exception:
                sp = [u, v]

        if not detailed_path:
            detailed_path.extend(sp)
        else:
            detailed_path.extend(sp[1:])

    if len(detailed_path) <= 1:
        return []

    # 2. Extract edge geometry coordinates (handling curved streets)
    road_coords = []
    for i in range(len(detailed_path) - 1):
        u_node = detailed_path[i]
        v_node = detailed_path[i + 1]

        edge_dict = graph.get_edge_data(u_node, v_node)
        has_geom = False

        if edge_dict:
            best_edge = min(edge_dict.values(), key=lambda d: d.get("length", 0))
            if "geometry" in best_edge:
                has_geom = True
                # shapely LineString coords are in (lon, lat) order
                for lon_val, lat_val in best_edge["geometry"].coords:
                    point = [float(lat_val), float(lon_val)]
                    if not road_coords or road_coords[-1] != point:
                        road_coords.append(point)

        if not has_geom:
            lat1, lon1 = get_node_lat_lon(u_node, graph, nodes_metadata)
            lat2, lon2 = get_node_lat_lon(v_node, graph, nodes_metadata)
            p1 = [lat1, lon1]
            p2 = [lat2, lon2]
            if not road_coords or road_coords[-1] != p1:
                road_coords.append(p1)
            road_coords.append(p2)

    return road_coords


def build_interactive_route_map(
    graph: Optional[nx.MultiDiGraph],
    instance: Dict[str, Any],
    solution: SolutionResult,
    algorithm_name: str,
    save_path: str,
    nodes_metadata: Optional[Dict[str, Dict[str, float]]] = None
) -> folium.Map:
    """
    Builds a standalone interactive folium.Map showing real road-following
    routes, interactive stop popups, layer toggling, and depot markers.

    Args:
        graph: OSMnx road network graph.
        instance: VRP instance dictionary with depot and customers.
        solution: SolutionResult object containing .routes and .total_cost.
        algorithm_name: Display name of the algorithm (e.g., 'QPSO', 'Genetic Algorithm').
        save_path: Destination path for saving the standalone HTML file.
        nodes_metadata: Optional pre-loaded nodes_metadata dictionary.

    Returns:
        folium.Map instance.
    """
    if graph is None:
        graph = load_default_graph()

    if nodes_metadata is None:
        nodes_metadata = load_nodes_metadata()

    if isinstance(instance, str):
        if os.path.exists(instance):
            with open(instance, "r") as f:
                instance = json.load(f)
        else:
            inst_path = os.path.join("data", "instances", f"{instance}.json")
            with open(inst_path, "r") as f:
                instance = json.load(f)

    # Determine center coordinates from metadata
    if nodes_metadata:
        lats = [v["lat"] for v in nodes_metadata.values() if v.get("lat") is not None]
        lons = [v["lon"] for v in nodes_metadata.values() if v.get("lon") is not None]
        center_lat = float(np.mean(lats)) if lats else KHARAGPUR_CENTER[0]
        center_lon = float(np.mean(lons)) if lons else KHARAGPUR_CENTER[1]
    else:
        center_lat, center_lon = KHARAGPUR_CENTER

    # Initialize Folium Map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True
    )
    Fullscreen().add_to(m)

    # 1. Depot Setup
    depot_nid = int(instance["depot"]["node_id"])
    depot_lat, depot_lon = get_node_lat_lon(depot_nid, graph, nodes_metadata)

    depot_group = folium.FeatureGroup(name="★ Central Depot", show=True)

    depot_popup_html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; min-width: 200px; color: #1e293b;">
        <div style="background: #e63946; color: white; padding: 6px 10px; border-radius: 4px; font-weight: bold; margin-bottom: 8px;">
            CENTRAL POSTAL DEPOT
        </div>
        <b>Hub Location:</b> Kharagpur Division<br>
        <b>OSM Node ID:</b> <code>{depot_nid}</code><br>
        <b>Latitude:</b> {depot_lat:.5f}<br>
        <b>Longitude:</b> {depot_lon:.5f}<br>
        <b>Fleet Capacity:</b> {instance.get('vehicle_capacity', 100)} units/vehicle
    </div>
    """

    folium.Marker(
        location=[depot_lat, depot_lon],
        icon=folium.Icon(icon="home", color="red", icon_color="white"),
        popup=folium.Popup(depot_popup_html, max_width=300),
        tooltip="Central Depot (Start / Return)"
    ).add_to(depot_group)
    depot_group.add_to(m)

    # 2. Vehicle Routes Setup
    routes = solution.routes if hasattr(solution, "routes") else solution.get("routes", [])
    total_cost = solution.total_cost if hasattr(solution, "total_cost") else solution.get("cost", 0.0)

    customers = instance.get("customers", [])
    cust_demands = {int(c["node_id"]): c.get("demand", 0) for c in customers}

    for v_idx, route in enumerate(routes):
        v_num = v_idx + 1
        color = PALETTE[v_idx % len(PALETTE)]
        icon_color = FOLIUM_ICON_COLORS[v_idx % len(FOLIUM_ICON_COLORS)]

        customer_stops = [nid for nid in route if int(nid) != depot_nid]
        if not customer_stops:
            continue

        v_group = folium.FeatureGroup(
            name=f"Vehicle {v_num} ({len(customer_stops)} stops)",
            show=True
        )

        # Extract actual physical road geometry coordinates
        road_coords = extract_road_geometry_coords(graph, route, nodes_metadata)

        if road_coords:
            folium.PolyLine(
                locations=road_coords,
                color=color,
                weight=4.5,
                opacity=0.88,
                tooltip=f"Vehicle {v_num} Route ({len(customer_stops)} stops)"
            ).add_to(v_group)

        # Customer stop markers
        for seq, stop_nid in enumerate(customer_stops, start=1):
            stop_nid_int = int(stop_nid)
            s_lat, s_lon = get_node_lat_lon(stop_nid_int, graph, nodes_metadata)
            d_val = cust_demands.get(stop_nid_int, 0)

            popup_html = f"""
            <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; min-width: 210px; color: #1e293b;">
                <div style="background: {color}; color: white; padding: 5px 8px; border-radius: 4px; font-weight: bold; margin-bottom: 8px;">
                    Vehicle {v_num} — Stop #{seq}
                </div>
                <b>Customer Node:</b> <code>{stop_nid_int}</code><br>
                <b>Delivery Demand:</b> <span style="font-weight: bold; color: #0284c7;">{d_val} units</span><br>
                <b>Sequence:</b> {seq} of {len(customer_stops)}<br>
                <b>Coordinates:</b> {s_lat:.5f}, {s_lon:.5f}
            </div>
            """

            folium.Marker(
                location=[s_lat, s_lon],
                icon=folium.Icon(color=icon_color, icon="info-sign"),
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Vehicle {v_num} | Stop #{seq} (Demand: {d_val})"
            ).add_to(v_group)

        v_group.add_to(m)

    # 3. Layer Control for toggling vehicle routes
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    # 4. Interactive Floating Header Info Panel
    instance_id = instance.get("instance_id", "Instance")
    num_cust = len(customers)
    num_veh = len([r for r in routes if len(r) > 2])

    banner_html = f"""
    <div style="
        position: fixed;
        top: 20px;
        left: 65px;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.95);
        padding: 12px 18px;
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        pointer-events: auto;
    ">
        <div style="font-size: 15px; font-weight: bold; color: #0f172a; margin-bottom: 4px;">
            {algorithm_name} — {instance_id}
        </div>
        <div style="font-size: 12px; color: #334155; line-height: 1.5;">
            <b>Total Cost:</b> <span style="color: #059669; font-weight: bold;">{total_cost:,.2f} m</span><br>
            <b>Vehicles Deployed:</b> {num_veh} &nbsp;|&nbsp; <b>Customers:</b> {num_cust}<br>
            <span style="font-size: 11px; color: #64748b;">Road-following OpenStreetMap routing</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(banner_html))

    # Save to HTML
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    m.save(save_path)
    print(f"Saved interactive route map to: '{save_path}'")

    return m


def generate_all_interactive_maps(
    instance_ids: Optional[List[str]] = None,
    algorithm_names: Optional[List[str]] = None,
    output_dir: str = os.path.join("results", "route_maps", "interactive")
):
    """
    Batch generates standalone interactive HTML maps for all combinations of instances and algorithms.
    """
    graph = load_default_graph()
    nodes_metadata = load_nodes_metadata()

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
            save_name = f"{inst_id}_{algo_slug}.html"
            save_path = os.path.join(output_dir, save_name)

            try:
                solution = load_solution_for_pair(inst_id, algo, instance=instance)
                build_interactive_route_map(
                    graph=graph,
                    instance=instance,
                    solution=solution,
                    algorithm_name=algo,
                    save_path=save_path,
                    nodes_metadata=nodes_metadata
                )
            except Exception as e:
                print(f"Error generating interactive map for {inst_id} - {algo}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interactive Folium route map visualizations for VRP solutions.")
    parser.add_argument("--instance", type=str, default="instance_n20", help="Instance ID (e.g. instance_n20)")
    parser.add_argument("--algorithm", type=str, default="QPSO", help="Algorithm name (e.g. QPSO, 'Genetic Algorithm')")
    parser.add_argument("--all", action="store_true", help="Generate interactive maps for all instances and algorithms")
    parser.add_argument("--output_dir", type=str, default=os.path.join("results", "route_maps", "interactive"), help="Output directory")

    args = parser.parse_args()

    if args.all:
        generate_all_interactive_maps(output_dir=args.output_dir)
    else:
        inst_id = args.instance
        algo = args.algorithm
        algo_slug = get_algo_key(algo)
        out_file = os.path.join(args.output_dir, f"{inst_id}_{algo_slug}.html")

        G = load_default_graph()
        meta = load_nodes_metadata()
        inst_path = os.path.join("data", "instances", f"{inst_id}.json")
        with open(inst_path, "r") as f:
            inst = json.load(f)

        sol = load_solution_for_pair(inst_id, algo, instance=inst)
        build_interactive_route_map(
            graph=G,
            instance=inst,
            solution=sol,
            algorithm_name=algo,
            save_path=out_file,
            nodes_metadata=meta
        )

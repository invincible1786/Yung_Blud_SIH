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
    depot_meta = instance.get("depot", {})
    depot_name = depot_meta.get("name", "Central Postal Depot")
    depot_cat = depot_meta.get("category", "building:warehouse").replace("_", " ").title()
    depot_cap_kg = depot_meta.get("capacity_kg", 5000)

    depot_group = folium.FeatureGroup(name="★ Primary Hub Depot", show=True)

    depot_popup_html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; min-width: 220px; color: #1e293b;">
        <div style="background: #e63946; color: white; padding: 6px 10px; border-radius: 4px; font-weight: bold; margin-bottom: 8px;">
            {depot_name.upper()}
        </div>
        <b>Facility Class:</b> {depot_cat}<br>
        <b>OSM Node ID:</b> <code>{depot_nid}</code><br>
        <b>Latitude / Longitude:</b> {depot_lat:.5f}, {depot_lon:.5f}<br>
        <b>Storage Capacity:</b> <span style="color: #059669; font-weight: bold;">{depot_cap_kg:,.0f} kg</span><br>
        <b>Fleet Standard Capacity:</b> {instance.get('vehicle_capacity', 100)} kg/vehicle
    </div>
    """

    folium.Marker(
        location=[depot_lat, depot_lon],
        icon=folium.Icon(icon="home", color="red", icon_color="white"),
        popup=folium.Popup(depot_popup_html, max_width=320),
        tooltip=f"Primary Depot: {depot_name} (Capacity: {depot_cap_kg} kg)"
    ).add_to(depot_group)
    depot_group.add_to(m)

    # 1b. Additional Candidate Depots Layer
    depot_json_path = os.path.join("data", "processed", "depot_nodes.json")
    if os.path.exists(depot_json_path):
        try:
            with open(depot_json_path, "r") as f:
                d_nodes_data = json.load(f)
            cand_group = folium.FeatureGroup(name="Alternative Hub Candidates", show=False)
            for d in d_nodes_data.get("depots", []):
                if int(d.get("graph_node", -1)) == depot_nid:
                    continue
                d_lat, d_lon = d["lat"], d["lon"]
                d_name = d["name"]
                d_cap = d.get("capacity_kg", 3000)
                d_cat = d.get("category", "hub").replace("_", " ").title()
                c_popup = f"""
                <div style="font-family: 'Segoe UI', Arial; font-size: 12px; min-width: 180px;">
                    <b style="color: #d97706;">{d_name}</b><br>
                    <b>Category:</b> {d_cat}<br>
                    <b>Capacity:</b> {d_cap} kg<br>
                    <b>Source:</b> {d.get('source', 'osm')}
                </div>
                """
                folium.Marker(
                    location=[d_lat, d_lon],
                    icon=folium.Icon(icon="briefcase", color="orange", icon_color="white"),
                    popup=folium.Popup(c_popup, max_width=250),
                    tooltip=f"Candidate: {d_name} ({d_cap} kg)"
                ).add_to(cand_group)
            cand_group.add_to(m)
        except Exception:
            pass

    # 2. Vehicle Routes Setup
    routes = solution.routes if hasattr(solution, "routes") else solution.get("routes", [])
    total_cost = solution.total_cost if hasattr(solution, "total_cost") else solution.get("cost", 0.0)

    customers = instance.get("customers", [])
    cust_meta_map = {int(c["node_id"]): c for c in customers}

    for v_idx, route in enumerate(routes):
        v_num = v_idx + 1
        color = PALETTE[v_idx % len(PALETTE)]
        icon_color = FOLIUM_ICON_COLORS[v_idx % len(FOLIUM_ICON_COLORS)]

        customer_stops = [nid for nid in route if int(nid) != depot_nid]
        if not customer_stops:
            continue

        route_load = sum(cust_meta_map.get(int(nid), {}).get("demand", 0) for nid in customer_stops)
        if route_load <= 15:
            veh_label = "Two-Wheeler Courier"
        elif route_load <= 200:
            veh_label = "Three-Wheeler Tempo"
        else:
            veh_label = "Light Commercial Vehicle (LCV)"

        v_group = folium.FeatureGroup(
            name=f"Vehicle {v_num} [{veh_label}] ({len(customer_stops)} stops, {route_load} kg)",
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
                tooltip=f"Vehicle {v_num} [{veh_label}] | Payload: {route_load} kg"
            ).add_to(v_group)

        # Customer stop markers
        for seq, stop_nid in enumerate(customer_stops, start=1):
            stop_nid_int = int(stop_nid)
            s_lat, s_lon = get_node_lat_lon(stop_nid_int, graph, nodes_metadata)
            c_info = cust_meta_map.get(stop_nid_int, {})
            d_val = c_info.get("demand", 0)
            d_units = c_info.get("demand_units", 1)
            c_name = c_info.get("name", f"Customer {stop_nid_int}")
            c_cat = c_info.get("category", "amenity:commercial").replace("_", " ").title()
            tw = c_info.get("time_window", [480, 600])
            start_h, start_m = divmod(tw[0], 60)
            end_h, end_m = divmod(tw[1], 60)
            tw_str = f"{start_h:02d}:{start_m:02d} – {end_h:02d}:{end_m:02d}"
            serv_time = c_info.get("service_time_min", 5)

            # Icon choice based on category
            icon_name = "tag"
            if "shop" in c_cat.lower() or "market" in c_cat.lower():
                icon_name = "shopping-cart"
            elif "hospital" in c_cat.lower() or "clinic" in c_cat.lower() or "pharmacy" in c_cat.lower():
                icon_name = "medkit"
            elif "food" in c_cat.lower() or "restaurant" in c_cat.lower() or "cafe" in c_cat.lower():
                icon_name = "cutlery"
            elif "office" in c_cat.lower() or "bank" in c_cat.lower():
                icon_name = "briefcase"

            popup_html = f"""
            <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; min-width: 230px; color: #1e293b;">
                <div style="background: {color}; color: white; padding: 5px 8px; border-radius: 4px; font-weight: bold; margin-bottom: 8px;">
                    Vehicle {v_num} — Stop #{seq} of {len(customer_stops)}
                </div>
                <b>Destination:</b> <span style="color: #0f172a; font-weight: 600;">{c_name}</span><br>
                <b>Category:</b> {c_cat}<br>
                <b>Payload Demand:</b> <span style="font-weight: bold; color: #0284c7;">{d_val} kg</span> ({d_units} units)<br>
                <b>Delivery Window:</b> <span style="color: #b45309; font-weight: bold;">{tw_str}</span><br>
                <b>Service Duration:</b> {serv_time} mins<br>
                <b>OSM Node ID:</b> <code>{stop_nid_int}</code>
            </div>
            """

            folium.Marker(
                location=[s_lat, s_lon],
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix="fa" if icon_name in ["medkit", "cutlery", "shopping-cart"] else "glyphicon"),
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"Vehicle {v_num} | Stop #{seq}: {c_name} ({d_val} kg | {tw_str})"
            ).add_to(v_group)

        v_group.add_to(m)

    # 2b. Traffic Congestion Overlay Layer (Optional)
    weights_path = os.path.join("data", "processed", "time_aware_weights.json")
    if os.path.exists(weights_path):
        try:
            with open(weights_path, "r") as f:
                tw_data = json.load(f)
            tw_weights = tw_data.get("weights", {})
            cong_group = folium.FeatureGroup(name="Peak Traffic Congestion Corridors", show=False)

            # Highlight top congested edges
            c_count = 0
            for u, v, k, data in graph.edges(keys=True, data=True):
                edge_k = f"{u}_{v}_{k}"
                e_dict = tw_weights.get(edge_k, {}).get("evening_peak", {})
                if not e_dict:
                    continue
                mult = float(np.mean(list(e_dict.values())))
                if mult >= 1.6:
                    if "geometry" in data:
                        seg_coords = [(lat, lon) for lon, lat in data["geometry"].coords]
                    else:
                        seg_coords = [(graph.nodes[u]["y"], graph.nodes[u]["x"]), (graph.nodes[v]["y"], graph.nodes[v]["x"])]
                    folium.PolyLine(
                        locations=seg_coords,
                        color="#ef4444" if mult >= 2.0 else "#f97316",
                        weight=4,
                        opacity=0.8,
                        tooltip=f"Evening Peak Congestion: {mult:.2f}x delay"
                    ).add_to(cong_group)
                    c_count += 1
                    if c_count > 300:
                        break
            cong_group.add_to(m)
        except Exception:
            pass

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

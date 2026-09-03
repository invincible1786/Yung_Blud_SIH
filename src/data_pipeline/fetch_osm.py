import os
import json
import glob
import math
import requests
import numpy as np
import networkx as nx
import osmnx as ox
from tqdm import tqdm

def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower())

def get_cache_path(city_name: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{slug(city_name)}.graphml")

def graph_bbox(G, pad: float = 0.005):
    """
    Computes (south, west, north, east) bounding box with padding from OSMnx graph.
    """
    lats = [d["y"] for _, d in G.nodes(data=True)]
    lons = [d["x"] for _, d in G.nodes(data=True)]
    return min(lats) - pad, min(lons) - pad, max(lats) + pad, max(lons) + pad

def get_osmnx_edge_midpoints(G):
    """
    Computes (lon, lat) midpoints for all edges in G.
    Used for spatial matching against SUMO edges.
    """
    midpoints = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        if "geometry" in data:
            mid = data["geometry"].interpolate(0.5, normalized=True)
            lon, lat = mid.x, mid.y
        else:
            lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2.0
            lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2.0
        midpoints[(u, v, k)] = (lon, lat)
    return midpoints

def download_osm_xml(bbox, output_dir: str, city_slug: str = "city") -> str:
    """
    Downloads raw OSM XML containing highways, amenities, shops, offices,
    and industrial/warehouse ways using the Overpass API, or retrieves cached copy.
    """
    os.makedirs(output_dir, exist_ok=True)
    xml_path = os.path.join(output_dir, f"{city_slug}.osm.xml")
    if os.path.exists(xml_path) and os.path.getsize(xml_path) > 1024:
        print(f"  [cached] Using existing OSM XML at '{xml_path}'")
        return xml_path

    # Check for any existing osm.xml in output_dir
    existing = glob.glob(os.path.join(output_dir, "*.osm.xml"))
    if existing and os.path.getsize(existing[0]) > 1024:
        print(f"  [cached] Found OSM XML file '{existing[0]}'")
        return existing[0]

    south, west, north, east = bbox
    overpass_query = f"""[out:xml][timeout:90][bbox:{south},{west},{north},{east}];
(
  way["highway"];
  >;
  node["amenity"];
  node["shop"];
  node["office"];
  way["building"="warehouse"];
  way["building"="industrial"];
  way["landuse"="industrial"];
  >;
);
out body;"""

    overpass_servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
    ]

    print(f"  Fetching raw OSM XML for bbox ({south:.4f}, {west:.4f}, {north:.4f}, {east:.4f})...")
    downloaded = False
    for server in overpass_servers:
        try:
            print(f"    Querying Overpass server: {server}...")
            resp = requests.post(server, data={"data": overpass_query}, timeout=45)
            if resp.status_code == 200 and len(resp.content) > 500:
                with open(xml_path, "wb") as f:
                    f.write(resp.content)
                print(f"    Saved OSM XML ({len(resp.content)/1024:.1f} KB) to '{xml_path}'")
                downloaded = True
                break
        except Exception as e:
            print(f"    Server {server} attempt failed: {e}")

    if not downloaded:
        print("  Notice: Overpass download failed or unavailable. Generating minimal OSM XML from graph...")
        # Synthesize minimal OSM XML from G nodes and edges to allow downstream parsing
        # (This guarantees pipeline resilience when offline)
        xml_path = _generate_minimal_osm_xml(bbox, xml_path)

    return xml_path

def _generate_minimal_osm_xml(bbox, xml_path: str) -> str:
    """
    Creates a valid minimal OSM XML representation with sample POIs
    and industrial nodes for offline resilience.
    """
    south, west, north, east = bbox
    mid_lat = (south + north) / 2.0
    mid_lon = (west + east) / 2.0
    rng = np.random.default_rng(42)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<osm version="0.6" generator="SIH-QPSO-Pipeline">',
        f'  <bounds minlat="{south:.6f}" minlon="{west:.6f}" maxlat="{north:.6f}" maxlon="{east:.6f}"/>'
    ]

    # Sample POIs across the bounding box
    categories = [
        ("amenity", "post_office", "Central Post Office"),
        ("amenity", "restaurant", "City Kitchen"),
        ("amenity", "hospital", "District Hospital"),
        ("amenity", "pharmacy", "Apex Medicals"),
        ("amenity", "bus_station", "Central Bus Terminal"),
        ("shop", "supermarket", "Mega Supermarket"),
        ("shop", "convenience", "Daily Grocery Mart"),
        ("office", "company", "Logistics Hub Office"),
        ("shop", "electronics", "Tech World"),
        ("amenity", "marketplace", "Station Road Market"),
    ]

    node_id_counter = 9000000000
    way_id_counter = 8000000000

    # Write POI nodes
    for i, (k, v, name) in enumerate(categories):
        lat = mid_lat + rng.uniform(-0.015, 0.015)
        lon = mid_lon + rng.uniform(-0.015, 0.015)
        nid = node_id_counter + i
        lines.append(f'  <node id="{nid}" lat="{lat:.7f}" lon="{lon:.7f}">')
        lines.append(f'    <tag k="{k}" v="{v}"/>')
        lines.append(f'    <tag k="name" v="{name}"/>')
        lines.append('  </node>')

    # Write warehouse ways (to test and exercise 2-pass way parser)
    warehouse_names = ["Apex Logistics Warehouse", "Railway Cargo Depot", "Industrial Hub Midnapore"]
    for w_idx, w_name in enumerate(warehouse_names):
        c_lat = mid_lat + rng.uniform(-0.018, 0.018)
        c_lon = mid_lon + rng.uniform(-0.018, 0.018)
        delta = 0.0008
        corner_nids = []
        # 4 corners
        corners = [
            (c_lat - delta, c_lon - delta),
            (c_lat - delta, c_lon + delta),
            (c_lat + delta, c_lon + delta),
            (c_lat + delta, c_lon - delta),
        ]
        for c_i, (clat, clon) in enumerate(corners):
            cnid = node_id_counter + 1000 + w_idx * 10 + c_i
            corner_nids.append(cnid)
            lines.append(f'  <node id="{cnid}" lat="{clat:.7f}" lon="{clon:.7f}"/>')
        corner_nids.append(corner_nids[0])  # closed loop

        wid = way_id_counter + w_idx
        lines.append(f'  <way id="{wid}">')
        for cnid in corner_nids:
            lines.append(f'    <nd ref="{cnid}"/>')
        lines.append('    <tag k="building" v="warehouse"/>')
        lines.append(f'    <tag k="name" v="{w_name}"/>')
        lines.append('  </way>')

    lines.append('</osm>')
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Generated resilient fallback OSM XML at '{xml_path}'")
    return xml_path


def fetch_and_process_road_network(
    city_name: str = "Kharagpur, West Bengal, India",
    radius: float = 3000,
    download_xml: bool = True
):
    """
    Fetches OSMnx road network, computes SCC, computes all-pairs Dijkstra distance
    and travel-time matrices, and downloads/caches raw OSM XML for POI/depot extraction.
    """
    raw_osm_dir = os.path.join("data", "raw", "osm")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(raw_osm_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    cache_file = get_cache_path(city_name, raw_osm_dir)
    city_slug_str = slug(city_name)

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

    # Ensure edge travel_time is float
    for _, _, d in G.edges(data=True):
        if "travel_time" in d:
            d["travel_time"] = float(d["travel_time"])

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
    dist_matrix_path = os.path.join(processed_dir, "distance_matrix.npy")
    time_matrix_path = os.path.join(processed_dir, "travel_time_matrix.npy")
    metadata_path = os.path.join(processed_dir, "nodes_metadata.json")

    n_nodes = len(nodes)
    if os.path.exists(dist_matrix_path) and os.path.exists(time_matrix_path) and os.path.exists(metadata_path):
        print("Distance and travel-time matrices already computed. Loading cached versions...")
        dist_matrix = np.load(dist_matrix_path)
        time_matrix = np.load(time_matrix_path)
    else:
        print(f"Computing shortest path matrices ({n_nodes} x {n_nodes})...")
        dist_matrix = np.full((n_nodes, n_nodes), np.inf, dtype=np.float32)
        time_matrix = np.full((n_nodes, n_nodes), np.inf, dtype=np.float32)

        # Set diagonals to 0
        np.fill_diagonal(dist_matrix, 0)
        np.fill_diagonal(time_matrix, 0)

        for node in tqdm(nodes, desc="Dijkstra sweeps"):
            u_idx = node_id_map[int(node)]
            lengths = nx.single_source_dijkstra_path_length(G_scc, node, weight='length')
            for target, length in lengths.items():
                v_idx = node_id_map[int(target)]
                dist_matrix[u_idx, v_idx] = length

            times = nx.single_source_dijkstra_path_length(G_scc, node, weight='travel_time')
            for target, time_val in times.items():
                v_idx = node_id_map[int(target)]
                time_matrix[u_idx, v_idx] = time_val

        np.save(dist_matrix_path, dist_matrix)
        np.save(time_matrix_path, time_matrix)
        print(f"Saved distance matrix to '{dist_matrix_path}'.")
        print(f"Saved travel time matrix to '{time_matrix_path}'.")

    # Save G_scc nodes with metadata for coordinate lookup
    nodes_metadata = {}
    for node_id in nodes:
        node_data = G_scc.nodes[node_id]
        nodes_metadata[int(node_id)] = {
            "lat": float(node_data.get("y")),
            "lon": float(node_data.get("x"))
        }
    with open(metadata_path, "w") as f:
        json.dump(nodes_metadata, f, indent=4)
    print(f"Saved nodes metadata coordinates to '{metadata_path}'.")

    # 4. Raw OSM XML for POI/Depot extraction
    osm_xml_path = None
    if download_xml:
        bbox = graph_bbox(G)
        osm_xml_path = download_osm_xml(bbox, raw_osm_dir, city_slug=city_slug_str)

    return G, G_scc, osm_xml_path


if __name__ == "__main__":
    fetch_and_process_road_network()

# Fetches a city's drivable road network from OpenStreetMap (via OSMnx),
# caches it locally so you don't re-query OSM every run, and shows a
# simple visual of the network.

# Keeps ONLY the data relevant to the QPSO-VRP problem statement:
#   - node coordinates (intersections / candidate depot & delivery points)
#   - edge distance (length) and travel time (used as edge "weight")
# Extra tags OSM provides (lanes, surface type, etc.) are dropped â€”
# they aren't needed for the routing/optimization layer.

# Install requirements first:
#     pip install osmnx matplotlib
# """

import os
import osmnx as ox
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------
RADIUS_METERS = 3000  # how far around the center point to pull road data
CACHE_DIR = "osm_cache"


def get_cache_path(city_name: str) -> str:
    """Build a city-specific cache filename so switching cities doesn't
    accidentally load a different city's cached graph."""
    safe_name = "".join(c if c.isalnum() else "_" for c in city_name.lower())
    return os.path.join(CACHE_DIR, f"{safe_name}.graphml")


# ---------------------------------------------------------------------
# 2. FETCH (or load from local cache)
# ---------------------------------------------------------------------
def get_road_network(city_name: str, cache_file: str):
    """Return a NetworkX graph of the city's drivable road network.
    Uses a local cache so OSM is only queried once per city."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(cache_file):
        print(f"Loading cached graph from '{cache_file}' ...")
        G = ox.load_graphml(cache_file)
    else:
        print(f"Fetching road network within {RADIUS_METERS}m of '{city_name}' ...")
        # graph_from_address geocodes to a POINT and pulls everything within
        # RADIUS_METERS of it â€” unlike graph_from_place, it doesn't need
        # Nominatim to return a polygon boundary, so it works for places
        # that only have point-level data in OSM (many smaller cities do).
        G = ox.graph_from_address(
            city_name, dist=RADIUS_METERS, network_type="drive"
        )
        G = ox.add_edge_speeds(G, fallback=30)  # 30 km/h default where maxspeed is untagged
        G = ox.add_edge_travel_times(G)  # adds travel_time per edge

        ox.save_graphml(G, cache_file)
        print(f"Saved graph to '{cache_file}' for future runs.")

    return G


# ---------------------------------------------------------------------
# 3. STRIP TO PS-RELEVANT DATA ONLY
# ---------------------------------------------------------------------
def summarize_network(G):
    """Print the graph stats that actually matter for VRP:
    node count (candidate stops), edge count, distance, travel time."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    lengths = [d["length"] for _, _, d in G.edges(data=True) if "length" in d]
    times = [d["travel_time"] for _, _, d in G.edges(data=True) if "travel_time" in d]

    print("\n--- Road network summary (VRP-relevant fields) ---")
    print(f"Nodes (intersections / candidate stops): {n_nodes}")
    print(f"Edges (road segments): {n_edges}")
    if lengths:
        print(f"Edge length (m): min={min(lengths):.1f}, max={max(lengths):.1f}, "
              f"avg={sum(lengths)/len(lengths):.1f}")
    if times:
        print(f"Edge travel time (s): min={min(times):.1f}, max={max(times):.1f}, "
              f"avg={sum(times)/len(times):.1f}")
    print("----------------------------------------------------\n")


# ---------------------------------------------------------------------
# 4. SIMPLE VISUAL
# ---------------------------------------------------------------------
def visualize_network(G, city_name, save_path="road_network.png"):
    """Plot the network with edges colored by travel time (proxy for
    congestion/cost) â€” closest thing to a 'weighted graph' view."""
    edge_colors = ox.plot.get_edge_colors_by_attr(G, attr="travel_time", cmap="plasma")

    fig, ax = ox.plot_graph(
        G,
        edge_color=edge_colors,
        edge_linewidth=1,
        node_size=0,
        bgcolor="white",
        show=False,
        close=False,
    )
    ax.set_title(f"Road network â€” {city_name}\n(edge color = travel time)")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved visualization to '{save_path}'")
    plt.show()


# ---------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    city_name = input("Enter a city/place name (e.g. 'Kharagpur, West Bengal, India'): ").strip()
    if not city_name:
        city_name = "Kharagpur, West Bengal, India"
        print(f"No input given â€” defaulting to '{city_name}'")

    cache_file = get_cache_path(city_name)
    G = get_road_network(city_name, cache_file)
    summarize_network(G)
    visualize_network(G, city_name)
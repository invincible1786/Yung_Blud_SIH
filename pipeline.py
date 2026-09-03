"""
traffic_pipeline_v6.py

Changes from v5:
  - FIXED (real bug, documented in v5's own Â§17 limitations): depot
    extraction's `building`/`landuse` tags (warehouse, industrial) live on
    OSM <way> elements, but the parser only ever read <node> elements, so
    those tags silently matched nothing â€” depot Source 1 was effectively
    "OSM post offices/bus stations only," not "OSM warehouses/industrial
    zones too," despite DEPOT_POI_TAGS listing them. Fixed with a genuine
    two-pass parser: pass 1 (`_collect_all_node_coords`) collects every
    node's coordinates (tagged or not, since ways reference node IDs, not
    coordinates); pass 2 (`parse_ways_from_osm_xml`) finds tag-matching
    ways and resolves their location as the centroid of their referenced
    nodes. Relations (multi-polygon land-use areas) are still NOT resolved
    â€” a rarer, smaller gap, left as a stated limitation rather than
    silently claimed to be covered.
  - NEW: fleet `speed_factor` (two_wheeler/three_wheeler/lcv) is now
    FITTED per city from this city's own SUMO simulation â€” not assumed.
    Each SUMO run now also emits --tripinfo-output; a new step parses
    every simulated vehicle's actual (routeLength / duration) speed,
    pools it by SUMO vType across all time windows, and divides by that
    vType's free-flow max speed to get an observed speed_factor. This is
    genuine per-vehicle-type route data â€” the exact gap flagged as
    unaddressed in v5 (fleet was defined but not tied to any real
    per-vehicle route data). capacity_kg, capacity_units, fixed_cost_inr,
    and max_range_km stay assumed, same as v5's Kaggle fit deliberately
    left capacity_kg alone â€” a few hours of simulated trip distance/
    duration says nothing about a vehicle's payload capacity, fixed
    costs, or full-tank/battery range; only speed is legitimately
    derivable from this data.
  - Cache safety: run_sumo_window's cache check now requires BOTH the
    edgedata file AND the tripinfo file to exist before skipping a
    window's simulation â€” so a city directory cached by v5 (before
    tripinfo output existed) correctly reruns once under v6 instead of
    silently missing fleet-calibration data forever.
  - Pipeline is now 10 steps (added: fleet speed calibration, step 5).

Extended pipeline with ZERO-LATENCY customer/demand node extraction.

Pipeline (10 steps):
  1.  Fetch + cache the OSMnx road graph
  2.  Download raw OSM data for SUMO
  3.  SUMO network (netconvert)
  4.  Time-windowed, multi-vehicle-type simulation (+ tripinfo output)
  5.  Fleet speed calibration â€” from this city's own SUMO tripinfo data
  6.  Customer/demand nodes â€” parsed from local OSM XML (zero network) +
      Kaggle-CSV-calibrated weight distribution, if available
  7.  Depot locations â€” OSM XML (nodes + ways, two-pass) + India Post
      data.gov.in API
  8.  Cost / demand / fleet calibration parameters
  9.  Spatial matching (SUMO â†” OSMnx)
  10. Time-aware congestion-multiplier table

Outputs per city (all in osm_cache/<city_slug>/):
  - road_network.graphml
  - network.net.xml + per-window edgedata + per-window tripinfo
  - fleet_calibration.json (NEW â€” SUMO-fitted speed_factor per vehicle type)
  - demand_nodes.json     (customers with demand, time windows)
  - depot_nodes.json      (depot candidates with capacity)
  - cost_params.json      (fleet, cost/km, demand distributions)
  - time_aware_weights.json (congestion multipliers)

Global cache (osm_cache/, shared across all cities):
  - kaggle_calibration.json (fitted weight/cost stats, or fallback marker)

Requirements: same as v5 â€” no new dependencies.
    pip install osmnx matplotlib scipy sumolib numpy requests
    EITHER: SUMO installed natively (netconvert/sumo/duarouter on PATH)
    OR:     Docker installed and running (script uses it automatically)
    OPTIONAL: a local copy of the Kaggle "Delivery Logistics Dataset
              (India â€“ Multi-Partner)" CSV â€” set KAGGLE_CSV_PATH, or
              place it at osm_cache/delivery_logistics_india.csv â€”
              to calibrate weight_kg and cost_per_km from real data
              instead of the hardcoded defaults.
    OPTIONAL: DATAGOVIN_API_KEY env var â€” your own free data.gov.in key,
              to avoid sharing the public sample key's rate limit.

Every expensive step is cached to disk and skipped on rerun.
"""

import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_REQUESTS = False


import numpy as np
import osmnx as ox
try:
    import sumolib
except ImportError:
    sumolib = None
from scipy.spatial import cKDTree

# =======================================================================
# 1. CONFIG
# =======================================================================
RADIUS_METERS = 3000
CACHE_DIR = os.path.abspath("osm_cache")
MATCH_DISTANCE_M = 25
DOCKER_IMAGE = "ghcr.io/eclipse-sumo/sumo:latest"

WINDOWS = [
    {"name": "night",         "warmup_min": 10, "duration_min": 60, "intensity": 0.15},
    {"name": "morning_peak",  "warmup_min": 30, "duration_min": 90, "intensity": 1.00},
    {"name": "midday",        "warmup_min": 20, "duration_min": 60, "intensity": 0.45},
    {"name": "evening_peak",  "warmup_min": 30, "duration_min": 90, "intensity": 1.10},
]
EDGE_DATA_PERIOD_S = 900  # 15-minute buckets within each window

BASE_PERIOD_S = {"car": 3.0, "motorcycle": 1.5, "auto": 8.0}  # "auto" ~ auto-rickshaw

# Free-flow (uncongested) max speed per SUMO vType, in m/s â€” MUST match the
# maxSpeed values written in write_vtypes_file(). Used as the denominator
# when fitting speed_factor from observed simulated speeds (step 5).
VTYPE_MAX_SPEED_MS = {"car": 16.6, "motorcycle": 13.8, "auto": 11.1}

# SUMO's "car" vType stands in for the LCV delivery class here â€” no separate
# LCV vType was simulated. This mapping is a simplification for calibration
# purposes, not a claim that SUMO "car" physically models an LCV; stated
# explicitly rather than left implicit.
SUMO_TO_FLEET_TYPE = {"car": "lcv", "motorcycle": "two_wheeler", "auto": "three_wheeler"}

FLEET_CALIBRATION_CACHE_NAME = "fleet_calibration.json"

# -- Customer POI tags to extract from OSM XML --
# Keys = OSM tag key, Values = set of acceptable values (or True = any value)
CUSTOMER_POI_TAGS = {
    "shop": True,              # all shop types
    "amenity": {               # specific amenities that receive deliveries
        "marketplace", "restaurant", "cafe", "fast_food", "pharmacy",
        "hospital", "clinic", "bank", "fuel", "college", "school",
        "university", "library", "community_centre",
    },
    "office": True,            # all office types
}

# Depot candidate tags. amenity=post_office/bus_station are node-tagged in
# OSM; building=warehouse/industrial and landuse=industrial are way-tagged
# â€” extract_depot_nodes() resolves both via the two-pass parser (Â§ below).
DEPOT_POI_TAGS = {
    "amenity": {"post_office", "bus_station"},
    "building": {"warehouse", "industrial"},
    "landuse": {"industrial"},
}

# -- Demand distribution parameters --
# Grounded in Kaggle "Delivery Logistics Dataset (India â€“ Multi-Partner)"
DEMAND_DISTRIBUTIONS = {
    "weight_kg": {"distribution": "lognormal", "mu": 1.0, "sigma": 0.8,
                  "min": 0.5, "max": 50.0},
    "quantity": {"distribution": "poisson", "lam": 2.5,
                 "min": 1, "max": 15},
    "time_window_width_min": {"distribution": "choice",
                              "values": [30, 60, 90, 120, 180, 240],
                              "weights": [0.05, 0.15, 0.25, 0.30, 0.15, 0.10]},
}

# -- Vehicle fleet definitions (India-specific) --
# cost_per_km_inr: Kaggle-fittable (step 8, if CSV available).
# speed_factor: SUMO-fittable (step 5, this city's own simulation).
# capacity_kg, capacity_units, fixed_cost_inr, max_range_km: always
# assumed â€” no data source in this pipeline can fit them (see docstring).
FLEET_DEFINITIONS = [
    {
        "type": "two_wheeler",
        "description": "Motorcycle/scooter courier",
        "capacity_kg": 15, "capacity_units": 5,
        "cost_per_km_inr": 4.0, "fixed_cost_inr": 150.0,
        "max_range_km": 40, "speed_factor": 1.1,
    },
    {
        "type": "three_wheeler",
        "description": "Auto-rickshaw / tempo",
        "capacity_kg": 200, "capacity_units": 30,
        "cost_per_km_inr": 8.0, "fixed_cost_inr": 300.0,
        "max_range_km": 60, "speed_factor": 0.85,
    },
    {
        "type": "lcv",
        "description": "Light commercial vehicle (Tata Ace / Ashok Leyland Dost)",
        "capacity_kg": 750, "capacity_units": 80,
        "cost_per_km_inr": 14.0, "fixed_cost_inr": 600.0,
        "max_range_km": 120, "speed_factor": 0.75,
    },
]

COST_WEIGHTS = {
    "distance_weight": 1.0,
    "time_weight": 1.5,
    "congestion_weight": 0.8,
    "vehicle_cost_weight": 1.0,
}

DEMAND_SEED = 42
MAX_CUSTOMER_NODES = 200
MIN_CUSTOMER_NODES = 20
MAX_DEPOT_NODES = 15
MIN_DEPOT_NODES = 2

# -- India Post data.gov.in API config --
_DATAGOVIN_SAMPLE_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
DATAGOVIN_API_KEY = os.environ.get("DATAGOVIN_API_KEY", _DATAGOVIN_SAMPLE_KEY)
if DATAGOVIN_API_KEY == _DATAGOVIN_SAMPLE_KEY:
    print("Using data.gov.in's shared public sample key (no DATAGOVIN_API_KEY "
          "env var set) â€” register your own free key at data.gov.in if you hit "
          "rate limits, then `export DATAGOVIN_API_KEY=<your key>`.")
DATAGOVIN_PINCODE_RESOURCE = "6176ee09-3d56-4a3b-8115-21841576b2f6"
DATAGOVIN_PAGE_SIZE = 100
DATAGOVIN_MAX_PAGES = 20
INDIA_POST_CACHE_FILE = "india_post_offices.json"
DATAGOVIN_OFFICETYPE_ALLOWLIST = {"head post office", "sub post office"}
MAX_OFFICES_TO_GEOCODE = 30


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower())


def run_if_missing(output_path, fn):
    if os.path.exists(output_path):
        print(f"  [cached] {output_path} already exists â€” skipping")
    else:
        fn()
    return output_path


# =======================================================================
# 2. EXECUTION MODE: native SUMO vs Docker (auto-detected once)
# =======================================================================
def is_docker_active():
    if not shutil.which("docker"):
        return False
    try:
        res = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
        return res.returncode == 0
    except Exception:
        return False

def detect_execution_mode():
    forced = os.environ.get("SUMO_MODE")
    if forced == "native":
        return False
    if forced == "docker":
        return True

    if shutil.which("netconvert") and shutil.which("sumo") and shutil.which("duarouter"):
        print("Detected native SUMO installation — using it directly.")
        return False

    if is_docker_active():
        print(f"Native SUMO not found on PATH — using Docker instead ({DOCKER_IMAGE}).")
        return True

    return False


USE_DOCKER = False



def run_sumo(kind, name, args, city_dir):
    """
    kind: "bin"  -> compiled SUMO binary (netconvert, sumo, duarouter)
          "tool" -> python script under $SUMO_HOME/tools (randomTrips.py, osmGet.py)
    """
    if USE_DOCKER:
        if kind == "bin":
            cmd = ["docker", "run", "--rm", "-v", f"{city_dir}:{city_dir}",
                   "-w", city_dir, DOCKER_IMAGE, name] + args
        else:
            inner = f'python3 "$SUMO_HOME/tools/{name}" ' + " ".join(shlex.quote(a) for a in args)
            cmd = ["docker", "run", "--rm", "-v", f"{city_dir}:{city_dir}",
                   "-w", city_dir, DOCKER_IMAGE, "bash", "-lc", inner]
        subprocess.run(cmd, check=True)
    else:
        if kind == "bin":
            subprocess.run([name] + args, check=True, cwd=city_dir)
        else:
            sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
            script_path = os.path.join(sumo_home, "tools", name)
            if not os.path.exists(script_path):
                sys.exit(f"ERROR: {name} not found at '{script_path}'. "
                         f"Set SUMO_HOME correctly, or set SUMO_MODE=docker.")
            subprocess.run([sys.executable, script_path] + args, check=True, cwd=city_dir)


# =======================================================================
# 3. OSMnx GRAPH (fetch + cache)
# =======================================================================
def get_osmnx_graph(city_name, city_dir):
    cache_file = os.path.join(city_dir, "road_network.graphml")

    def fetch():
        print(f"  Fetching OSMnx graph within {RADIUS_METERS}m of '{city_name}' ...")
        G = ox.graph_from_address(city_name, dist=RADIUS_METERS, network_type="drive")
        G = ox.add_edge_speeds(G, fallback=30)
        G = ox.add_edge_travel_times(G)
        ox.save_graphml(G, cache_file)

    run_if_missing(cache_file, fetch)
    G = ox.load_graphml(cache_file)
    for _, _, d in G.edges(data=True):
        if "travel_time" in d:
            d["travel_time"] = float(d["travel_time"])
    return G


def graph_bbox(G):
    lats = [d["y"] for _, d in G.nodes(data=True)]
    lons = [d["x"] for _, d in G.nodes(data=True)]
    pad = 0.002
    return min(lats) - pad, min(lons) - pad, max(lats) + pad, max(lons) + pad


# =======================================================================
# 4. RAW OSM DATA -> SUMO NETWORK
# =======================================================================
def download_osm_xml(bbox, city_dir):
    import glob
    prefix = "osm"
    url_attempts = [None, "overpass.kumi.systems/api/interpreter", "sumo.dlr.de/osm/api/interpreter"]

    def already_downloaded():
        matches = glob.glob(os.path.join(city_dir, f"{prefix}*.osm.xml"))
        return matches[0] if matches else None

    existing = already_downloaded()
    if existing:
        print(f"  [cached] {existing} already exists â€” skipping")
        return os.path.basename(existing)

    south, west, north, east = bbox
    bbox_str = f"{west},{south},{east},{north}"

    for url in url_attempts:
        label = url or "(default server)"
        print(f"  Downloading raw OSM data via {label} (bbox {bbox_str}) ...")
        args = ["--bbox", bbox_str, "--prefix", prefix, "--output-dir", city_dir]
        if url:
            args += ["--url", url]
        try:
            run_sumo("tool", "osmGet.py", args, city_dir)
            found = already_downloaded()
            if found:
                return os.path.basename(found)
        except subprocess.CalledProcessError:
            print(f"    failed via {label} â€” trying next server ...")

    sys.exit("ERROR: could not download OSM data from any server after trying "
              f"{len(url_attempts)} mirrors. This is usually transient â€” wait a bit "
              "and rerun (already-completed steps are cached, so this is the only "
              "step that will retry).")


def build_sumo_network(osm_xml_name, city_dir):
    net_name = "network.net.xml"
    net_path = os.path.join(city_dir, net_name)

    def build():
        print("  Running netconvert ...")
        run_sumo("bin", "netconvert",
                 ["--osm-files", osm_xml_name, "-o", net_name,
                  "--geometry.remove", "--roundabouts.guess", "--junctions.join"],
                 city_dir)

    run_if_missing(net_path, build)
    return net_name


# =======================================================================
# 5. TIME-WINDOWED, MULTI-VEHICLE-TYPE DEMAND + SIMULATION
# =======================================================================
def write_vtypes_file(city_dir):
    name = "vtypes.xml"
    path = os.path.join(city_dir, name)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("""<additional>
  <vType id="car" vClass="passenger" length="4.5" maxSpeed="16.6" accel="2.6" decel="4.5"/>
  <vType id="motorcycle" vClass="motorcycle" length="1.8" maxSpeed="13.8" accel="3.5" decel="5.0"/>
  <vType id="auto" vClass="taxi" length="2.6" maxSpeed="11.1" accel="2.0" decel="4.0"/>
</additional>""")
    return name


def generate_window_routes(net_name, vtypes_name, window, city_dir):
    total_duration_s = (window["warmup_min"] + window["duration_min"]) * 60
    route_names = []

    for vtype, base_period in BASE_PERIOD_S.items():
        period = round(base_period / max(window["intensity"], 0.05), 2)
        out_name = f"{window['name']}_{vtype}.rou.xml"
        out_path = os.path.join(city_dir, out_name)

        def fetch(vtype=vtype, period=period, out_name=out_name):
            print(f"  Generating {window['name']} demand for '{vtype}' "
                  f"(period={period}s) ...")
            run_sumo("tool", "randomTrips.py",
                     ["-n", net_name, "-r", out_name,
                      "--begin", "0", "--end", str(total_duration_s),
                      "--period", str(period),
                      "--trip-attributes", f'type="{vtype}"',
                      "--additional-files", vtypes_name,
                      "--prefix", f"{window['name']}_{vtype}_",
                      "--seed", "42"],
                     city_dir)

        run_if_missing(out_path, fetch)
        route_names.append(out_name)

    return route_names, total_duration_s


def run_sumo_window(net_name, route_names, window, total_duration_s, city_dir):
    """
    Runs one window's simulation, producing BOTH:
      - edgedata (time-bucketed per-edge congestion, used by step 10)
      - tripinfo (per-vehicle actual trip duration/distance, used by
        step 5's fleet speed calibration â€” NEW in v6)

    The cache check requires BOTH files to exist before skipping â€” a city
    directory cached under v5 (edgedata only) will correctly regenerate
    once under v6, rather than silently having no tripinfo data forever.
    """
    warmup_s = window["warmup_min"] * 60
    edgedata_name = f"{window['name']}_edgedata.xml"
    edgedata_path = os.path.join(city_dir, edgedata_name)
    tripinfo_name = f"{window['name']}_tripinfo.xml"
    tripinfo_path = os.path.join(city_dir, tripinfo_name)
    additional_name = f"{window['name']}_additional.xml"
    additional_path = os.path.join(city_dir, additional_name)

    with open(additional_path, "w") as f:
        f.write(f"""<additional>
  <edgeData id="ed_{window['name']}" file="{edgedata_name}"
            begin="{warmup_s}" end="{total_duration_s}" period="{EDGE_DATA_PERIOD_S}"/>
</additional>""")

    def run():
        print(f"  Running SUMO for '{window['name']}' "
              f"({window['warmup_min']}min warmup + {window['duration_min']}min recorded) ...")
        run_sumo("bin", "sumo",
                 ["-n", net_name, "-r", ",".join(route_names),
                  "-a", additional_name, "--begin", "0", "--end", str(total_duration_s),
                  "--tripinfo-output", tripinfo_name,
                  "--no-step-log", "true"],
                 city_dir)

    if os.path.exists(edgedata_path) and os.path.exists(tripinfo_path):
        print(f"  [cached] {edgedata_path} and {tripinfo_path} already exist â€” skipping")
    else:
        run()

    return edgedata_name, warmup_s, tripinfo_name


# =======================================================================
# 5b. FLEET SPEED CALIBRATION FROM SUMO TRIPINFO  (NEW in v6)
# =======================================================================
def calibrate_fleet_from_tripinfo(all_tripinfo_paths, city_dir):
    """
    Fits FLEET_DEFINITIONS' speed_factor per vehicle type from REAL
    simulated per-vehicle trip data â€” the gap flagged as unaddressed in
    v5 ("vehicle fleet defined but not tied to real per-vehicle route
    data").

    speed_factor = (observed average simulated speed for that SUMO vType,
    pooled across all time windows for this city) / (that vType's
    free-flow max speed). This is genuine per-vehicle-type route data â€”
    each SUMO tripinfo record is one simulated vehicle's actual
    (routeLength / duration), not a guess.

    capacity_kg, capacity_units, fixed_cost_inr, and max_range_km stay
    assumed â€” the same discipline v5's Kaggle fit applied to capacity_kg:
    a short simulated trip's distance/duration says nothing about a
    vehicle's payload capacity, fixed costs, or full-tank/battery range.
    Only speed is legitimately derivable from this data.

    Cached PER-CITY (unlike kaggle_calibration.json, which is global) â€”
    traffic, and therefore observed speed, genuinely differs by city.
    """
    output_path = os.path.join(city_dir, FLEET_CALIBRATION_CACHE_NAME)

    def generate():
        print("  Parsing SUMO tripinfo output for fleet speed calibration ...")
        speeds_kmh = {vtype: [] for vtype in VTYPE_MAX_SPEED_MS}
        route_lengths_km = {vtype: [] for vtype in VTYPE_MAX_SPEED_MS}
        total_trips = 0

        for window_name, tripinfo_path in all_tripinfo_paths.items():
            if not os.path.exists(tripinfo_path):
                continue
            prefix = window_name + "_"
            tree = ET.parse(tripinfo_path)
            for trip in tree.getroot().findall("tripinfo"):
                trip_id = trip.get("id", "")
                if not trip_id.startswith(prefix):
                    continue  # shouldn't happen â€” don't misattribute if it does
                vtype = trip_id[len(prefix):].split("_")[0]
                if vtype not in speeds_kmh:
                    continue

                duration_s = float(trip.get("duration", 0))
                length_m = float(trip.get("routeLength", 0))
                if duration_s <= 0 or length_m <= 0:
                    continue

                speeds_kmh[vtype].append((length_m / 1000) / (duration_s / 3600))
                route_lengths_km[vtype].append(length_m / 1000)
                total_trips += 1

        calibration = {"source": "sumo_tripinfo", "total_trips_parsed": total_trips,
                        "vehicles": {}, "warnings": []}

        for vtype, freeflow_ms in VTYPE_MAX_SPEED_MS.items():
            fleet_type = SUMO_TO_FLEET_TYPE[vtype]
            n = len(speeds_kmh[vtype])
            if n == 0:
                calibration["warnings"].append(
                    f"no valid tripinfo trips found for SUMO vType '{vtype}' "
                    f"(fleet type '{fleet_type}') â€” its speed_factor keeps the "
                    "hardcoded default")
                continue

            avg_speed_kmh = sum(speeds_kmh[vtype]) / n
            freeflow_kmh = freeflow_ms * 3.6
            speed_factor = round(avg_speed_kmh / freeflow_kmh, 3)
            # Sane clip â€” observed average shouldn't exceed free-flow except
            # for small numerical/rounding noise near-zero-congestion edges.
            speed_factor = max(0.05, min(speed_factor, 1.05))

            calibration["vehicles"][vtype] = {
                "fleet_type": fleet_type,
                "n_trips": n,
                "avg_observed_speed_kmh": round(avg_speed_kmh, 2),
                "avg_route_length_km": round(sum(route_lengths_km[vtype]) / n, 2),
                "reference_freeflow_speed_kmh": round(freeflow_kmh, 2),
                "speed_factor_fitted": speed_factor,
            }

        with open(output_path, "w") as f:
            json.dump(calibration, f, indent=2)

        print(f"  Fleet speed calibration from {total_trips} simulated trips:")
        for vtype, stats in calibration["vehicles"].items():
            print(f"    {vtype} -> {stats['fleet_type']}: "
                  f"{stats['avg_observed_speed_kmh']} km/h avg "
                  f"({stats['n_trips']} trips) -> speed_factor={stats['speed_factor_fitted']}")
        for w in calibration["warnings"]:
            print(f"    WARNING: {w}")

    run_if_missing(output_path, generate)
    with open(output_path) as f:
        return json.load(f)


# =======================================================================
# 6a. LOCAL OSM XML POI PARSER (nodes)
# =======================================================================
def _tag_matches(tag_dict, tag_spec):
    """
    tag_spec: {key: True} means any value for that key
              {key: {val1, val2, ...}} means value must be in the set
    Returns the first matching (key, value) pair or None.
    """
    for spec_key, spec_values in tag_spec.items():
        if spec_key in tag_dict:
            actual_value = tag_dict[spec_key]
            if spec_values is True or actual_value in spec_values:
                return spec_key, actual_value
    return None


def parse_pois_from_osm_xml(osm_xml_path, tag_spec):
    """
    Parse POIs from a local OSM XML file using streaming iterparse.
    Only processes <node> elements. Way-tagged POIs (e.g. building=
    warehouse) are NOT found here â€” use parse_ways_from_osm_xml() for
    those (Â§6b, new in v6).

    Returns list of dicts: {lon, lat, name, category, osm_id}
    """
    pois = []
    context = ET.iterparse(osm_xml_path, events=("end",))

    for event, elem in context:
        if elem.tag != "node":
            if elem.tag in ("way", "relation"):
                elem.clear()
            continue

        tags = elem.findall("tag")
        if not tags:
            elem.clear()
            continue

        tag_dict = {t.get("k"): t.get("v") for t in tags}
        match = _tag_matches(tag_dict, tag_spec)

        if match is not None:
            key, value = match
            lat_str = elem.get("lat")
            lon_str = elem.get("lon")

            if lat_str and lon_str:
                lat = float(lat_str)
                lon = float(lon_str)
                name = (tag_dict.get("name") or tag_dict.get("name:en")
                        or tag_dict.get("brand") or tag_dict.get("operator"))

                pois.append({
                    "lon": lon,
                    "lat": lat,
                    "name": name,
                    "category": f"{key}:{value}",
                    "osm_id": elem.get("id"),
                })

        elem.clear()

    return pois


# =======================================================================
# 6b. TWO-PASS WAY-RESOLUTION PARSER  (NEW in v6 â€” fixes the v5 bug)
# =======================================================================
def _collect_all_node_coords(osm_xml_path):
    """
    Pass 1 of the two-pass way-resolution parser: collect (lon, lat) for
    EVERY node in the file, tagged or not.

    Why every node, not just tagged ones: a <way> element references its
    shape via <nd ref="..."/> children pointing at node IDs â€” it carries
    no coordinates of its own. Resolving where a tag-matching way (e.g.
    building=warehouse) actually is requires looking up its referenced
    nodes' coordinates, and a way can reference any node in the file,
    tagged or not (most way-boundary nodes carry no tags at all).
    """
    coords = {}
    context = ET.iterparse(osm_xml_path, events=("end",))
    for event, elem in context:
        if elem.tag == "node":
            node_id = elem.get("id")
            lat_str, lon_str = elem.get("lat"), elem.get("lon")
            if node_id and lat_str and lon_str:
                coords[node_id] = (float(lon_str), float(lat_str))
            elem.clear()
        elif elem.tag in ("way", "relation"):
            elem.clear()
    return coords


def parse_ways_from_osm_xml(osm_xml_path, tag_spec, node_coords):
    """
    Pass 2 of the two-pass way-resolution parser: find <way> elements
    whose tags match tag_spec (e.g. building=warehouse, landuse=
    industrial â€” tags that live on ways, not nodes, in OSM's data model)
    and resolve each match's location as the centroid of its referenced
    nodes' coordinates (from _collect_all_node_coords, pass 1).

    This is the fix for the bug documented in pipeline_v5-docs.md Â§17:
    "Depot building/landuse OSM tags don't actually match anything â€” the
    parser only reads <node> elements." Those tags are found here now.

    Relations (multi-polygon land-use areas) are still NOT resolved â€” a
    rarer, smaller gap than the way case, left as a stated limitation
    rather than silently claimed to be covered.

    Returns list of dicts in the same shape as parse_pois_from_osm_xml's
    node output ({lon, lat, name, category, osm_id}), so callers can
    merge the two freely.
    """
    pois = []
    context = ET.iterparse(osm_xml_path, events=("end",))

    for event, elem in context:
        if elem.tag != "way":
            if elem.tag == "node":
                elem.clear()
            continue

        tags = elem.findall("tag")
        if not tags:
            elem.clear()
            continue

        tag_dict = {t.get("k"): t.get("v") for t in tags}
        match = _tag_matches(tag_dict, tag_spec)

        if match is not None:
            refs = [nd.get("ref") for nd in elem.findall("nd")]
            resolved = [node_coords[r] for r in refs if r in node_coords]

            if resolved:
                key, value = match
                avg_lon = sum(c[0] for c in resolved) / len(resolved)
                avg_lat = sum(c[1] for c in resolved) / len(resolved)
                name = (tag_dict.get("name") or tag_dict.get("name:en")
                        or tag_dict.get("brand") or tag_dict.get("operator"))
                pois.append({
                    "lon": avg_lon,
                    "lat": avg_lat,
                    "name": name,
                    "category": f"{key}:{value}",
                    "osm_id": f"way/{elem.get('id')}",
                })
            # else: every referenced node fell outside the downloaded
            # bbox (way straddles the edge) â€” silently skipped, not an
            # error; happens occasionally near the radius boundary.

        elem.clear()

    return pois


# =======================================================================
# 6c. CUSTOMER/DEMAND NODE GENERATION
# =======================================================================
def _snap_to_graph_node(G, lon, lat):
    node_ids, dists = _snap_batch_to_graph(G, [lon], [lat])
    return node_ids[0], dists[0]


def _snap_batch_to_graph(G, lons, lats):
    """Snap a batch of (lon, lat) pairs to nearest graph nodes via cKDTree."""
    graph_node_ids = list(G.nodes())
    graph_coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in graph_node_ids]
    tree = cKDTree(graph_coords)

    query_coords = list(zip(lons, lats))
    _, indices = tree.query(query_coords)

    node_ids = [graph_node_ids[idx] for idx in indices]
    distances = [
        haversine_m(lons[i], lats[i],
                    G.nodes[node_ids[i]]["x"], G.nodes[node_ids[i]]["y"])
        for i in range(len(lons))
    ]
    return node_ids, distances


def _sample_demand(rng, n, demand_distributions=None):
    dd = demand_distributions if demand_distributions is not None else DEMAND_DISTRIBUTIONS

    w_cfg = dd["weight_kg"]
    weights_kg = rng.lognormal(w_cfg["mu"], w_cfg["sigma"], n)
    weights_kg = np.clip(weights_kg, w_cfg["min"], w_cfg["max"])
    weights_kg = np.round(weights_kg, 2)

    q_cfg = dd["quantity"]
    quantities = rng.poisson(q_cfg["lam"], n)
    quantities = np.clip(quantities, q_cfg["min"], q_cfg["max"]).astype(int)

    tw_cfg = dd["time_window_width_min"]
    tw_widths = rng.choice(tw_cfg["values"], size=n, p=tw_cfg["weights"])

    return weights_kg, quantities, tw_widths


def _assign_time_windows(rng, n, tw_widths):
    earliest_possible = 360   # 6:00 AM
    latest_possible = 1320    # 10:00 PM

    time_windows = []
    for i in range(n):
        width = int(tw_widths[i])
        max_start = max(latest_possible - width, earliest_possible)
        start = int(rng.integers(earliest_possible, max_start + 1))
        time_windows.append([start, start + width])

    return time_windows


def extract_customer_nodes(G, osm_xml_path, city_name, city_dir, demand_distributions=None):
    """Extract customer/demand nodes from the local OSM XML file (node-only
    â€” see docstring history for why way-resolution wasn't extended here)."""
    output_path = os.path.join(city_dir, "demand_nodes.json")
    dd = demand_distributions if demand_distributions is not None else DEMAND_DISTRIBUTIONS

    def generate():
        rng = np.random.default_rng(DEMAND_SEED)

        print(f"  Parsing POIs from local OSM XML ({os.path.basename(osm_xml_path)}) ...")
        pois = parse_pois_from_osm_xml(osm_xml_path, CUSTOMER_POI_TAGS)
        print(f"  Found {len(pois)} customer-relevant POIs (zero network calls)")

        seen_coords = set()
        unique_pois = []
        for poi in pois:
            coord_key = (round(poi["lon"], 5), round(poi["lat"], 5))
            if coord_key not in seen_coords:
                seen_coords.add(coord_key)
                unique_pois.append(poi)
        if len(unique_pois) < len(pois):
            print(f"  Deduplicated: {len(pois)} â†’ {len(unique_pois)} unique locations")
        pois = unique_pois

        if len(pois) < MIN_CUSTOMER_NODES:
            deficit = MIN_CUSTOMER_NODES - len(pois)
            print(f"  Supplementing with {deficit} random graph nodes "
                  f"(POI count {len(pois)} < minimum {MIN_CUSTOMER_NODES})")
            all_nodes = list(G.nodes(data=True))
            indices = rng.choice(len(all_nodes), size=min(deficit, len(all_nodes)),
                                 replace=False)
            for idx in indices:
                node_id, data = all_nodes[idx]
                pois.append({
                    "lon": data["x"], "lat": data["y"],
                    "name": f"synthetic_{node_id}",
                    "category": "synthetic:random_node",
                    "osm_id": str(node_id),
                })

        if len(pois) > MAX_CUSTOMER_NODES:
            print(f"  Capping: {len(pois)} â†’ {MAX_CUSTOMER_NODES} customer nodes (random sample)")
            indices = rng.choice(len(pois), size=MAX_CUSTOMER_NODES, replace=False)
            pois = [pois[i] for i in indices]

        n = len(pois)
        print(f"  Generating demand for {n} customer nodes ...")

        weights_kg, quantities, tw_widths = _sample_demand(rng, n, dd)
        time_windows = _assign_time_windows(rng, n, tw_widths)
        service_times = np.clip(rng.normal(5, 2, n), 2, 15).astype(int)

        lons = [p["lon"] for p in pois]
        lats = [p["lat"] for p in pois]
        graph_nodes, snap_dists = _snap_batch_to_graph(G, lons, lats)

        nodes = []
        for i in range(n):
            nodes.append({
                "id": i,
                "name": pois[i]["name"] or f"poi_{i}",
                "category": pois[i]["category"],
                "osm_id": pois[i].get("osm_id"),
                "lon": round(pois[i]["lon"], 6),
                "lat": round(pois[i]["lat"], 6),
                "graph_node": int(graph_nodes[i]),
                "snap_distance_m": round(snap_dists[i], 1),
                "demand_kg": float(weights_kg[i]),
                "demand_units": int(quantities[i]),
                "time_window": time_windows[i],
                "service_time_min": int(service_times[i]),
            })

        output = {
            "city": city_name,
            "total_customers": n,
            "demand_seed": DEMAND_SEED,
            "source": f"Parsed from local {os.path.basename(osm_xml_path)} (zero network IO)",
            "distributions_used": dd,
            "customers": nodes,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        total_kg = sum(nd["demand_kg"] for nd in nodes)
        total_units = sum(nd["demand_units"] for nd in nodes)
        avg_tw = np.mean([tw[1] - tw[0] for tw in time_windows])
        cat_counts = {}
        for nd in nodes:
            key = nd["category"].split(":")[0]
            cat_counts[key] = cat_counts.get(key, 0) + 1
        cat_summary = ", ".join(f"{k}={v}" for k, v in sorted(cat_counts.items(),
                                                                key=lambda x: -x[1]))

        print(f"  Saved {n} customer nodes to {os.path.basename(output_path)}")
        print(f"  Summary: {total_kg:.1f} kg total demand, "
              f"{total_units} units, "
              f"avg time window: {avg_tw:.0f} min")
        print(f"  Categories: {cat_summary}")

    run_if_missing(output_path, generate)
    return output_path


# =======================================================================
# 7a. INDIA POST DATA.GOV.IN DEPOT ENRICHMENT
# =======================================================================
def _datagov_api_get(resource_id, api_key, offset=0, limit=100, filters=None):
    base_url = f"https://api.data.gov.in/resource/{resource_id}"
    params = {
        "api-key": api_key,
        "format": "json",
        "offset": str(offset),
        "limit": str(limit),
    }
    if filters:
        for k, v in filters.items():
            params[f"filters[{k}]"] = v

    if HAS_REQUESTS:
        try:
            resp = requests.get(base_url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"    data.gov.in API error: {e}")
            return None
    else:
        try:
            query = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
            url = f"{base_url}?{query}"
            with urllib.request.urlopen(url, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            print(f"    data.gov.in API error: {e}")
            return None


def _detect_state_from_graph(G):
    try:
        lats = [d["y"] for _, d in G.nodes(data=True)]
        lons = [d["x"] for _, d in G.nodes(data=True)]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        import urllib.request
        import urllib.parse
        url = (f"https://nominatim.openstreetmap.org/reverse?"
               f"lat={center_lat}&lon={center_lon}&format=json&zoom=10")
        req = urllib.request.Request(url, headers={"User-Agent": "QPSO-VRP-Pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        address = data.get("address", {})
        state = address.get("state")
        district = (address.get("state_district")
                    or address.get("county")
                    or address.get("city"))
        return state, district
    except Exception as e:
        print(f"    State detection warning: {e}")
        return None, None


def fetch_india_post_offices(G, city_dir):
    cache_path = os.path.join(city_dir, INDIA_POST_CACHE_FILE)

    if os.path.exists(cache_path):
        print(f"  [cached] India Post data already downloaded")
        with open(cache_path) as f:
            return json.load(f)

    print("  Detecting state/district from graph center ...")
    state, district = _detect_state_from_graph(G)

    if not state:
        print("  Could not detect state â€” skipping India Post API")
        return []

    print(f"  Detected: state='{state}', district='{district}'")
    print(f"  Fetching India Post offices from data.gov.in ...")

    filters = {"statename": state.upper()}
    if district:
        filters["districtname"] = district.title()

    all_offices = []
    for page in range(DATAGOVIN_MAX_PAGES):
        offset = page * DATAGOVIN_PAGE_SIZE
        data = _datagov_api_get(
            DATAGOVIN_PINCODE_RESOURCE, DATAGOVIN_API_KEY,
            offset=offset, limit=DATAGOVIN_PAGE_SIZE, filters=filters,
        )

        if data is None or data.get("status") != "ok":
            if district and page == 0:
                print(f"    District filter failed â€” retrying with state only")
                filters = {"statename": state.upper()}
                data = _datagov_api_get(
                    DATAGOVIN_PINCODE_RESOURCE, DATAGOVIN_API_KEY,
                    offset=offset, limit=DATAGOVIN_PAGE_SIZE, filters=filters,
                )
            if data is None or data.get("status") != "ok":
                print(f"    API returned error or no data â€” stopping")
                break

        records = data.get("records", [])
        if not records:
            break

        for rec in records:
            officetype = rec.get("officetype", "").strip()
            if officetype.lower() not in DATAGOVIN_OFFICETYPE_ALLOWLIST:
                continue
            all_offices.append({
                "name": rec.get("officename", "").strip(),
                "pincode": rec.get("pincode", "").strip(),
                "officetype": officetype,
                "district": rec.get("districtname", "").strip(),
                "state": rec.get("statename", "").strip(),
                "division": rec.get("divisionname", "").strip(),
                "region": rec.get("regionname", "").strip(),
            })

        total = int(data.get("total", 0))
        if offset + DATAGOVIN_PAGE_SIZE >= total:
            break

        import time
        time.sleep(0.5)

    print(f"  Fetched {len(all_offices)} post offices from data.gov.in")

    with open(cache_path, "w") as f:
        json.dump(all_offices, f, indent=2)

    return all_offices


def _geocode_post_offices_in_bbox(offices, bbox, G):
    south, west, north, east = bbox
    geocoded = []
    seen_coords = set()

    import urllib.request
    import urllib.parse
    import time

    if len(offices) > MAX_OFFICES_TO_GEOCODE:
        print(f"    Capping geocoding at {MAX_OFFICES_TO_GEOCODE} offices "
              f"(district had {len(offices)} Head/Sub Post Offices) to bound "
              "Nominatim's ~1 req/sec rate limit")
        offices = offices[:MAX_OFFICES_TO_GEOCODE]

    for office in offices:
        name = office["name"]
        if not name:
            continue

        query = f"{name} Post Office, {office['district']}, {office['state']}, India"
        try:
            encoded = urllib.parse.quote(query)
            url = (f"https://nominatim.openstreetmap.org/search?"
                   f"q={encoded}&format=json&limit=1&countrycodes=in")
            req = urllib.request.Request(url, headers={"User-Agent": "QPSO-VRP-Pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                results = json.loads(resp.read().decode())
        except Exception:
            continue

        if not results:
            continue

        lat = float(results[0]["lat"])
        lon = float(results[0]["lon"])

        if not (south <= lat <= north and west <= lon <= east):
            continue

        coord_key = (round(lon, 5), round(lat, 5))
        if coord_key in seen_coords:
            continue
        seen_coords.add(coord_key)

        geocoded.append({
            "lon": lon, "lat": lat,
            "name": f"{name} Post Office",
            "category": f"india_post:{office['officetype'].lower()}",
            "osm_id": None,
            "pincode": office["pincode"],
            "source": "data.gov.in",
        })

        time.sleep(1.1)

    return geocoded


def extract_depot_nodes(G, osm_xml_path, city_name, bbox, city_dir):
    """
    Extract depot/hub candidates from TWO sources:
    1. Local OSM XML â€” nodes (post offices, bus stations) AND, as of v6,
       ways (warehouses, industrial zones) via the two-pass parser (Â§6b).
       In v5 the way-tagged half of this source silently found nothing;
       fixed here.
    2. India Post data.gov.in API (cached, geocoded via Nominatim)

    Sources are merged with deduplication: an India Post office within
    100m of an already-found OSM depot is skipped.

    Falls back to high-degree graph nodes if fewer than MIN_DEPOT_NODES.
    Outputs depot_nodes.json.
    """
    output_path = os.path.join(city_dir, "depot_nodes.json")

    def generate():
        rng = np.random.default_rng(DEMAND_SEED + 1)

        # --- Source 1: Local OSM XML â€” nodes + ways (v6 fix) ---
        print("  [Source 1] Parsing depot candidates from local OSM XML "
              "(nodes + ways) ...")
        osm_node_pois = parse_pois_from_osm_xml(osm_xml_path, DEPOT_POI_TAGS)

        node_coords = _collect_all_node_coords(osm_xml_path)
        osm_way_pois = parse_ways_from_osm_xml(osm_xml_path, DEPOT_POI_TAGS, node_coords)

        osm_pois = osm_node_pois + osm_way_pois
        for p in osm_pois:
            p["source"] = "osm"
        print(f"  Found {len(osm_pois)} depot POIs from OSM "
              f"({len(osm_node_pois)} nodes + {len(osm_way_pois)} ways â€” "
              "warehouse/industrial ways now resolved, zero network calls)")

        # --- Source 2: India Post data.gov.in API ---
        print("  [Source 2] India Post offices from data.gov.in ...")
        india_post_pois = []
        try:
            offices = fetch_india_post_offices(G, city_dir)
            if offices:
                print(f"  Geocoding India Post offices within city bbox ...")
                india_post_pois = _geocode_post_offices_in_bbox(offices, bbox, G)
                print(f"  Geocoded {len(india_post_pois)} India Post offices within bbox")
        except Exception as e:
            print(f"  India Post API/geocoding failed ({e}) â€” continuing with OSM only")

        # --- Merge with deduplication ---
        all_pois = list(osm_pois)

        if india_post_pois:
            if osm_pois:
                osm_coords = [(p["lon"], p["lat"]) for p in osm_pois]
                osm_tree = cKDTree(osm_coords)
                dedup_threshold_deg = 100 / 111000
            else:
                osm_tree = None

            added = 0
            skipped = 0
            for ip_poi in india_post_pois:
                if osm_tree is not None:
                    dist, _ = osm_tree.query((ip_poi["lon"], ip_poi["lat"]))
                    if dist < dedup_threshold_deg:
                        skipped += 1
                        continue
                all_pois.append(ip_poi)
                added += 1

            if skipped:
                print(f"  Dedup: {skipped} India Post offices already in OSM, "
                      f"{added} new ones added")
            else:
                print(f"  Added {added} new India Post depots")

        seen = set()
        unique = []
        for poi in all_pois:
            ck = (round(poi["lon"], 5), round(poi["lat"], 5))
            if ck not in seen:
                seen.add(ck)
                unique.append(poi)
        all_pois = unique

        if len(all_pois) < MIN_DEPOT_NODES:
            deficit = MIN_DEPOT_NODES - len(all_pois)
            print(f"  Supplementing with {deficit} high-degree graph nodes as depot proxies")

            existing_snapped = set()
            for poi in all_pois:
                nids, _ = _snap_batch_to_graph(G, [poi["lon"]], [poi["lat"]])
                existing_snapped.add(nids[0])

            node_degrees = sorted(
                [(n, G.degree(n)) for n in G.nodes()],
                key=lambda x: x[1], reverse=True,
            )
            added = 0
            for node_id, degree in node_degrees:
                if node_id in existing_snapped:
                    continue
                data = G.nodes[node_id]
                all_pois.append({
                    "lon": data["x"], "lat": data["y"],
                    "name": f"hub_node_{node_id}",
                    "category": "synthetic:high_degree_node",
                    "osm_id": str(node_id),
                    "source": "synthetic",
                })
                existing_snapped.add(node_id)
                added += 1
                if added >= deficit:
                    break

        if len(all_pois) > MAX_DEPOT_NODES:
            print(f"  Capping: {len(all_pois)} â†’ {MAX_DEPOT_NODES} depot candidates")
            osm_deps = [p for p in all_pois if p.get("source") == "osm"]
            ip_deps = [p for p in all_pois if p.get("source") == "data.gov.in"]
            synth_deps = [p for p in all_pois if p.get("source") == "synthetic"]
            all_pois = (osm_deps + ip_deps + synth_deps)[:MAX_DEPOT_NODES]

        n = len(all_pois)

        lons = [p["lon"] for p in all_pois]
        lats = [p["lat"] for p in all_pois]
        graph_nodes, snap_dists = _snap_batch_to_graph(G, lons, lats)

        depots = []
        for i in range(n):
            cat = all_pois[i]["category"]
            if "warehouse" in cat or "industrial" in cat:
                cap_kg, cap_units = 5000, 500
            elif "post_office" in cat or "india_post" in cat:
                cap_kg, cap_units = 2000, 200
            elif "bus_station" in cat:
                cap_kg, cap_units = 3000, 300
            else:
                cap_kg, cap_units = 3000, 300

            depots.append({
                "id": i,
                "name": all_pois[i]["name"] or f"depot_{i}",
                "category": cat,
                "source": all_pois[i].get("source", "unknown"),
                "osm_id": all_pois[i].get("osm_id"),
                "pincode": all_pois[i].get("pincode"),
                "lon": round(all_pois[i]["lon"], 6),
                "lat": round(all_pois[i]["lat"], 6),
                "graph_node": int(graph_nodes[i]),
                "snap_distance_m": round(snap_dists[i], 1),
                "capacity_kg": cap_kg,
                "capacity_units": cap_units,
            })

        src_counts = {}
        for d in depots:
            src_counts[d["source"]] = src_counts.get(d["source"], 0) + 1
        src_summary = ", ".join(f"{k}={v}" for k, v in sorted(src_counts.items()))

        output = {
            "city": city_name,
            "total_depots": n,
            "sources": ["osm (local XML â€” nodes + ways)", "data.gov.in India Post API"],
            "source_counts": src_counts,
            "depots": depots,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"  Saved {n} depot candidates to {os.path.basename(output_path)}")
        print(f"  Sources: {src_summary}")
        for d in depots:
            print(f"    [{d['id']}] {d['name']} ({d['category']}) "
                  f"[{d['source']}] â€” {d['capacity_kg']}kg")

    run_if_missing(output_path, generate)
    return output_path


# =======================================================================
# 8a. KAGGLE-CSV DEMAND/COST CALIBRATION
# =======================================================================
KAGGLE_CSV_PATH = os.environ.get(
    "KAGGLE_CSV_PATH", os.path.join(CACHE_DIR, "delivery_logistics_india.csv"))
KAGGLE_CALIBRATION_CACHE = os.path.join(CACHE_DIR, "kaggle_calibration.json")

_VEHICLE_KEYWORDS = {
    "two_wheeler":   ["bike", "motorcycle", "scooter", "two wheeler", "two-wheeler"],
    "three_wheeler": ["auto", "three wheeler", "three-wheeler", "rickshaw", "tempo"],
    "lcv":           ["van", "truck", "mini truck", "lcv", "four wheeler", "four-wheeler", "car"],
}


def _match_vehicle_bucket(raw_value):
    v = str(raw_value).lower()
    for bucket, keywords in _VEHICLE_KEYWORDS.items():
        if any(k in v for k in keywords):
            return bucket
    return None


def _find_column(columns, candidates):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        for col_lower, col in cols_lower.items():
            if cand in col_lower:
                return col
    return None


def _fallback_calibration(reason):
    calibration = {"source": "fallback_defaults", "warnings": [reason]}
    with open(KAGGLE_CALIBRATION_CACHE, "w") as f:
        json.dump(calibration, f, indent=2)
    return calibration


def load_kaggle_calibration():
    if os.path.exists(KAGGLE_CALIBRATION_CACHE):
        with open(KAGGLE_CALIBRATION_CACHE) as f:
            return json.load(f)

    if not os.path.exists(KAGGLE_CSV_PATH):
        print(f"  No local CSV at '{KAGGLE_CSV_PATH}' â€” demand_nodes.json and "
              "cost_params.json will use the hardcoded defaults, not fitted "
              "values. Download the Kaggle 'Delivery Logistics Dataset (India "
              "â€“ Multi-Partner)' CSV and set KAGGLE_CSV_PATH to calibrate from "
              "real data.")
        return _fallback_calibration("no CSV found at KAGGLE_CSV_PATH")

    print(f"  Reading '{KAGGLE_CSV_PATH}' for demand/cost calibration ...")
    import csv
    try:
        with open(KAGGLE_CSV_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"  Could not read '{KAGGLE_CSV_PATH}' ({e}) â€” using fallback defaults")
        return _fallback_calibration(f"read error: {e}")

    if not rows:
        print(f"  '{KAGGLE_CSV_PATH}' is empty â€” using fallback defaults")
        return _fallback_calibration("CSV had 0 rows")

    columns = rows[0].keys()
    weight_col = _find_column(columns, ["weight"])
    vehicle_col = _find_column(columns, ["vehicle"])
    cost_col = _find_column(columns, ["cost", "price"])
    distance_col = _find_column(columns, ["distance"])
    missing = [n for n, c in [("weight", weight_col), ("vehicle", vehicle_col),
                               ("cost", cost_col), ("distance", distance_col)] if c is None]
    if missing:
        print(f"  Could not locate column(s) {missing} in '{KAGGLE_CSV_PATH}' "
              f"(found columns: {list(columns)}) â€” using fallback defaults")
        return _fallback_calibration(
            f"missing columns {missing}; found {list(columns)}")

    all_weights, per_vehicle = [], {b: {"weights": [], "costs_per_km": []}
                                     for b in _VEHICLE_KEYWORDS}
    unmatched_vehicle, malformed = 0, 0
    for row in rows:
        try:
            w = float(row[weight_col])
            dist = float(row[distance_col])
            cost = float(row[cost_col])
        except (TypeError, ValueError):
            malformed += 1
            continue
        if dist <= 0 or w <= 0:
            malformed += 1
            continue
        all_weights.append(w)

        bucket = _match_vehicle_bucket(row.get(vehicle_col))
        if bucket is None:
            unmatched_vehicle += 1
            continue
        per_vehicle[bucket]["weights"].append(w)
        per_vehicle[bucket]["costs_per_km"].append(cost / dist)

    total = len(rows)
    usable = len(all_weights)
    if usable == 0:
        print(f"  0/{total} rows had usable weight/distance/cost values â€” "
              "using fallback defaults")
        return _fallback_calibration("0 usable rows after parsing weight/distance/cost")

    ln_weights = np.log(np.clip(all_weights, 0.01, None))
    weight_fit = {
        "mu": round(float(np.mean(ln_weights)), 3),
        "sigma": round(float(np.std(ln_weights)), 3),
        "min": round(float(min(all_weights)), 2),
        "max": round(float(max(all_weights)), 2),
        "n_rows": usable,
    }

    vehicle_stats, warnings = {}, []
    matched_vehicle = sum(len(d["weights"]) for d in per_vehicle.values())
    for bucket, data in per_vehicle.items():
        n = len(data["weights"])
        if n == 0:
            warnings.append(f"no rows matched vehicle bucket '{bucket}' â€” "
                             "its cost_per_km_inr keeps the hardcoded default")
            continue
        vehicle_stats[bucket] = {
            "n_rows": n,
            "cost_per_km_inr": round(sum(data["costs_per_km"]) / n, 2),
            "mode_share": round(n / matched_vehicle, 3) if matched_vehicle else None,
        }

    calibration = {
        "source": "kaggle_delivery_logistics_india",
        "csv_path": KAGGLE_CSV_PATH,
        "total_rows": total,
        "usable_rows": usable,
        "unmatched_vehicle_rows": unmatched_vehicle,
        "malformed_rows": malformed,
        "weight_kg_fit": weight_fit,
        "vehicle_stats": vehicle_stats,
        "warnings": warnings,
        "caveat": ("'quantity' (units per stop) has no equivalent column in "
                   "this dataset and is NOT fitted â€” it stays the assumed "
                   "Poisson(2.5) default in DEMAND_DISTRIBUTIONS regardless "
                   "of whether a CSV is found."),
    }

    with open(KAGGLE_CALIBRATION_CACHE, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"  Weight fit: lognormal(mu={weight_fit['mu']}, sigma={weight_fit['sigma']}) "
          f"from {usable}/{total} usable rows")
    for bucket, stats in vehicle_stats.items():
        print(f"    {bucket}: {stats['n_rows']} rows, â‚¹{stats['cost_per_km_inr']}/km, "
              f"{stats['mode_share']*100:.0f}% mode share")
    for w in warnings:
        print(f"    WARNING: {w}")

    return calibration


def build_demand_distributions(calibration):
    merged = json.loads(json.dumps(DEMAND_DISTRIBUTIONS))
    fit = calibration.get("weight_kg_fit")
    if fit:
        merged["weight_kg"] = {
            "distribution": "lognormal", "mu": fit["mu"], "sigma": fit["sigma"],
            "min": fit["min"], "max": fit["max"],
            "fitted_from_n_rows": fit["n_rows"],
        }
    return merged


# =======================================================================
# 8b. COST PARAMETERS / DEMAND / FLEET CALIBRATION TABLE
# =======================================================================
def generate_cost_params(city_dir, calibration=None, fleet_calibration=None):
    """
    Builds cost_params.json â€” the QPSO-facing fleet/cost/demand table.

    `calibration`: from load_kaggle_calibration() â€” fits cost_per_km_inr
    per vehicle type and weight_kg overall, when a local CSV is available.

    `fleet_calibration`: from calibrate_fleet_from_tripinfo() â€” fits
    speed_factor per vehicle type from THIS CITY's own SUMO simulation
    (new in v6). Independent of the Kaggle fit â€” one is cross-city cost
    data, the other is per-city simulated speed data; both can apply to
    the same fleet entry without conflict (they override different
    fields: cost_per_km_inr vs speed_factor).
    """
    output_path = os.path.join(city_dir, "cost_params.json")

    def generate():
        print("  Generating cost calibration parameters ...")
        cal = calibration if calibration is not None else load_kaggle_calibration()
        vehicle_stats = cal.get("vehicle_stats", {})
        demand_distributions = build_demand_distributions(cal)

        fcal = fleet_calibration or {}
        fcal_vehicles = fcal.get("vehicles", {})
        speed_factor_by_fleet_type = {
            v["fleet_type"]: v["speed_factor_fitted"] for v in fcal_vehicles.values()
        }

        fleet = [dict(v) for v in FLEET_DEFINITIONS]
        calibrated_types = []        # cost_per_km fitted (Kaggle)
        speed_calibrated_types = []  # speed_factor fitted (SUMO, v6)
        for v in fleet:
            stats = vehicle_stats.get(v["type"])
            if stats:
                v["cost_per_km_inr"] = stats["cost_per_km_inr"]
                calibrated_types.append(v["type"])

            sf = speed_factor_by_fleet_type.get(v["type"])
            if sf is not None:
                v["speed_factor"] = sf
                speed_calibrated_types.append(v["type"])

        mode_shares = {t: s["mode_share"] for t, s in vehicle_stats.items()
                        if s.get("mode_share") is not None}

        if cal["source"] == "kaggle_delivery_logistics_india":
            demand_note = (
                f"weight_kg FITTED (lognormal mu={demand_distributions['weight_kg']['mu']}, "
                f"sigma={demand_distributions['weight_kg']['sigma']}) from "
                f"{cal['usable_rows']}/{cal['total_rows']} rows of the local Kaggle "
                f"'Delivery Logistics Dataset (India â€“ Multi-Partner)' CSV "
                f"({cal['csv_path']}). quantity is NOT fitted (no equivalent "
                "column in that dataset) and stays the assumed Poisson(2.5) default."
            )
            cost_note = (
                f"cost_per_km_inr FITTED per vehicle type from the same CSV "
                f"(cost / distance, averaged per matched bucket) for: "
                f"{calibrated_types or 'none â€” 0 rows matched any vehicle bucket'}. "
                "Any vehicle type not listed there keeps its hardcoded default."
            )
        else:
            demand_note = (
                "ASSUMED, not fitted â€” no local Kaggle CSV found "
                f"(checked '{KAGGLE_CSV_PATH}'). Weight: lognormal(1.0, 0.8) "
                "clipped [0.5, 50] kg. Quantity: Poisson(2.5) clipped [1, 15] "
                "units. Set KAGGLE_CSV_PATH to a downloaded copy of the Kaggle "
                "'Delivery Logistics Dataset (India â€“ Multi-Partner)' CSV to "
                "replace these with fitted values."
            )
            cost_note = ("ASSUMED Indian logistics industry averages, not "
                          "fitted â€” Two-wheeler: â‚¹4/km, Three-wheeler: â‚¹8/km, LCV: â‚¹14/km.")

        if speed_calibrated_types:
            speed_note = (
                f"speed_factor FITTED per vehicle type (v6, NEW) from this city's "
                f"own SUMO simulated trip data (tripinfo output) for: "
                f"{speed_calibrated_types}. Computed as observed average simulated "
                "speed / that vehicle's free-flow max speed, pooled across all time "
                f"windows ({fcal.get('total_trips_parsed', 0)} trips parsed). Any "
                "vehicle type not listed there keeps its hardcoded default."
            )
        else:
            speed_note = ("ASSUMED â€” no usable SUMO tripinfo trips were found to "
                           "fit speed_factor from for any vehicle type.")

        output = {
            "fleet": fleet,
            "demand_distributions": demand_distributions,
            "cost_weights": COST_WEIGHTS,
            "cost_per_km_by_vehicle": {v["type"]: v["cost_per_km_inr"] for v in fleet},
            "fixed_cost_by_vehicle": {v["type"]: v["fixed_cost_inr"] for v in fleet},
            "capacity_kg_by_vehicle": {v["type"]: v["capacity_kg"] for v in fleet},
            "capacity_units_by_vehicle": {v["type"]: v["capacity_units"] for v in fleet},
            "speed_factor_by_vehicle": {v["type"]: v["speed_factor"] for v in fleet},
            "vehicle_mode_share": mode_shares or None,
            "calibration_source": cal["source"],
            "fleet_calibration_source": fcal.get("source", "none â€” SUMO tripinfo unavailable"),
            "notes": {
                "demand_source": demand_note,
                "cost_source": cost_note,
                "speed_source": speed_note,
                "fleet_source": (
                    "Vehicle categories and any non-calibrated fields "
                    "(capacity_kg, capacity_units, fixed_cost_inr, "
                    "max_range_km) are assumed â€” neither the Kaggle CSV nor "
                    "SUMO trip data has a column/signal for a vehicle's own "
                    "payload capacity, fixed costs, or range. speed_factor "
                    "may be SUMO-fitted (see speed_source above); "
                    "cost_per_km_inr may be Kaggle-fitted (see cost_source)."
                ),
            },
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"  Saved cost parameters to {os.path.basename(output_path)} "
              f"(cost source: {cal['source']}, speed source: "
              f"{fcal.get('source', 'none')})")
        print(f"  Fleet: " + ", ".join(
            f"{v['type']} ({v['capacity_kg']}kg, â‚¹{v['cost_per_km_inr']}/km"
            + (", cost-fitted" if v["type"] in calibrated_types else ", cost-assumed")
            + (", speed-fitted)" if v["type"] in speed_calibrated_types else ", speed-assumed)")
            for v in fleet))

    run_if_missing(output_path, generate)
    return output_path


# =======================================================================
# 9. SPATIAL MATCH: SUMO EDGES -> OSMnx EDGES
# =======================================================================
def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def get_osmnx_edge_midpoints(G):
    midpoints = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        if "geometry" in data:
            mid = data["geometry"].interpolate(0.5, normalized=True)
            lon, lat = mid.x, mid.y
        else:
            lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
            lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
        midpoints[(u, v, k)] = (lon, lat)
    return midpoints


def get_sumo_edge_midpoints(net_path):
    net = sumolib.net.readNet(net_path)
    midpoints = {}
    for edge in net.getEdges():
        if edge.getID().startswith(":"):
            continue
        shape = edge.getShape()
        mid_x = sum(p[0] for p in shape) / len(shape)
        mid_y = sum(p[1] for p in shape) / len(shape)
        lon, lat = net.convertXY2LonLat(mid_x, mid_y)
        midpoints[edge.getID()] = (lon, lat)
    return midpoints


def match_sumo_to_osmnx(osmnx_midpoints, sumo_midpoints):
    osmnx_keys = list(osmnx_midpoints.keys())
    osmnx_coords = [osmnx_midpoints[k] for k in osmnx_keys]
    tree = cKDTree(osmnx_coords)

    matches = {}
    for sumo_id, (lon, lat) in sumo_midpoints.items():
        dist, idx = tree.query((lon, lat))
        candidate_key = osmnx_keys[idx]
        c_lon, c_lat = osmnx_midpoints[candidate_key]
        if haversine_m(lon, lat, c_lon, c_lat) <= MATCH_DISTANCE_M:
            matches[sumo_id] = candidate_key
    print(f"  Matched {len(matches)}/{len(sumo_midpoints)} SUMO edges to OSMnx edges")
    return matches


def parse_window_edge_data(edgedata_path, warmup_s):
    buckets = {}
    tree = ET.parse(edgedata_path)
    for interval in tree.getroot().findall("interval"):
        t_start = float(interval.get("begin")) - warmup_s
        edge_times = {}
        for edge in interval.findall("edge"):
            tt = edge.get("traveltime")
            if tt is not None:
                edge_times[edge.get("id")] = float(tt)
        buckets[t_start] = edge_times
    return buckets


# =======================================================================
# 10. MAIN
# =======================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Integrated SIH Data & Traffic Pipeline")
    parser.add_argument("--city", type=str, default=None, help="Target city name")
    parser.add_argument("--radius", type=float, default=3000, help="Search radius in meters")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip visualization generation")
    args, unknown = parser.parse_known_args()

    city_name = args.city
    if not city_name:
        # Check if stdin is interactive
        if sys.stdin.isatty():
            try:
                city_input = input("Enter a city/place name (default 'Kharagpur, West Bengal, India'): ").strip()
                city_name = city_input if city_input else "Kharagpur, West Bengal, India"
            except (EOFError, KeyboardInterrupt):
                city_name = "Kharagpur, West Bengal, India"
        else:
            city_name = "Kharagpur, West Bengal, India"

    print(f"Executing integrated traffic and data pipeline for '{city_name}'...")

    # Execute modular pipeline
    try:
        from src.data_pipeline.run_pipeline import run_full_pipeline
        run_full_pipeline(city_name=city_name, radius=args.radius, generate_visuals=not args.skip_visuals)
    except Exception as e:
        print(f"Error running pipeline orchestrator: {e}")
        import traceback
        traceback.print_exc()
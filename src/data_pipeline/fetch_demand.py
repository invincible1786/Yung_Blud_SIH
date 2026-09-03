import os
import csv
import json
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Dict, Any, List, Optional, Tuple

CUSTOMER_POI_TAGS = {
    "shop": True,
    "amenity": {
        "marketplace", "restaurant", "cafe", "fast_food", "pharmacy",
        "hospital", "clinic", "bank", "fuel", "college", "school",
        "university", "library", "community_centre",
    },
    "office": True,
}

DEMAND_DISTRIBUTIONS = {
    "weight_kg": {"distribution": "lognormal", "mu": 1.0, "sigma": 0.8, "min": 0.5, "max": 50.0},
    "quantity": {"distribution": "poisson", "lam": 2.5, "min": 1, "max": 15},
    "time_window_width_min": {
        "distribution": "choice",
        "values": [30, 60, 90, 120, 180, 240],
        "weights": [0.05, 0.15, 0.25, 0.30, 0.15, 0.10]
    },
}

_VEHICLE_KEYWORDS = {
    "two_wheeler":   ["bike", "motorcycle", "scooter", "two wheeler", "two-wheeler", "ev bike"],
    "three_wheeler": ["auto", "three wheeler", "three-wheeler", "rickshaw", "tempo"],
    "lcv":           ["van", "truck", "mini truck", "lcv", "four wheeler", "four-wheeler", "car", "ev van"],
}

DEMAND_SEED = 42
MIN_CUSTOMER_NODES = 120
MAX_CUSTOMER_NODES = 250


def _tag_matches(tag_dict: Dict[str, str], tag_spec: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    for spec_key, spec_values in tag_spec.items():
        if spec_key in tag_dict:
            actual_value = tag_dict[spec_key]
            if spec_values is True or actual_value in spec_values:
                return spec_key, actual_value
    return None


def parse_pois_from_osm_xml(osm_xml_path: str, tag_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parses POI nodes from an OSM XML file with streaming iterparse.
    """
    if not os.path.exists(osm_xml_path):
        return []

    pois = []
    try:
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
    except Exception as e:
        print(f"  Warning parsing OSM XML: {e}")

    return pois


def _match_vehicle_bucket(raw_value: Any) -> Optional[str]:
    v = str(raw_value).lower()
    for bucket, keywords in _VEHICLE_KEYWORDS.items():
        if any(k in v for k in keywords):
            return bucket
    return None


def _find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        for col_lower, col in cols_lower.items():
            if cand in col_lower:
                return col
    return None


def load_kaggle_calibration(raw_dir: str = os.path.join("data", "raw"),
                             processed_dir: str = os.path.join("data", "processed")) -> Dict[str, Any]:
    """
    Calibrates demand weight distribution (lognormal mu, sigma) and vehicle cost_per_km
    from the Kaggle delivery dataset.
    """
    cache_path = os.path.join(processed_dir, "kaggle_calibration.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    csv_path = os.path.join(raw_dir, "kaggle_delivery.csv")
    root_csv = "Delivery_Logistics.csv"
    if os.path.exists(root_csv) and not os.path.exists(csv_path):
        shutil.move(root_csv, csv_path)

    if not os.path.exists(csv_path):
        print(f"  Kaggle delivery CSV not found at '{csv_path}'. Using robust defaults.")
        default_cal = {
            "source": "fallback_defaults",
            "weight_kg_fit": {"mu": 1.0, "sigma": 0.8, "min": 0.5, "max": 50.0, "n_rows": 0},
            "vehicle_stats": {
                "two_wheeler": {"cost_per_km_inr": 4.0, "mode_share": 0.45},
                "three_wheeler": {"cost_per_km_inr": 8.0, "mode_share": 0.35},
                "lcv": {"cost_per_km_inr": 14.0, "mode_share": 0.20}
            }
        }
        with open(cache_path, "w") as f:
            json.dump(default_cal, f, indent=2)
        return default_cal

    print(f"  Fitting demand and fleet economics from '{csv_path}'...")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"'{csv_path}' is empty!")

    columns = list(rows[0].keys())
    weight_col = _find_column(columns, ["package_weight", "weight"])
    vehicle_col = _find_column(columns, ["vehicle_type", "vehicle"])
    cost_col = _find_column(columns, ["delivery_cost", "cost", "price"])
    distance_col = _find_column(columns, ["distance_km", "distance"])

    all_weights = []
    per_vehicle = {b: {"weights": [], "costs_per_km": []} for b in _VEHICLE_KEYWORDS}

    for row in rows:
        try:
            w = float(row[weight_col])
            dist = float(row[distance_col])
            cost = float(row[cost_col])
        except (TypeError, ValueError, KeyError):
            continue

        if dist <= 0 or w <= 0:
            continue

        all_weights.append(w)
        bucket = _match_vehicle_bucket(row.get(vehicle_col))
        if bucket is not None:
            per_vehicle[bucket]["weights"].append(w)
            per_vehicle[bucket]["costs_per_km"].append(cost / dist)

    ln_weights = np.log(np.clip(all_weights, 0.01, None))
    weight_fit = {
        "mu": round(float(np.mean(ln_weights)), 3),
        "sigma": round(float(np.std(ln_weights)), 3),
        "min": round(float(min(all_weights)), 2),
        "max": round(float(max(all_weights)), 2),
        "n_rows": len(all_weights),
    }

    vehicle_stats = {}
    matched_total = sum(len(d["weights"]) for d in per_vehicle.values())
    for bucket, data in per_vehicle.items():
        n = len(data["weights"])
        if n > 0:
            vehicle_stats[bucket] = {
                "n_rows": n,
                "cost_per_km_inr": round(float(sum(data["costs_per_km"]) / n), 2),
                "mode_share": round(float(n / matched_total), 3) if matched_total else None,
            }
        else:
            default_costs = {"two_wheeler": 4.0, "three_wheeler": 8.0, "lcv": 14.0}
            vehicle_stats[bucket] = {
                "n_rows": 0,
                "cost_per_km_inr": default_costs.get(bucket, 10.0),
                "mode_share": 0.33,
            }

    calibration = {
        "source": "kaggle_delivery_logistics_india",
        "csv_path": csv_path,
        "total_rows": len(rows),
        "usable_rows": len(all_weights),
        "weight_kg_fit": weight_fit,
        "vehicle_stats": vehicle_stats,
    }

    with open(cache_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"  Calibrated weight distribution: lognormal(mu={weight_fit['mu']}, sigma={weight_fit['sigma']})")
    for b, s in vehicle_stats.items():
        print(f"    {b}: INR {s['cost_per_km_inr']}/km (mode share: {s.get('mode_share', 0)*100:.1f}%)")

    return calibration


def build_demand_distributions(calibration: Dict[str, Any]) -> Dict[str, Any]:
    merged = json.loads(json.dumps(DEMAND_DISTRIBUTIONS))
    fit = calibration.get("weight_kg_fit")
    if fit:
        merged["weight_kg"] = {
            "distribution": "lognormal",
            "mu": fit["mu"],
            "sigma": fit["sigma"],
            "min": fit["min"],
            "max": fit["max"],
            "fitted_from_n_rows": fit.get("n_rows", 0),
        }
    return merged


def _sample_demand(rng: np.random.Generator, n: int, dd: Dict[str, Any]):
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


def _assign_time_windows(rng: np.random.Generator, n: int, tw_widths: np.ndarray):
    earliest = 360   # 06:00 AM (minutes from midnight)
    latest = 1320    # 10:00 PM (minutes from midnight)
    windows = []
    for i in range(n):
        w = int(tw_widths[i])
        max_start = max(latest - w, earliest)
        start = int(rng.integers(earliest, max_start + 1))
        windows.append([start, start + w])
    return windows


def _snap_batch_to_coords(coords_list: List[Tuple[float, float]], lons: List[float], lats: List[float]) -> Tuple[List[int], List[float]]:
    """
    Snaps query (lon, lat) points to nearest node indices via KDTree.
    """
    tree = cKDTree(coords_list)
    query_coords = list(zip(lons, lats))
    dists, indices = tree.query(query_coords)
    return indices.tolist(), dists.tolist()


def process_demand(city_name: str = "Kharagpur, West Bengal, India",
                   raw_dir: str = os.path.join("data", "raw"),
                   processed_dir: str = os.path.join("data", "processed"),
                   osm_xml_path: Optional[str] = None):
    """
    Full demand processing pipeline:
    1. Calibrates Kaggle demand distributions and fleet cost per km.
    2. Extracts POI customer nodes from OSM XML with fallback to graph nodes.
    3. Samples realistic demands, time windows, and service times.
    4. Outputs demand_nodes.json and backward-compatible demand_vector.csv.
    """
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load node map & metadata
    node_map_path = os.path.join(processed_dir, "node_id_map.json")
    nodes_metadata_path = os.path.join(processed_dir, "nodes_metadata.json")
    if not os.path.exists(node_map_path) or not os.path.exists(nodes_metadata_path):
        raise FileNotFoundError("Run fetch_osm.py first to generate node_id_map.json and nodes_metadata.json!")

    with open(node_map_path, "r") as f:
        node_id_map = json.load(f)
    with open(nodes_metadata_path, "r") as f:
        nodes_metadata = json.load(f)

    # 2. Calibrate Kaggle dataset
    calibration = load_kaggle_calibration(raw_dir, processed_dir)
    dd = build_demand_distributions(calibration)

    # 3. Locate OSM XML
    if osm_xml_path is None or not os.path.exists(osm_xml_path):
        candidates = [
            os.path.join(raw_dir, "osm", "kharagpur_west_bengal_india.osm.xml"),
            os.path.join(raw_dir, "osm", "kharagpur__west_bengal__india.osm.xml"),
        ] + [p for p in os.listdir(os.path.join(raw_dir, "osm")) if p.endswith(".osm.xml")]
        for cand in candidates:
            p = cand if os.path.isabs(cand) else os.path.join(raw_dir, "osm", cand)
            if os.path.exists(p):
                osm_xml_path = p
                break

    pois = []
    if osm_xml_path and os.path.exists(osm_xml_path):
        print(f"  Parsing POIs from OSM XML: '{osm_xml_path}'...")
        pois = parse_pois_from_osm_xml(osm_xml_path, CUSTOMER_POI_TAGS)
        print(f"  Extracted {len(pois)} customer POIs from OSM.")

    rng = np.random.default_rng(DEMAND_SEED)

    # Deduplicate POIs
    seen = set()
    unique_pois = []
    for p in pois:
        coord_key = (round(p["lon"], 5), round(p["lat"], 5))
        if coord_key not in seen:
            seen.add(coord_key)
            unique_pois.append(p)
    pois = unique_pois

    # Build coordinate list from SCC nodes
    scc_node_ids = [int(nid) for nid in node_id_map.keys()]
    scc_coords = [(nodes_metadata[str(nid)]["lon"], nodes_metadata[str(nid)]["lat"]) for nid in scc_node_ids]

    # Supplement with random graph nodes if needed
    if len(pois) < MIN_CUSTOMER_NODES:
        deficit = max(MIN_CUSTOMER_NODES - len(pois), 30)
        print(f"  Supplementing with {deficit} graph nodes as customer stops...")
        sampled_indices = rng.choice(len(scc_node_ids), size=min(deficit, len(scc_node_ids)), replace=False)
        for idx in sampled_indices:
            nid = scc_node_ids[idx]
            m = nodes_metadata[str(nid)]
            pois.append({
                "lon": m["lon"],
                "lat": m["lat"],
                "name": f"Customer_Stop_{nid}",
                "category": "amenity:commercial_stop",
                "osm_id": str(nid)
            })

    # Cap if too many
    if len(pois) > MAX_CUSTOMER_NODES:
        sample_idx = rng.choice(len(pois), size=MAX_CUSTOMER_NODES, replace=False)
        pois = [pois[i] for i in sample_idx]

    n_cust = len(pois)
    weights_kg, quantities, tw_widths = _sample_demand(rng, n_cust, dd)
    time_windows = _assign_time_windows(rng, n_cust, tw_widths)
    service_times = np.clip(rng.normal(5, 2, n_cust), 2, 15).astype(int)

    # Snap POIs to SCC road nodes
    p_lons = [p["lon"] for p in pois]
    p_lats = [p["lat"] for p in pois]
    snapped_indices, snap_dists = _snap_batch_to_coords(scc_coords, p_lons, p_lats)

    customer_nodes = []
    vector_rows = []

    for i in range(n_cust):
        snapped_node_id = scc_node_ids[snapped_indices[i]]
        poi = pois[i]
        c_record = {
            "id": i,
            "name": poi["name"] or f"Customer_{i}",
            "category": poi["category"],
            "osm_id": poi.get("osm_id"),
            "lon": round(poi["lon"], 6),
            "lat": round(poi["lat"], 6),
            "graph_node": snapped_node_id,
            "snap_distance_deg": round(snap_dists[i], 6),
            "demand_kg": float(weights_kg[i]),
            "demand_units": int(quantities[i]),
            "time_window": time_windows[i],
            "service_time_min": int(service_times[i]),
        }
        customer_nodes.append(c_record)

        # Vector row for backward compatibility
        vector_rows.append({
            "node_id": snapped_node_id,
            "demand": int(np.ceil(weights_kg[i])),
            "demand_units": int(quantities[i]),
            "time_window_start": time_windows[i][0],
            "time_window_end": time_windows[i][1],
            "service_time_min": int(service_times[i]),
            "vehicle_type_hint": "two_wheeler" if weights_kg[i] <= 15 else ("three_wheeler" if weights_kg[i] <= 200 else "lcv"),
            "poi_category": poi["category"],
            "customer_name": poi["name"] or f"Stop_{snapped_node_id}",
        })

    # Save demand_nodes.json
    demand_nodes_path = os.path.join(processed_dir, "demand_nodes.json")
    with open(demand_nodes_path, "w") as f:
        json.dump({
            "city": city_name,
            "total_customers": n_cust,
            "distributions_used": dd,
            "customers": customer_nodes
        }, f, indent=2)
    print(f"  Saved {n_cust} rich customer nodes to '{demand_nodes_path}'.")

    # Save backward-compatible demand_vector.csv
    demand_vector_path = os.path.join(processed_dir, "demand_vector.csv")
    pd.DataFrame(vector_rows).to_csv(demand_vector_path, index=False)
    print(f"  Saved updated demand vector to '{demand_vector_path}'.")

    return customer_nodes


if __name__ == "__main__":
    process_demand()

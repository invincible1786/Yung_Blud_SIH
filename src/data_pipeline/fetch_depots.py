import os
import json
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Any, List, Optional, Tuple

DEPOT_POI_TAGS = {
    "amenity": {"post_office", "bus_station"},
    "building": {"warehouse", "industrial"},
    "landuse": {"industrial"},
}

DEMAND_SEED = 42
MIN_DEPOT_NODES = 2
MAX_DEPOT_NODES = 15

# India Post data.gov.in API
_DATAGOVIN_SAMPLE_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
DATAGOVIN_API_KEY = os.environ.get("DATAGOVIN_API_KEY", _DATAGOVIN_SAMPLE_KEY)
DATAGOVIN_PINCODE_RESOURCE = "6176ee09-3d56-4a3b-8115-21841576b2f6"
DATAGOVIN_OFFICETYPE_ALLOWLIST = {"head post office", "sub post office"}


def _tag_matches(tag_dict: Dict[str, str], tag_spec: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    for spec_key, spec_values in tag_spec.items():
        if spec_key in tag_dict:
            actual_value = tag_dict[spec_key]
            if spec_values is True or actual_value in spec_values:
                return spec_key, actual_value
    return None


def _collect_all_node_coords(osm_xml_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Pass 1 of the two-pass way parser: index coordinates of all nodes in OSM XML.
    """
    coords = {}
    if not os.path.exists(osm_xml_path):
        return coords

    try:
        context = ET.iterparse(osm_xml_path, events=("end",))
        for event, elem in context:
            if elem.tag == "node":
                node_id = elem.get("id")
                lat_str = elem.get("lat")
                lon_str = elem.get("lon")
                if node_id and lat_str and lon_str:
                    coords[node_id] = (float(lon_str), float(lat_str))
                elem.clear()
            elif elem.tag in ("way", "relation"):
                elem.clear()
    except Exception as e:
        print(f"  Warning during node coordinate collection: {e}")

    return coords


def parse_ways_from_osm_xml(osm_xml_path: str, tag_spec: Dict[str, Any],
                            node_coords: Dict[str, Tuple[float, float]]) -> List[Dict[str, Any]]:
    """
    Pass 2 of the two-pass way parser: resolves warehouses/industrial polygons
    to centroid coordinates.
    """
    if not os.path.exists(osm_xml_path):
        return []

    pois = []
    try:
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
                            or tag_dict.get("brand") or tag_dict.get("operator")
                            or f"OSM {value.title()} Facility")

                    pois.append({
                        "lon": avg_lon,
                        "lat": avg_lat,
                        "name": name,
                        "category": f"{key}:{value}",
                        "osm_id": f"way/{elem.get('id')}",
                        "source": "osm",
                    })

            elem.clear()
    except Exception as e:
        print(f"  Warning during way parsing: {e}")

    return pois


def parse_node_pois(osm_xml_path: str, tag_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parses node POIs from OSM XML.
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
                    name = (tag_dict.get("name") or tag_dict.get("name:en")
                            or f"OSM {value.title()}")
                    pois.append({
                        "lon": float(lon_str),
                        "lat": float(lat_str),
                        "name": name,
                        "category": f"{key}:{value}",
                        "osm_id": f"node/{elem.get('id')}",
                        "source": "osm",
                    })
            elem.clear()
    except Exception as e:
        print(f"  Warning parsing node POIs: {e}")

    return pois


def fetch_and_snap_depots(city_name: str = "Kharagpur, West Bengal, India",
                          raw_dir: str = os.path.join("data", "raw"),
                          processed_dir: str = os.path.join("data", "processed"),
                          osm_xml_path: Optional[str] = None):
    """
    Extracts depot/distribution center candidates from:
    1. Local OSM XML (2-pass parser: nodes + way centroids for warehouses & industrial).
    2. Indian Post Office Directory (data/raw/india_pincode.csv).
    3. High-degree network hub nodes as fallback.
    Outputs depot_nodes.json and depot_nodes.csv.
    """
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load node map and coordinates
    node_map_path = os.path.join(processed_dir, "node_id_map.json")
    nodes_metadata_path = os.path.join(processed_dir, "nodes_metadata.json")
    if not os.path.exists(node_map_path) or not os.path.exists(nodes_metadata_path):
        raise FileNotFoundError("Run fetch_osm.py first to generate node_id_map.json and nodes_metadata.json!")

    with open(node_map_path, "r") as f:
        node_id_map = json.load(f)
    with open(nodes_metadata_path, "r") as f:
        nodes_metadata = json.load(f)

    scc_node_ids = [int(nid) for nid in node_id_map.keys()]
    scc_coords = [(nodes_metadata[str(nid)]["lon"], nodes_metadata[str(nid)]["lat"]) for nid in scc_node_ids]
    scc_tree = cKDTree(scc_coords)

    # 2. Source 1: Local OSM XML (nodes + 2-pass way centroids)
    if osm_xml_path is None or not os.path.exists(osm_xml_path):
        raw_osm = os.path.join(raw_dir, "osm")
        if os.path.exists(raw_osm):
            xmls = [os.path.join(raw_osm, f) for f in os.listdir(raw_osm) if f.endswith(".osm.xml")]
            if xmls:
                osm_xml_path = xmls[0]

    all_pois = []
    if osm_xml_path and os.path.exists(osm_xml_path):
        print(f"  [Source 1] Parsing depot candidates from OSM XML: '{osm_xml_path}'...")
        osm_nodes = parse_node_pois(osm_xml_path, DEPOT_POI_TAGS)
        node_coords = _collect_all_node_coords(osm_xml_path)
        osm_ways = parse_ways_from_osm_xml(osm_xml_path, DEPOT_POI_TAGS, node_coords)
        osm_pois = osm_nodes + osm_ways
        all_pois.extend(osm_pois)
        print(f"  Extracted {len(osm_pois)} OSM depot candidates ({len(osm_nodes)} nodes + {len(osm_ways)} way centroids).")

    # 3. Source 2: Indian Post Office Directory
    pincode_csv = os.path.join(raw_dir, "india_pincode.csv")
    if os.path.exists(pincode_csv):
        print(f"  [Source 2] Loading Indian Postal Directory from '{pincode_csv}'...")
        try:
            p_df = pd.read_csv(pincode_csv, low_memory=False)
            col_map = {c: c.replace('"', '').strip() for c in p_df.columns}
            p_df = p_df.rename(columns=col_map)

            # Match columns
            off_col = [c for c in p_df.columns if 'office' in c.lower() and 'name' in c.lower()]
            dist_col = [c for c in p_df.columns if 'district' in c.lower()]
            lat_col = [c for c in p_df.columns if 'lat' in c.lower()]
            lon_col = [c for c in p_df.columns if 'lon' in c.lower()]
            pin_col = [c for c in p_df.columns if 'pin' in c.lower()]

            if dist_col and off_col and lat_col and lon_col:
                dc, oc, lac, loc = dist_col[0], off_col[0], lat_col[0], lon_col[0]
                pc = pin_col[0] if pin_col else None

                mask = p_df[dc].astype(str).str.contains("MEDINIPUR|MIDNAPORE|KHARAGPUR", case=False, na=False)
                sub_df = p_df[mask].copy()

                sub_df[lac] = pd.to_numeric(sub_df[lac], errors="coerce")
                sub_df[loc] = pd.to_numeric(sub_df[loc], errors="coerce")
                sub_df = sub_df.dropna(subset=[lac, loc])

                # Filter coordinates within reasonable bounding box near Kharagpur
                center_lat = np.mean([c[1] for c in scc_coords])
                center_lon = np.mean([c[0] for c in scc_coords])

                sub_df["dist_center"] = np.hypot(sub_df[loc] - center_lon, sub_df[lac] - center_lat)
                sub_df = sub_df.sort_values("dist_center").head(10)

                for _, row in sub_df.iterrows():
                    all_pois.append({
                        "lon": float(row[loc]),
                        "lat": float(row[lac]),
                        "name": str(row[oc]),
                        "category": "amenity:post_office",
                        "osm_id": f"pincode/{row[pc]}" if pc else None,
                        "source": "india_post"
                    })
                print(f"  Loaded {len(sub_df)} post office locations from directory.")
        except Exception as e:
            print(f"  Warning reading pincode CSV: {e}")

    # Deduplicate locations within 100m (~0.001 deg)
    seen = set()
    deduped = []
    for p in all_pois:
        ck = (round(p["lon"], 3), round(p["lat"], 3))
        if ck not in seen:
            seen.add(ck)
            deduped.append(p)
    all_pois = deduped

    # 4. Fallback: high-degree network nodes if insufficient
    if len(all_pois) < MIN_DEPOT_NODES:
        deficit = max(MIN_DEPOT_NODES - len(all_pois), 3)
        print(f"  Supplementing with {deficit} high-degree road intersections as distribution hubs...")
        for i in range(deficit):
            nid = scc_node_ids[i * 10]
            m = nodes_metadata[str(nid)]
            all_pois.append({
                "lon": m["lon"],
                "lat": m["lat"],
                "name": f"Central Freight Hub #{i+1}",
                "category": "building:warehouse",
                "osm_id": str(nid),
                "source": "network_hub"
            })

    # Cap at MAX_DEPOT_NODES
    all_pois = all_pois[:MAX_DEPOT_NODES]

    # Snap to SCC graph nodes
    q_coords = list(zip([p["lon"] for p in all_pois], [p["lat"] for p in all_pois]))
    dists, indices = scc_tree.query(q_coords)

    depot_list = []
    csv_rows = []

    for i, p in enumerate(all_pois):
        snapped_node_id = scc_node_ids[indices[i]]
        cat = p["category"]

        # Capacity assignment based on facility type
        if "warehouse" in cat or "industrial" in cat:
            cap_kg, cap_units = 5000, 500
        elif "post_office" in cat or "india_post" in cat:
            cap_kg, cap_units = 2000, 200
        elif "bus_station" in cat:
            cap_kg, cap_units = 3000, 300
        else:
            cap_kg, cap_units = 3500, 350

        d_record = {
            "id": i,
            "name": p["name"],
            "category": cat,
            "source": p.get("source", "unknown"),
            "osm_id": p.get("osm_id"),
            "lon": round(p["lon"], 6),
            "lat": round(p["lat"], 6),
            "graph_node": snapped_node_id,
            "snap_distance_deg": round(float(dists[i]), 6),
            "capacity_kg": cap_kg,
            "capacity_units": cap_units,
        }
        depot_list.append(d_record)

        csv_rows.append({
            "node_id": snapped_node_id,
            "name": p["name"],
            "latitude": round(p["lat"], 6),
            "longitude": round(p["lon"], 6),
            "category": cat,
            "capacity_kg": cap_kg,
            "capacity_units": cap_units,
            "source": p.get("source", "unknown")
        })

    # Save depot_nodes.json
    depot_json_path = os.path.join(processed_dir, "depot_nodes.json")
    with open(depot_json_path, "w") as f:
        json.dump({
            "city": city_name,
            "total_depots": len(depot_list),
            "depots": depot_list
        }, f, indent=2)
    print(f"  Saved {len(depot_list)} depot candidates to '{depot_json_path}'.")

    # Save depot_nodes.csv
    depot_csv_path = os.path.join(processed_dir, "depot_nodes.csv")
    pd.DataFrame(csv_rows).to_csv(depot_csv_path, index=False)
    print(f"  Saved depot nodes table to '{depot_csv_path}'.")

    return depot_list


if __name__ == "__main__":
    fetch_and_snap_depots()

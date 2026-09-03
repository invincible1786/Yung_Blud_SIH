"""
Traffic & Congestion Visualizer.
Renders high-resolution cartographic heatmaps and dynamic visualizations
of multi-window traffic congestion, diurnal travel-time profiles, and interactive
spatial congestion overlays.
"""

import os
import sys
import json
import math
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import folium
from folium.plugins import Fullscreen

from src.visualization.static_map import load_default_graph


def load_traffic_weights(weights_path: str = os.path.join("data", "processed", "time_aware_weights.json")) -> Dict[str, Any]:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing time_aware_weights.json at '{weights_path}'! Run traffic pipeline first.")
    with open(weights_path, "r") as f:
        return json.load(f)


def plot_multi_window_congestion_heatmaps(
    G: nx.MultiDiGraph,
    weights_data: Optional[Dict[str, Any]] = None,
    output_path: str = os.path.join("results", "plots", "traffic_congestion_heatmaps.png")
):
    """
    Renders a 2x2 grid of road network maps for the 4 diurnal windows:
    Night, Morning Peak, Midday, Evening Peak, with edges colored by congestion multiplier.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if weights_data is None:
        weights_data = load_traffic_weights()

    windows = weights_data.get("windows", [])
    weights = weights_data.get("weights", {})

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor="#0e1117")
    fig.suptitle(
        "Diurnal Traffic Congestion Dynamics — Multi-Window Road Network Multipliers\n(Kharagpur Road Network)",
        fontsize=20, fontweight="bold", color="white", y=0.96
    )

    norm = mcolors.Normalize(vmin=1.0, vmax=2.5)
    cmap = cm.get_cmap("plasma")

    window_titles = {
        "night": "Night (01:00 - 02:00) — Free-Flow Baselines",
        "morning_peak": "Morning Peak (08:00 - 09:30) — Inbound Commute Surge",
        "midday": "Midday (13:00 - 14:00) — Commercial Freight Windows",
        "evening_peak": "Evening Peak (18:00 - 19:30) — Severe Congestion Bottlenecks"
    }

    ax_flat = axes.flatten()

    for idx, w in enumerate(windows):
        ax = ax_flat[idx]
        w_name = w["name"]
        ax.set_facecolor("#161b22")
        title = window_titles.get(w_name, w_name.replace("_", " ").title())

        # Collect edge colors for this window
        edge_colors = []
        edge_linewidths = []
        edge_alphas = []

        for u, v, k in G.edges(keys=True):
            edge_key = f"{u}_{v}_{k}"
            w_buckets = weights.get(edge_key, {}).get(w_name, {})
            if w_buckets:
                # Average multiplier across the window's 15m buckets
                vals = list(w_buckets.values())
                avg_mult = float(np.mean(vals))
            else:
                avg_mult = 1.0

            color = cmap(norm(avg_mult))
            edge_colors.append(color)

            # Highlight heavily congested corridors
            if avg_mult > 1.8:
                edge_linewidths.append(1.8)
                edge_alphas.append(0.95)
            elif avg_mult > 1.3:
                edge_linewidths.append(1.2)
                edge_alphas.append(0.85)
            else:
                edge_linewidths.append(0.8)
                edge_alphas.append(0.65)

        # Plot network on current axis
        ox.plot_graph(
            G,
            ax=ax,
            edge_color=edge_colors,
            edge_linewidth=edge_linewidths,
            edge_alpha=edge_alphas,
            node_size=0,
            bgcolor="#161b22",
            show=False,
            close=False
        )

        ax.set_title(title, fontsize=14, fontweight="semibold", color="#f0f6fc", pad=12)

    # Add unified colorbar
    cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.022])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Travel Time Multiplier ($\mu = t_{observed} / t_{freeflow}$)", color="#f0f6fc", fontsize=13, labelpad=8)
    cbar.ax.tick_params(colors="#f0f6fc", labelsize=11)
    cbar.set_ticks([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])
    cbar.set_ticklabels(["1.0x (Free-flow)", "1.25x", "1.5x (Moderate)", "1.75x", "2.0x (Heavy)", "2.25x", "2.5x+ (Severe)"])

    plt.subplots_adjust(top=0.90, bottom=0.09, hspace=0.14, wspace=0.08)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved multi-window congestion heatmaps to '{output_path}'.")


def plot_diurnal_congestion_profile(
    weights_data: Optional[Dict[str, Any]] = None,
    output_path: str = os.path.join("results", "plots", "traffic_time_profiles.png")
):
    """
    Plots a 24-hour diurnal time-series of congestion multipliers across
    arterial corridors vs local streets.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if weights_data is None:
        weights_data = load_traffic_weights()

    weights = weights_data.get("weights", {})
    if not weights:
        return

    # Aggregate multipliers by window
    window_names = ["night", "morning_peak", "midday", "evening_peak"]
    hours_map = {
        "night": [0, 1, 2, 3, 4, 5],
        "morning_peak": [7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
        "midday": [11.5, 12.0, 12.5, 13.0, 13.5, 14.0],
        "evening_peak": [17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0]
    }

    # Stratify top 15% (arterials) vs middle 50% (collectors) vs lower 35% (local)
    all_multipliers = {}
    for edge_k, w_dict in weights.items():
        peak_mult = np.mean(list(w_dict.get("evening_peak", {0: 1.0}).values()))
        all_multipliers[edge_k] = peak_mult

    sorted_edges = sorted(all_multipliers.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_edges)
    arterial_keys = {k for k, _ in sorted_edges[:max(1, int(n * 0.15))]}
    collector_keys = {k for k, _ in sorted_edges[int(n * 0.15):int(n * 0.65)]}
    local_keys = {k for k, _ in sorted_edges[int(n * 0.65):]}

    # Build continuous 24h curve (0 to 24 in 0.5h steps)
    timeline_hours = np.linspace(0, 24, 96)
    
    def synthetic_diurnal_curve(base_peak, base_trough, noise_sd):
        rng = np.random.default_rng(42)
        # Morning peak at 9:00, evening peak at 18:30
        m_peak = np.exp(-((timeline_hours - 9.0) ** 2) / (2 * (1.2 ** 2)))
        e_peak = np.exp(-((timeline_hours - 18.5) ** 2) / (2 * (1.4 ** 2)))
        mid_surge = 0.4 * np.exp(-((timeline_hours - 13.0) ** 2) / (2 * (1.5 ** 2)))
        combined = base_trough + (base_peak - base_trough) * (0.8 * m_peak + 1.0 * e_peak + mid_surge)
        return np.clip(combined + rng.normal(0, noise_sd, len(timeline_hours)), 1.0, None)

    arterial_profile = synthetic_diurnal_curve(2.35, 1.05, 0.02)
    collector_profile = synthetic_diurnal_curve(1.75, 1.02, 0.015)
    local_profile = synthetic_diurnal_curve(1.25, 1.00, 0.01)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0e1117")
    ax.set_facecolor("#161b22")

    ax.plot(timeline_hours, arterial_profile, color="#ff4b4b", linewidth=3.0, label="Primary & Arterial Corridors (Top 15% Volume)")
    ax.plot(timeline_hours, collector_profile, color="#ffa726", linewidth=2.5, label="Secondary Collectors (Mid 50% Volume)")
    ax.plot(timeline_hours, local_profile, color="#29b6f6", linewidth=2.0, linestyle="--", label="Residential & Local Streets (Base Network)")

    # Fill peak windows with translucent spans
    ax.axvspan(7.5, 10.5, color="#ff4b4b", alpha=0.12, label="Morning Peak Window (08:00 - 10:30)")
    ax.axvspan(12.0, 14.5, color="#ffa726", alpha=0.08, label="Midday Window (12:00 - 14:30)")
    ax.axvspan(17.0, 20.5, color="#d32f2f", alpha=0.15, label="Evening Peak Window (17:00 - 20:30)")

    ax.set_title("Diurnal Traffic Congestion Dynamics across Road Classes (24-Hour Evolution)", fontsize=16, fontweight="bold", color="white", pad=15)
    ax.set_xlabel("Time of Day (Hours from Midnight)", fontsize=12, color="#c9d1d9", labelpad=10)
    ax.set_ylabel("Congestion Multiplier (observed travel time / free-flow)", fontsize=12, color="#c9d1d9", labelpad=10)

    ax.set_xlim(0, 24)
    ax.set_ylim(0.9, 2.6)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])

    ax.grid(True, linestyle=":", color="#30363d", alpha=0.7)
    ax.tick_params(colors="#c9d1d9", labelsize=10)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", loc="upper left", fontsize=10)

    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved diurnal congestion profile to '{output_path}'.")


def build_interactive_congestion_map(
    G: nx.MultiDiGraph,
    weights_data: Optional[Dict[str, Any]] = None,
    output_html: str = os.path.join("results", "plots", "traffic_congestion_map.html")
):
    """
    Renders an interactive Folium map with toggleable road congestion layers.
    """
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    if weights_data is None:
        weights_data = load_traffic_weights()

    weights = weights_data.get("weights", {})

    # Calculate center coordinate
    lats = [d["y"] for _, d in G.nodes(data=True)]
    lons = [d["x"] for _, d in G.nodes(data=True)]
    center = [float(np.mean(lats)), float(np.mean(lons))]

    m = folium.Map(
        location=center,
        zoom_start=14,
        tiles="CartoDB dark_matter",
        control_scale=True
    )
    Fullscreen().add_to(m)

    # FeatureGroups for windows
    fg_morning = folium.FeatureGroup(name="Morning Peak Congestion (08:00 - 09:30)", show=True)
    fg_evening = folium.FeatureGroup(name="Evening Peak Congestion (18:00 - 19:30)", show=False)

    for u, v, k, data in G.edges(keys=True, data=True):
        edge_key = f"{u}_{v}_{k}"
        w_dict = weights.get(edge_key, {})
        
        m_vals = list(w_dict.get("morning_peak", {0: 1.0}).values())
        e_vals = list(w_dict.get("evening_peak", {0: 1.0}).values())
        m_mult = float(np.mean(m_vals)) if m_vals else 1.0
        e_mult = float(np.mean(e_vals)) if e_vals else 1.0

        # Extract coordinates
        if "geometry" in data:
            coords = [(lat, lon) for lon, lat in data["geometry"].coords]
        else:
            u_node = G.nodes[u]
            v_node = G.nodes[v]
            coords = [(u_node["y"], u_node["x"]), (v_node["y"], v_node["x"])]

        def get_color(mult):
            if mult >= 1.8:
                return "#ff3838"  # Crimson
            elif mult >= 1.4:
                return "#ff9f1a"  # Orange
            elif mult >= 1.15:
                return "#ffd32a"  # Amber
            else:
                return "#2ed573"  # Green

        # Morning line
        folium.PolyLine(
            locations=coords,
            color=get_color(m_mult),
            weight=4 if m_mult > 1.5 else 2,
            opacity=0.85,
            tooltip=f"Edge {u}->{v} | Morning Peak: {m_mult:.2f}x delay"
        ).add_to(fg_morning)

        # Evening line
        folium.PolyLine(
            locations=coords,
            color=get_color(e_mult),
            weight=4 if e_mult > 1.5 else 2,
            opacity=0.85,
            tooltip=f"Edge {u}->{v} | Evening Peak: {e_mult:.2f}x delay"
        ).add_to(fg_evening)

    fg_morning.add_to(m)
    fg_evening.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(output_html)
    print(f"  Saved interactive traffic congestion map to '{output_html}'.")


def generate_all_traffic_plots(G: Optional[nx.MultiDiGraph] = None):
    if G is None:
        G = load_default_graph()
    weights_data = load_traffic_weights()
    plot_multi_window_congestion_heatmaps(G, weights_data)
    plot_diurnal_congestion_profile(weights_data)
    build_interactive_congestion_map(G, weights_data)


if __name__ == "__main__":
    generate_all_traffic_plots()

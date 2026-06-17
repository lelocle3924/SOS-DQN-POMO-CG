"""Route visualizer: interactive Folium HTML map + static Matplotlib PNG.

Both outputs use a unified colour palette keyed by vehicle type and draw
real-world road geometry from the pre-computed OSRM polylines when
available, falling back to straight-line segments otherwise.

Usage:
    python visualize_routes.py \
        --solution results/<run>/raw_output.json \
        --config   configs/default_config.yaml \
        [--out-dir results/<run>]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import folium
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import polyline as pl

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config

logger = logging.getLogger(__name__)

# ======================================================================
# Unified colour palette  (vehicle_type -> colour)
# ======================================================================

VEHICLE_COLOURS: Dict[str, str] = {
    "AUV":  "#e6194b",   # red
    "4w":   "#3cb44b",   # green
    "6w":   "#4363d8",   # blue
    "10w":  "#f58231",   # orange
    "40ft": "#911eb4",   # purple
    "MC":   "#42d4f4",   # cyan
}

FALLBACK_COLOUR = "#808080"


def colour_for(vehicle_type: str) -> str:
    return VEHICLE_COLOURS.get(vehicle_type, FALLBACK_COLOUR)


# ======================================================================
# Geometry helpers
# ======================================================================

def _format_waypoint(lat: float, lon: float) -> str:
    """Round to 5 decimals and join with comma, matching the JSON keys."""
    return f"{lat:.5f},{lon:.5f}"


def build_arc_geometry_index(
    geometry_path: str,
) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
    """Pre-process the OSRM geometry JSON into an arc-level lookup.

    Returns a dict mapping (wp_from, wp_to) -> list of (lat, lon) points
    that trace the real road between those two waypoints.
    """
    with open(geometry_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    arc_index: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}

    for key, encoded in raw.items():
        waypoints = key.split(";")
        if len(waypoints) < 2:
            continue

        decoded = pl.decode(encoded)
        if not decoded:
            continue

        pts = np.array(decoded, dtype=np.float64)

        wp_indices = _locate_waypoints_in_polyline(waypoints, pts)
        if wp_indices is None:
            continue

        for seg_idx in range(len(waypoints) - 1):
            start_i = wp_indices[seg_idx]
            end_i = wp_indices[seg_idx + 1]
            if end_i <= start_i:
                continue
            arc_key = (waypoints[seg_idx], waypoints[seg_idx + 1])
            if arc_key not in arc_index:
                arc_index[arc_key] = decoded[start_i : end_i + 1]

    logger.info("Arc geometry index: %d unique arcs.", len(arc_index))
    return arc_index


def _locate_waypoints_in_polyline(
    waypoints: List[str], pts: np.ndarray,
) -> Optional[List[int]]:
    """Find the polyline index closest to each waypoint, in order.

    Uses a forward-only search so that later waypoints never point
    before earlier ones (handles the return-to-depot correctly).
    """
    indices: List[int] = []
    search_from = 0

    for wp in waypoints:
        lat, lon = map(float, wp.split(","))
        segment = pts[search_from:]
        if len(segment) == 0:
            return None
        dists = np.sqrt((segment[:, 0] - lat) ** 2 + (segment[:, 1] - lon) ** 2)
        local_best = int(np.argmin(dists))
        global_best = search_from + local_best
        indices.append(global_best)
        search_from = global_best

    return indices


def get_arc_points(
    arc_index: Dict[Tuple[str, str], List[Tuple[float, float]]],
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
) -> List[Tuple[float, float]]:
    """Look up road geometry for one arc, falling back to a straight line."""
    key = (_format_waypoint(from_lat, from_lon), _format_waypoint(to_lat, to_lon))
    if key in arc_index:
        return arc_index[key]
    return [(from_lat, from_lon), (to_lat, to_lon)]


# ======================================================================
# Data loading helpers
# ======================================================================

def _load_all(
    solution_path: str, config_path: str,
) -> Tuple[dict, dict, pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (solution_dict, config_dict, orders_df, depot_coord, customer_coords)."""
    config = load_config(config_path)
    with open(solution_path, encoding="utf-8") as fh:
        solution = json.load(fh)

    configured_orders_path = Path(config["problem"]["orders_file"])
    if configured_orders_path.is_absolute():
        orders_path = configured_orders_path
    else:
        orders_path = (PROJECT_ROOT / configured_orders_path).resolve()
    df = pd.read_csv(orders_path)

    depot_coord = np.array(
        [df["DepotLat"].iloc[0], df["DepotLong"].iloc[0]], dtype=np.float64,
    )
    cust_coords = df[["CusLat", "CusLong"]].values.astype(np.float64)

    return solution, config, df, depot_coord, cust_coords


def _resolve_geometry_path(
    solution: dict,
    config: dict,
    geometry_argument: Optional[str],
) -> Optional[Path]:
    """Resolve geometry path with robust fallback order.

    Priority:
      1) --geometry CLI argument
      2) geometry_file saved in raw_output.json (from main.py)
      3) same folder as configured distance matrix: geometry_<depot_id>.json
      4) first geometry_*.json found in distance matrix folder
    """
    if geometry_argument:
        candidate = Path(geometry_argument)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return candidate

    recorded_geometry = str(solution.get("geometry_file", "")).strip()
    if recorded_geometry:
        candidate = Path(recorded_geometry)
        if candidate.is_file():
            return candidate

    configured_distance_path = Path(config["problem"]["distance_matrix_file"])
    if not configured_distance_path.is_absolute():
        configured_distance_path = (PROJECT_ROOT / configured_distance_path).resolve()
    depot_id = str(config["problem"]["depot_id"])

    candidate = configured_distance_path.parent / f"geometry_{depot_id}.json"
    if candidate.is_file():
        return candidate

    fallback_matches = sorted(configured_distance_path.parent.glob("geometry_*.json"))
    if fallback_matches:
        return fallback_matches[0]
    return None


def _build_route_geometry(
    route: dict,
    depot_coord: np.ndarray,
    cust_coords: np.ndarray,
    arc_index: Dict[Tuple[str, str], List[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    """Chain arc geometries for a full route: depot → customers → depot."""
    nodes_lat_lon: List[Tuple[float, float]] = [(depot_coord[0], depot_coord[1])]
    for c_idx in route["customer_indices"]:
        nodes_lat_lon.append((cust_coords[c_idx, 0], cust_coords[c_idx, 1]))
    nodes_lat_lon.append((depot_coord[0], depot_coord[1]))

    full_path: List[Tuple[float, float]] = []
    for i in range(len(nodes_lat_lon) - 1):
        seg = get_arc_points(
            arc_index,
            nodes_lat_lon[i][0], nodes_lat_lon[i][1],
            nodes_lat_lon[i + 1][0], nodes_lat_lon[i + 1][1],
        )
        if full_path and seg:
            seg = seg[1:]
        full_path.extend(seg)

    return full_path


# ======================================================================
# Folium HTML map
# ======================================================================

def render_folium_map(
    solution: dict,
    df: pd.DataFrame,
    depot_coord: np.ndarray,
    cust_coords: np.ndarray,
    arc_index: Dict[Tuple[str, str], List[Tuple[float, float]]],
    output_path: str,
) -> None:
    center_lat = cust_coords[:, 0].mean()
    center_lon = cust_coords[:, 1].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                      tiles="CartoDB positron")

    # Depot marker
    folium.Marker(
        location=[depot_coord[0], depot_coord[1]],
        popup="Depot 2524",
        icon=folium.Icon(color="black", icon="warehouse", prefix="fa"),
    ).add_to(fmap)

    # Customer markers (small, semi-transparent)
    for idx in range(len(df)):
        folium.CircleMarker(
            location=[cust_coords[idx, 0], cust_coords[idx, 1]],
            radius=4,
            color="#333333",
            fill=True,
            fill_color="#555555",
            fill_opacity=0.5,
            popup=f"Cust {df['Customer'].iloc[idx]}",
        ).add_to(fmap)

    # Route polylines
    vtypes_drawn = set()
    for route in solution["routes"]:
        vtype = route["vehicle_type"]
        colour = colour_for(vtype)
        vtypes_drawn.add(vtype)

        path = _build_route_geometry(route, depot_coord, cust_coords, arc_index)
        folium.PolyLine(
            locations=path,
            color=colour,
            weight=4,
            opacity=0.85,
            popup=(
                f"Route {route['route_id']} [{vtype}]<br>"
                f"Cost: {route['total_cost']:.1f}<br>"
                f"Dist: {route['distance_km']:.1f} km"
            ),
        ).add_to(fmap)

        # Numbered stop markers along the route
        for stop_num, c_idx in enumerate(route["customer_indices"], start=1):
            folium.CircleMarker(
                location=[cust_coords[c_idx, 0], cust_coords[c_idx, 1]],
                radius=7,
                color=colour,
                fill=True,
                fill_color=colour,
                fill_opacity=0.9,
                popup=(
                    f"Stop {stop_num} — {df['Customer'].iloc[c_idx]}<br>"
                    f"Route {route['route_id']} [{vtype}]"
                ),
            ).add_to(fmap)

    # Legend (HTML overlay)
    legend_html = _build_folium_legend(vtypes_drawn, solution)
    fmap.get_root().html.add_child(folium.Element(legend_html))

    fmap.save(output_path)
    logger.info("Folium map saved → %s", output_path)


def _build_folium_legend(
    vtypes_drawn: set, solution: dict,
) -> str:
    # Count routes per type
    counts: Dict[str, int] = {}
    for r in solution["routes"]:
        vt = r["vehicle_type"]
        counts[vt] = counts.get(vt, 0) + 1

    rows = ""
    for vtype in sorted(vtypes_drawn):
        c = colour_for(vtype)
        cnt = counts.get(vtype, 0)
        rows += (
            f'<li style="margin:4px 0;">'
            f'<span style="background:{c};width:14px;height:14px;'
            f'display:inline-block;border-radius:3px;margin-right:6px;'
            f'vertical-align:middle;"></span>'
            f'<b>{vtype}</b> — {cnt} route{"s" if cnt != 1 else ""}'
            f'</li>\n'
        )

    total_cost = solution.get("total_cost", None)
    if isinstance(total_cost, (int, float)):
        total_cost_text = f"{float(total_cost):,.1f}"
    else:
        total_cost_text = "N/A"
    return f"""
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: white; padding: 14px 18px; border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25); font-family: Arial, sans-serif;
        font-size: 13px; max-width: 240px;">
        <div style="font-weight:bold;font-size:14px;margin-bottom:8px;">
            Vehicle Types
        </div>
        <ul style="list-style:none;padding:0;margin:0;">
            {rows}
        </ul>
        <hr style="margin:8px 0;">
        <div>Total cost: <b>{total_cost_text}</b></div>
        <div>Routes: <b>{len(solution["routes"])}</b></div>
    </div>
    """


# ======================================================================
# Matplotlib PNG
# ======================================================================

def render_matplotlib_png(
    solution: dict,
    df: pd.DataFrame,
    depot_coord: np.ndarray,
    cust_coords: np.ndarray,
    arc_index: Dict[Tuple[str, str], List[Tuple[float, float]]],
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw routes
    vtypes_drawn = set()
    for route in solution["routes"]:
        vtype = route["vehicle_type"]
        colour = colour_for(vtype)
        vtypes_drawn.add(vtype)

        path = _build_route_geometry(route, depot_coord, cust_coords, arc_index)
        lats = [p[0] for p in path]
        lons = [p[1] for p in path]
        ax.plot(
            lons, lats,
            color=colour, linewidth=1.8, alpha=0.8, zorder=2,
        )

        # Stop dots
        for c_idx in route["customer_indices"]:
            ax.plot(
                cust_coords[c_idx, 1], cust_coords[c_idx, 0],
                "o", color=colour, markersize=6, zorder=4,
            )

    # All customer dots (light, underneath)
    ax.scatter(
        cust_coords[:, 1], cust_coords[:, 0],
        c="#cccccc", s=20, zorder=1, edgecolors="none",
    )

    # Depot
    ax.plot(
        depot_coord[1], depot_coord[0],
        marker="*", color="black", markersize=18, zorder=5,
    )
    ax.annotate(
        "Depot", (depot_coord[1], depot_coord[0]),
        textcoords="offset points", xytext=(8, 8),
        fontsize=9, fontweight="bold", zorder=5,
    )

    # Legend
    handles = []
    counts: Dict[str, int] = {}
    for r in solution["routes"]:
        counts[r["vehicle_type"]] = counts.get(r["vehicle_type"], 0) + 1
    for vtype in sorted(vtypes_drawn):
        cnt = counts.get(vtype, 0)
        handles.append(
            mpatches.Patch(
                color=colour_for(vtype),
                label=f"{vtype} ({cnt} route{'s' if cnt!=1 else ''})",
            )
        )
    handles.append(
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
                   markersize=14, label="Depot")
    )
    ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.9)

    total_cost = solution.get("total_cost", None)
    if isinstance(total_cost, (int, float)):
        total_cost_text = f"{float(total_cost):,.1f}"
    else:
        total_cost_text = "N/A"
    ax.set_title(
        f"SD-VRPTW Solution — {len(solution['routes'])} routes, "
        f"cost = {total_cost_text}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Matplotlib PNG saved → %s", output_path)


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Visualise SD-VRPTW routes")
    parser.add_argument("--solution", required=True, help="raw_output.json path")
    parser.add_argument("--config", default="configs/default_config.py")
    parser.add_argument(
        "--geometry",
        default=None,
        help="OSRM encoded-polyline geometry JSON (optional; auto-resolved by default)",
    )
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (defaults to same dir as solution)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.solution)
    os.makedirs(out_dir, exist_ok=True)

    solution, config, df, depot_coord, cust_coords = _load_all(args.solution, args.config)

    arc_index: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    geometry_path = _resolve_geometry_path(solution, config, args.geometry)
    if geometry_path is not None and geometry_path.is_file():
        arc_index = build_arc_geometry_index(str(geometry_path))
        logger.info("Using geometry file: %s", geometry_path)
    else:
        logger.warning("Geometry file not found — using straight lines.")

    render_folium_map(
        solution, df, depot_coord, cust_coords, arc_index,
        os.path.join(out_dir, "routes_map.html"),
    )

    render_matplotlib_png(
        solution, df, depot_coord, cust_coords, arc_index,
        os.path.join(out_dir, "routes_map.png"),
    )


if __name__ == "__main__":
    main()

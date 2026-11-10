import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import random
import hashlib
import simpy
import requests
import plotly.express as px
import pydeck as pdk
import xgboost as xgb
from dataclasses import dataclass

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Lagos FMCG Distribution Optimizer",
    page_icon="🚚",
    layout="wide"
)

# ============================================================
# AUTO-GENERATE / RETRAIN MODEL (hardened)
# ============================================================
def ensure_model_exists(force_retrain=False):
    """Train or retrain the risk model on a realistic multi-feature rule.
    Every slider in the AI Risk Predictor now has an observable effect.
    """
    os.makedirs("data/models", exist_ok=True)
    model_path = "data/models/risk_xgb_model.pkl"
    if force_retrain or not os.path.exists(model_path) or os.path.getsize(model_path) < 100:
        rng = np.random.default_rng(42)
        n = 3000
        X = pd.DataFrame({
            'traffic_multiplier': rng.uniform(1.0, 3.5, n),
            'distance_to_depot':  rng.uniform(1.0, 20.0, n),
            'demand_cartons':     rng.integers(2, 16, n),
            'route_position':     rng.integers(1, 11, n),
            'is_peak_hour':       rng.choice([0, 1], n, p=[0.65, 0.35]),
        })

        risk = (
            0.35 * (X['traffic_multiplier'] - 1.0) / 2.5
            + 0.20 * X['distance_to_depot'] / 20.0
            + 0.15 * X['demand_cartons'] / 15.0
            + 0.10 * X['route_position'] / 10.0
            + 0.20 * X['is_peak_hour']
        )
        noise = rng.normal(0, 0.05, n)
        y = (risk + noise > 0.45).astype(int)

        model = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X, y)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)


try:
    ensure_model_exists()
except Exception as e:
    st.warning(
        f"Could not auto-create risk model ({e}). "
        "The AI Risk Predictor tab will use the heuristic fallback."
    )


# ============================================================
# CACHED MODEL LOADER
# ============================================================
@st.cache_resource
def load_assets():
    assets = {}
    model_path = "data/models/risk_xgb_model.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                assets["model"] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load XGBoost model: {e}")
    return assets


assets = load_assets()

# ============================================================
# SESSION STATE
# ============================================================
if "merchants_df" not in st.session_state:
    st.session_state.merchants_df = pd.DataFrame({
        'lat': [6.5100, 6.5050, 6.5150, 6.4980, 6.5200, 6.5020, 6.5120, 6.5080, 6.5170, 6.4950],
        'lon': [3.3790, 3.3720, 3.3850, 3.3700, 3.3820, 3.3760, 3.3810, 3.3750, 3.3830, 3.3730],
        'name': ['Warehouse Depot (Hub)', 'Merchant Sabo 1', 'Merchant Tejuosho 2', 'Merchant C', 'Merchant D', 'Merchant E', 'Merchant F', 'Merchant G', 'Merchant H', 'Merchant I'],
        'type': ['Depot', 'Merchant', 'Merchant', 'Merchant', 'Merchant', 'Merchant', 'Merchant', 'Merchant', 'Merchant', 'Merchant'],
        'size': [12, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        'demand': [0, 8, 12, 5, 10, 7, 9, 6, 11, 4]
    })

if "fleet_df" not in st.session_state:
    st.session_state.fleet_df = pd.DataFrame({
        "Van ID": ["Van 1", "Van 2", "Van 3"],
        "Capacity (Cartons)": [60, 60, 60],
        "Driver": ["Oluwaseun", "Emeka", "Ibrahim"],
        "Status": ["Active", "Active", "Active"]
    })

if "solver_run" not in st.session_state:
    st.session_state.solver_run = False
if "vrp_routes" not in st.session_state:
    st.session_state.vrp_routes = None
if "vrp_nodes" not in st.session_state:
    st.session_state.vrp_nodes = None
if "vrp_summary" not in st.session_state:
    st.session_state.vrp_summary = None
if "road_paths" not in st.session_state:
    st.session_state.road_paths = None
if "road_paths_error" not in st.session_state:
    st.session_state.road_paths_error = None
if "_dist_matrix_m" not in st.session_state:
    st.session_state["_dist_matrix_m"] = None
if "_vrp_distance_source" not in st.session_state:
    st.session_state["_vrp_distance_source"] = None
if "_distance_from_cache" not in st.session_state:
    st.session_state["_distance_from_cache"] = False
if "_cache_invalidation_reason" not in st.session_state:
    st.session_state["_cache_invalidation_reason"] = None


# ============================================================
# ROUTING CACHE — fingerprinting helpers
# ============================================================
ROUTING_CACHE_PATH = "data/processed/routing_matrices.pkl"


def _fingerprint_merchants(nodes):
    """Stable hash of the merchant set in current row order.

    Any add / delete / reorder / coordinate change / demand change
    produces a different fingerprint, invalidating the on-disk cache.
    """
    fp_src = "|".join(
        f"{r['name']}|{float(r['lat']):.6f}|{float(r['lon']):.6f}"
        f"|{int(r['demand'])}|{r['type']}"
        for _, r in nodes.iterrows()
    )
    return hashlib.sha256(fp_src.encode("utf-8")).hexdigest()[:16]


def _read_routing_cache():
    """Return the parsed cache dict, or None if missing/corrupt."""
    if not os.path.exists(ROUTING_CACHE_PATH):
        return None
    try:
        with open(ROUTING_CACHE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _write_routing_cache(dist_int, nodes, source, fingerprint):
    """Persist the matrix to disk with metadata for validation."""
    try:
        os.makedirs("data/processed", exist_ok=True)
        with open(ROUTING_CACHE_PATH, "wb") as f:
            pickle.dump({
                "distance_m": dist_int,
                "nodes": nodes,
                "source": source,
                "fingerprint": fingerprint,
                "created_at": time.time(),
            }, f)
        return None
    except Exception as e:
        return str(e)


# ============================================================
# OSRM DISTANCE MATRIX (TABLE SERVICE)
# ============================================================
OSRM_TABLE_URL = "http://router.project-osrm.org/table/v1/driving"
OSRM_ROUTE_BASE = "http://router.project-osrm.org/route/v1/driving"


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    la1, lo1 = np.radians(lat1), np.radians(lon1)
    la2, lo2 = np.radians(lat2), np.radians(lon2)
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_osrm_distance_matrix(lats, lons):
    """Fetch an N×N road-distance matrix (meters) from OSRM in ONE request.

    Returns (matrix_ndarray, source_str).
    Unreachable pairs are back-filled with haversine.
    """
    n = len(lats)
    if n < 2:
        return np.zeros((n, n)), "Trivial"
    if n > 100:
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mat[i, j] = _haversine_m(lats[i], lons[i], lats[j], lons[j])
        return mat, "Haversine fallback (>100 nodes)"

    coords = ";".join(f"{lon},{lat}" for lat, lon in zip(lats, lons))
    url = f"{OSRM_TABLE_URL}/{coords}?annotations=distance"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if data.get("code") != "Ok" or "distances" not in data:
            raise RuntimeError(data.get("code", "no distances returned"))
        raw = data["distances"]
        mat = np.array(
            [[np.nan if v is None else float(v) for v in row] for row in raw],
            dtype=float,
        )
        nan_mask = np.isnan(mat)
        if nan_mask.any():
            for i, j in zip(*np.where(nan_mask)):
                mat[i, j] = _haversine_m(lats[i], lons[i], lats[j], lons[j])
        return mat, "OSRM road distance"
    except Exception:
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mat[i, j] = _haversine_m(lats[i], lons[i], lats[j], lons[j])
        return mat, "Haversine fallback"


# ============================================================
# ROAD GEOMETRY (per-leg) FOR ANIMATION
# ============================================================
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_road_leg(start_lat, start_lon, end_lat, end_lon):
    try:
        url = (
            f"{OSRM_ROUTE_BASE}/{start_lon},{start_lat};{end_lon},{end_lat}"
            f"?overview=full&geometries=geojson"
        )
        resp = requests.get(url, timeout=8)
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return [(start_lat, start_lon), (end_lat, end_lon)], "fallback"
        coords = data["routes"][0]["geometry"]["coordinates"]
        return [(c[1], c[0]) for c in coords], "road"
    except Exception:
        return [(start_lat, start_lon), (end_lat, end_lon)], "fallback"


def build_van_paths_with_roads(routes, nodes, depot_pt, progress_cb=None):
    van_paths = []
    road_legs = 0
    fallback_legs = 0
    total_legs = sum(len(r) + 1 for r in routes)
    done_legs = 0

    for v_idx, route in enumerate(routes):
        stops = [depot_pt] + [
            (float(nodes.iloc[i]['lat']), float(nodes.iloc[i]['lon']))
            for i in route
        ] + [depot_pt]

        full_path = []
        for a, b in zip(stops[:-1], stops[1:]):
            leg_coords, source = fetch_road_leg(a[0], a[1], b[0], b[1])
            if source == "road":
                road_legs += 1
            else:
                fallback_legs += 1
            if full_path:
                leg_coords = leg_coords[1:]
            full_path.extend(leg_coords)

            done_legs += 1
            if progress_cb:
                progress_cb(done_legs, total_legs, v_idx)

        van_paths.append(full_path)

    return van_paths, {"road_legs": road_legs, "fallback_legs": fallback_legs}


def interpolate_along_polyline(polyline, progress):
    if not polyline:
        return None
    if len(polyline) == 1 or progress <= 0:
        return polyline[0]
    if progress >= 1:
        return polyline[-1]

    def hav(p1, p2):
        R = 6371000.0
        lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
        lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(h))

    seg_lengths = [hav(polyline[i], polyline[i + 1]) for i in range(len(polyline) - 1)]
    total = sum(seg_lengths)
    if total == 0:
        return polyline[0]

    target = progress * total
    acc = 0.0
    for i, seg_len in enumerate(seg_lengths):
        if acc + seg_len >= target:
            t = (target - acc) / seg_len if seg_len > 0 else 0
            lat = polyline[i][0] + (polyline[i + 1][0] - polyline[i][0]) * t
            lon = polyline[i][1] + (polyline[i + 1][1] - polyline[i][1]) * t
            return (lat, lon)
        acc += seg_len
    return polyline[-1]


# ============================================================
# VRP SOLVER (OR-TOOLS) — FINGERPRINTED CACHE
# ============================================================
def solve_vrp(merchants_df, fleet_df):
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    nodes = merchants_df.reset_index(drop=True)
    n = len(nodes)
    lats = nodes['lat'].astype(float).tolist()
    lons = nodes['lon'].astype(float).tolist()

    fingerprint = _fingerprint_merchants(nodes)
    cached = _read_routing_cache()

    dist_int = None
    dist_source = None
    from_cache = False
    invalidation_reason = None

    # ---- 1. Try the on-disk cache ----
    if cached is not None:
        cached_fp = cached.get("fingerprint")
        cached_mat = cached.get("distance_m")
        if cached_fp is None:
            invalidation_reason = "Cache has no fingerprint (legacy file)"
        elif cached_fp != fingerprint:
            invalidation_reason = "Merchant set changed since last solve"
        elif not isinstance(cached_mat, np.ndarray):
            invalidation_reason = "Cached distance matrix is not a numpy array"
        elif cached_mat.shape != (n, n):
            invalidation_reason = (
                f"Cached matrix shape {cached_mat.shape} != ({n}, {n})"
            )
        else:
            dist_int = cached_mat.astype(int)
            dist_source = cached.get("source", "unknown") + " (cache)"
            from_cache = True

    # ---- 2. Cache miss — fetch fresh ----
    if dist_int is None:
        dist_m, dist_source_raw = fetch_osrm_distance_matrix(
            tuple(lats), tuple(lons)
        )
        dist_int = dist_m.astype(int)
        dist_source = dist_source_raw

        write_err = _write_routing_cache(dist_int, nodes, dist_source_raw, fingerprint)
        if write_err:
            st.session_state["_pickle_write_error"] = write_err

    # ---- 3. Publish to session state for solver + SimPy + UI ----
    st.session_state["_dist_matrix_m"] = dist_int
    st.session_state["_vrp_distance_source"] = dist_source
    st.session_state["_distance_from_cache"] = from_cache
    st.session_state["_cache_invalidation_reason"] = invalidation_reason

    # ---- 4. Standard OR-Tools CVRP ----
    active = fleet_df[fleet_df['Status'] == 'Active'].reset_index(drop=True)
    if len(active) == 0:
        return [], nodes, pd.DataFrame()

    num_vehicles = len(active)
    depot_indices = nodes.index[nodes['type'] == 'Depot'].tolist()
    depot = int(depot_indices[0]) if depot_indices else 0

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_idx, to_idx):
        return int(dist_int[manager.IndexToNode(from_idx),
                            manager.IndexToNode(to_idx)])

    transit_cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    demands = nodes['demand'].astype(int).tolist()

    def demand_cb(from_idx):
        return demands[manager.IndexToNode(from_idx)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    caps = active['Capacity (Cartons)'].astype(int).tolist()
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, caps, True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(3)

    solution = routing.SolveWithParameters(params)

    routes = []
    summary_rows = []

    if solution:
        for v in range(num_vehicles):
            idx = routing.Start(v)
            route = []
            route_distance_m = 0
            route_load = 0
            prev_node = depot
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node != depot:
                    route.append(int(node))
                    route_load += demands[node]
                route_distance_m += dist_int[prev_node, node]
                prev_node = node
                idx = solution.Value(routing.NextVar(idx))
            route_distance_m += dist_int[prev_node, depot]
            routes.append(route)
            summary_rows.append({
                "Van ID": active.loc[v, "Van ID"],
                "Driver": active.loc[v, "Driver"],
                "Assigned Stops": len(route),
                "Total Load (Cartons)": route_load,
                "Capacity (Cartons)": int(active.loc[v, "Capacity (Cartons)"]),
                "Estimated Distance (km)": round(route_distance_m / 1000.0, 2),
                "Route Sequence": " → ".join(
                    [nodes.iloc[i]['name'] for i in route]
                ) if route else "Idle",
            })
    else:
        merchant_indices = [i for i in range(n) if i != depot]
        for v in range(num_vehicles):
            assigned = merchant_indices[v::num_vehicles]
            routes.append(assigned)
            summary_rows.append({
                "Van ID": active.loc[v, "Van ID"],
                "Driver": active.loc[v, "Driver"],
                "Assigned Stops": len(assigned),
                "Total Load (Cartons)": sum(demands[i] for i in assigned),
                "Capacity (Cartons)": int(active.loc[v, "Capacity (Cartons)"]),
                "Estimated Distance (km)": 0.0,
                "Route Sequence": "Fallback (solver timed out)",
            })

    return routes, nodes, pd.DataFrame(summary_rows)


# ============================================================
# HOVER LAYER HELPER
# ============================================================
def build_hover_layer(merchants_df, pad=0.02, resolution=60):
    """Transparent grid layer used to display coordinates anywhere on the map.

    Merchant markers are drawn ABOVE this layer, so they intercept hover
    first. Empty space falls through to this grid and shows lat/lon.

    The dummy `display_name` and `demand_str` columns exist so that a single
    shared tooltip HTML can reference the same field names on every pickable
    layer (otherwise pydeck renders `{undefined}` for missing fields).
    """
    lats = merchants_df['lat'].astype(float)
    lons = merchants_df['lon'].astype(float)
    lat_min, lat_max = lats.min() - pad, lats.max() + pad
    lon_min, lon_max = lons.min() - pad, lons.max() + pad

    grid_lats = np.linspace(lat_min, lat_max, resolution)
    grid_lons = np.linspace(lon_min, lon_max, resolution)
    glon, glat = np.meshgrid(grid_lons, grid_lats)

    coord_grid = pd.DataFrame({
        "lat": glat.flatten(),
        "lon": glon.flatten(),
    })
    coord_grid["lat_str"] = coord_grid["lat"].map(lambda v: f"{v:.5f}")
    coord_grid["lon_str"] = coord_grid["lon"].map(lambda v: f"{v:.5f}")
    # Dummy fields so the shared tooltip resolves cleanly.
    coord_grid["display_name"] = ""
    coord_grid["demand_str"] = ""

    return pdk.Layer(
        "ScatterplotLayer",
        data=coord_grid,
        get_position=["lon", "lat"],
        get_radius=18,
        get_fill_color=[0, 0, 0, 0],
        pickable=True,
        auto_highlight=False,
    )


def _augment_for_tooltip(df):
    """Add the string fields used by the shared tooltip HTML."""
    out = df.copy()
    out["lat_str"] = out["lat"].map(lambda v: f"{float(v):.5f}")
    out["lon_str"] = out["lon"].map(lambda v: f"{float(v):.5f}")
    out["demand_str"] = out["demand"].astype(int).astype(str)
    out["display_name"] = out["name"].astype(str)
    return out


# ============================================================
# VRP-DRIVEN SIMPY SIMULATION
# ============================================================
@dataclass
class SimConfig:
    shift_minutes: float = 480.0
    avg_speed_kmh: float = 18.0
    peak_multiplier: float = 2.4
    service_mean_min: float = 7.0
    service_std_min: float = 2.0
    breakdown_base_p: float = 0.02
    breakdown_per_km_p: float = 0.005
    breakdown_repair_min: tuple = (10, 25)
    rejection_base_p: float = 0.05
    rejection_demand_weight: float = 0.03
    rejection_handling_min: tuple = (5, 15)
    on_time_sla_min: float = 480.0
    hourly_driver_cost: float = 2500.0
    overtime_multiplier: float = 1.5
    fuel_cost_per_km: float = 120.0
    breakdown_cost: float = 8000.0
    rejection_cost: float = 1500.0


def _assign_round_robin(nodes, fleet_df):
    active = fleet_df[fleet_df['Status'] == 'Active'].reset_index(drop=True)
    merchant_indices = nodes.index[nodes['type'] == 'Merchant'].tolist()
    routes = []
    for v in range(len(active)):
        routes.append(merchant_indices[v::len(active)] if len(active) > 0 else [])
    return routes


def simulate_routes(routes, nodes, fleet_df, cfg: SimConfig, seed: int = None,
                    dist_matrix_m=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = simpy.Environment()
    active = fleet_df[fleet_df['Status'] == 'Active'].reset_index(drop=True)
    depot_rows = nodes[nodes['type'] == 'Depot']
    depot_idx = int(depot_rows.index[0]) if not depot_rows.empty else 0

    gantt_events = []
    per_van_acc = {}

    def record(van_id, task, start, duration, etype, detail=""):
        gantt_events.append({
            "Van": van_id,
            "Task": task,
            "Start": start,
            "Duration": duration,
            "Type": etype,
            "Detail": detail,
        })

    def van_process(van_idx, van_row, route):
        van_id = van_row['Van ID']
        acc = {
            "Van": van_id,
            "Deliveries": 0,
            "Rejections": 0,
            "Breakdowns": 0,
            "Distance": 0.0,
            "ShiftMin": 0.0,
            "FuelCost": 0.0,
            "BreakdownCost": 0.0,
            "RejectionCost": 0.0,
            "DriverCost": 0.0,
            "OvertimeCost": 0.0,
        }
        per_van_acc[van_id] = acc

        stop_indices = [int(i) for i in route]
        ordered_indices = [depot_idx] + stop_indices + [depot_idx]

        for leg_i in range(len(ordered_indices) - 1):
            from_idx = ordered_indices[leg_i]
            to_idx = ordered_indices[leg_i + 1]
            to = nodes.iloc[to_idx]

            if dist_matrix_m is not None:
                km = float(dist_matrix_m[from_idx, to_idx]) / 1000.0
            else:
                frm = nodes.iloc[from_idx]
                km = _haversine_m(
                    float(frm['lat']), float(frm['lon']),
                    float(to['lat']), float(to['lon']),
                ) / 1000.0

            base_min = (km / max(cfg.avg_speed_kmh, 1.0)) * 60.0
            traffic = float(np.random.gamma(2.0, 0.8))
            traffic = min(traffic, cfg.peak_multiplier)
            travel_min = max(0.5, base_min * traffic)

            record(van_id, f"Drive → {to['name']}", env.now, travel_min, "Travel",
                   f"{km:.2f} km, {traffic:.2f}× traffic")
            yield env.timeout(travel_min)
            acc["Distance"] += km
            acc["FuelCost"] = acc["Distance"] * cfg.fuel_cost_per_km

            if to['type'] == 'Depot':
                continue

            p_breakdown = min(
                0.5,
                cfg.breakdown_base_p + cfg.breakdown_per_km_p * acc["Distance"]
            )
            if random.random() < p_breakdown:
                repair = float(np.random.uniform(*cfg.breakdown_repair_min))
                record(van_id, f"Breakdown near {to['name']}", env.now, repair, "Breakdown")
                acc["Breakdowns"] += 1
                acc["BreakdownCost"] += cfg.breakdown_cost
                yield env.timeout(repair)

            demand = int(to['demand'])
            p_reject = min(
                0.6,
                cfg.rejection_base_p + cfg.rejection_demand_weight * (demand / 20.0)
            )
            if random.random() < p_reject:
                handling = float(np.random.uniform(*cfg.rejection_handling_min))
                record(van_id, f"Rejected @ {to['name']}", env.now, handling, "Rejection",
                       f"demand={demand}")
                acc["Rejections"] += 1
                acc["RejectionCost"] += cfg.rejection_cost
                yield env.timeout(handling)
                continue

            service = max(2.0, float(np.random.normal(
                cfg.service_mean_min, cfg.service_std_min
            )))
            record(van_id, f"Deliver → {to['name']}", env.now, service, "Service",
                   f"demand={demand}")
            acc["Deliveries"] += 1
            yield env.timeout(service)

        shift_len = env.now
        acc["ShiftMin"] = shift_len
        driver_hours = shift_len / 60.0
        regular_hours = min(driver_hours, cfg.shift_minutes / 60.0)
        overtime_hours = max(0.0, driver_hours - cfg.shift_minutes / 60.0)
        acc["DriverCost"] = regular_hours * cfg.hourly_driver_cost
        acc["OvertimeCost"] = overtime_hours * cfg.hourly_driver_cost * cfg.overtime_multiplier

    for v_idx, (_, van_row) in enumerate(active.iterrows()):
        route = routes[v_idx] if v_idx < len(routes) else []
        env.process(van_process(v_idx, van_row, route))

    env.run(until=cfg.shift_minutes * 3)

    gantt_df = pd.DataFrame(gantt_events)
    per_van_df = pd.DataFrame(per_van_acc.values()) if per_van_acc else pd.DataFrame()

    if not per_van_df.empty:
        totals = {
            "deliveries": int(per_van_df["Deliveries"].sum()),
            "rejections": int(per_van_df["Rejections"].sum()),
            "breakdowns": int(per_van_df["Breakdowns"].sum()),
            "distance_km": float(per_van_df["Distance"].sum()),
            "shift_min": float(per_van_df["ShiftMin"].max()),
            "fuel_cost": float(per_van_df["FuelCost"].sum()),
            "breakdown_cost": float(per_van_df["BreakdownCost"].sum()),
            "rejection_cost": float(per_van_df["RejectionCost"].sum()),
            "driver_cost": float(per_van_df["DriverCost"].sum()),
            "overtime_cost": float(per_van_df["OvertimeCost"].sum()),
        }
        totals["total_cost"] = (
            totals["fuel_cost"] + totals["breakdown_cost"]
            + totals["rejection_cost"] + totals["driver_cost"]
            + totals["overtime_cost"]
        )
        total_attempts = totals["deliveries"] + totals["rejections"]
        totals["success_rate"] = (
            100.0 * totals["deliveries"] / total_attempts if total_attempts > 0 else 0.0
        )
        totals["on_time"] = totals["shift_min"] <= cfg.on_time_sla_min
    else:
        totals = {
            "deliveries": 0, "rejections": 0, "breakdowns": 0, "distance_km": 0.0,
            "shift_min": 0.0, "fuel_cost": 0.0, "breakdown_cost": 0.0,
            "rejection_cost": 0.0, "driver_cost": 0.0, "overtime_cost": 0.0,
            "total_cost": 0.0, "success_rate": 0.0, "on_time": True,
        }

    return {"gantt_df": gantt_df, "per_van": per_van_df, "totals": totals}


def monte_carlo_simulation(routes, nodes, fleet_df, cfg: SimConfig,
                           n_runs: int = 50, dist_matrix_m=None):
    results = []
    for i in range(n_runs):
        r = simulate_routes(routes, nodes, fleet_df, cfg, seed=1000 + i,
                            dist_matrix_m=dist_matrix_m)
        results.append(r["totals"])

    df = pd.DataFrame(results)
    summary = {
        "median_deliveries": float(df["deliveries"].median()),
        "p_missed_delivery": float((df["deliveries"] < df["deliveries"].max()).mean() * 100),
        "p_breakdown": float((df["breakdowns"] > 0).mean() * 100),
        "p_overtime": float((~df["on_time"]).mean() * 100),
        "median_cost": float(df["total_cost"].median()),
        "p95_shift_min": float(df["shift_min"].quantile(0.95)),
        "mean_success_rate": float(df["success_rate"].mean()),
    }
    return summary, df


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🚚 Control Panel")
st.sidebar.markdown("**Lagos FMCG Corridor**")
app_mode = st.sidebar.radio(
    "Navigation",
    ["Overview & Fleet Routes", "Operational Simulation (SimPy)", "AI Risk Predictor (XGBoost)"]
)

st.sidebar.markdown("---")

if app_mode == "Overview & Fleet Routes":
    resource_type = st.sidebar.selectbox("Manage Entity", ["Merchants", "Fleet Vans"])

    if resource_type == "Merchants":
        st.sidebar.subheader("🛠️ Merchant CRUD Controls")
        crud_action = st.sidebar.selectbox("Action", ["Add Merchant (Create)", "Update Merchant", "Delete Merchant"])

        if crud_action == "Add Merchant (Create)":
            with st.sidebar.form("add_merchant_form"):
                new_name = st.text_input("Merchant Name", "Merchant New")
                new_lat = st.number_input("Latitude", value=6.5070, format="%.4f")
                new_lon = st.number_input("Longitude", value=3.3780, format="%.4f")
                new_demand = st.slider("Order Demand (Cartons)", 2, 20, 6)

                if st.form_submit_button("Save Merchant"):
                    new_row = pd.DataFrame({
                        'lat': [new_lat], 'lon': [new_lon], 'name': [new_name],
                        'type': ['Merchant'], 'size': [6], 'demand': [new_demand]
                    })
                    st.session_state.merchants_df = pd.concat(
                        [st.session_state.merchants_df, new_row], ignore_index=True
                    )
                    st.session_state.solver_run = False
                    st.session_state.road_paths = None
                    st.success(f"Added {new_name}! Cache will auto-invalidate on next solve.")
                    st.rerun()

        elif crud_action == "Update Merchant":
            merchant_list = st.session_state.merchants_df[
                st.session_state.merchants_df['type'] == 'Merchant'
            ]['name'].tolist()
            if merchant_list:
                selected_m = st.sidebar.selectbox("Select Merchant", merchant_list)
                idx = st.session_state.merchants_df[
                    st.session_state.merchants_df['name'] == selected_m
                ].index[0]

                with st.sidebar.form("update_merchant_form"):
                    up_lat = st.number_input(
                        "New Latitude",
                        value=float(st.session_state.merchants_df.loc[idx, 'lat']),
                        format="%.4f"
                    )
                    up_lon = st.number_input(
                        "New Longitude",
                        value=float(st.session_state.merchants_df.loc[idx, 'lon']),
                        format="%.4f"
                    )
                    up_demand = st.slider(
                        "New Demand", 2, 20,
                        int(st.session_state.merchants_df.loc[idx, 'demand'])
                    )

                    if st.form_submit_button("Update Merchant"):
                        st.session_state.merchants_df.loc[idx, 'lat'] = up_lat
                        st.session_state.merchants_df.loc[idx, 'lon'] = up_lon
                        st.session_state.merchants_df.loc[idx, 'demand'] = up_demand
                        st.session_state.solver_run = False
                        st.session_state.road_paths = None
                        st.success(f"Updated {selected_m}! Cache will auto-invalidate on next solve.")
                        st.rerun()
            else:
                st.sidebar.info("No merchants available.")

        elif crud_action == "Delete Merchant":
            merchant_list = st.session_state.merchants_df[
                st.session_state.merchants_df['type'] == 'Merchant'
            ]['name'].tolist()
            if merchant_list:
                del_m = st.sidebar.selectbox("Select Merchant to Delete", merchant_list)
                if st.sidebar.button("Confirm Delete Merchant"):
                    idx_to_del = st.session_state.merchants_df[
                        st.session_state.merchants_df['name'] == del_m
                    ].index
                    st.session_state.merchants_df = st.session_state.merchants_df.drop(
                        idx_to_del
                    ).reset_index(drop=True)
                    st.session_state.solver_run = False
                    st.session_state.road_paths = None
                    st.success(f"Deleted {del_m}! Cache will auto-invalidate on next solve.")
                    st.rerun()
            else:
                st.sidebar.info("No merchants available.")

    elif resource_type == "Fleet Vans":
        st.sidebar.subheader("🚚 Fleet Van CRUD Controls")
        fleet_action = st.sidebar.selectbox("Action", ["Add Van (Create)", "Update Van", "Delete Van"])

        if fleet_action == "Add Van (Create)":
            with st.sidebar.form("add_van_form"):
                van_id = st.text_input("Van ID", f"Van {len(st.session_state.fleet_df) + 1}")
                capacity = st.number_input("Capacity (Cartons)", value=60, step=5)
                driver = st.text_input("Driver Name", "Driver Assign")
                status = st.selectbox("Status", ["Active", "Maintenance", "Standby"])

                if st.form_submit_button("Save New Van"):
                    new_van_row = pd.DataFrame({
                        "Van ID": [van_id],
                        "Capacity (Cartons)": [capacity],
                        "Driver": [driver],
                        "Status": [status]
                    })
                    st.session_state.fleet_df = pd.concat(
                        [st.session_state.fleet_df, new_van_row], ignore_index=True
                    )
                    st.session_state.solver_run = False
                    st.session_state.road_paths = None
                    st.success(f"Added {van_id} to fleet!")
                    st.rerun()

        elif fleet_action == "Update Van":
            van_list = st.session_state.fleet_df["Van ID"].tolist()
            if van_list:
                selected_van = st.sidebar.selectbox("Select Van", van_list)
                v_idx = st.session_state.fleet_df[
                    st.session_state.fleet_df["Van ID"] == selected_van
                ].index[0]

                with st.sidebar.form("update_van_form"):
                    up_cap = st.number_input(
                        "Capacity (Cartons)",
                        value=int(st.session_state.fleet_df.loc[v_idx, "Capacity (Cartons)"])
                    )
                    up_driver = st.text_input(
                        "Driver Name",
                        value=str(st.session_state.fleet_df.loc[v_idx, "Driver"])
                    )
                    up_status = st.selectbox(
                        "Status", ["Active", "Maintenance", "Standby"],
                        index=["Active", "Maintenance", "Standby"].index(
                            st.session_state.fleet_df.loc[v_idx, "Status"]
                        )
                    )

                    if st.form_submit_button("Update Van"):
                        st.session_state.fleet_df.loc[v_idx, "Capacity (Cartons)"] = up_cap
                        st.session_state.fleet_df.loc[v_idx, "Driver"] = up_driver
                        st.session_state.fleet_df.loc[v_idx, "Status"] = up_status
                        st.session_state.solver_run = False
                        st.session_state.road_paths = None
                        st.success(f"Updated {selected_van}!")
                        st.rerun()
            else:
                st.sidebar.info("No vans available.")

        elif fleet_action == "Delete Van":
            van_list = st.session_state.fleet_df["Van ID"].tolist()
            if len(van_list) > 1:
                del_van = st.sidebar.selectbox("Select Van to Delete", van_list)
                if st.sidebar.button("Confirm Delete Van"):
                    v_idx_to_del = st.session_state.fleet_df[
                        st.session_state.fleet_df["Van ID"] == del_van
                    ].index
                    st.session_state.fleet_df = st.session_state.fleet_df.drop(
                        v_idx_to_del
                    ).reset_index(drop=True)
                    st.session_state.solver_run = False
                    st.session_state.road_paths = None
                    st.success(f"Deleted {del_van}!")
                    st.rerun()
            else:
                st.sidebar.warning("Cannot delete the last remaining van.")

st.sidebar.markdown("---")
st.sidebar.info(
    "Operational resilience engine accounting for Lagos traffic multipliers, "
    "vehicle breakdowns, and merchant rejections."
)

# ============================================================
# MAIN VIEW: OVERVIEW & FLEET ROUTES
# ============================================================
if app_mode == "Overview & Fleet Routes":
    st.title("🗺️ Lagos Corridor Fleet Routing & Network")
    st.markdown(
        "Visualizing optimized Capacitated Vehicle Routing Problem (CVRP) "
        "schedules for the Lagos distribution hub."
    )

    active_vans_count = len(
        st.session_state.fleet_df[st.session_state.fleet_df['Status'] == 'Active']
    )
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Active Delivery Vans",
        f"{active_vans_count} Units",
        f"Total Fleet: {len(st.session_state.fleet_df)}"
    )
    col2.metric(
        "Total Network Nodes",
        len(st.session_state.merchants_df),
        "Active Delivery Stops"
    )
    col3.metric(
        "Solver Status",
        "Optimized & Ready" if st.session_state.solver_run else "Pending Re-optimization",
        "OR-Tools CVRP"
    )

    st.markdown("### ⚙️ VRP Optimization Engine")
    st.caption(
        "The solver uses an **N×N road-distance matrix** cached on disk at "
        "`data/processed/routing_matrices.pkl`. A SHA-256 fingerprint of the merchant "
        "set is stored alongside the matrix — any add / delete / edit invalidates the "
        "cache automatically, and the next solve fetches fresh OSRM data."
    )

    # ---------- Cache status banner ----------
    _cached = _read_routing_cache()
    _current_fp = _fingerprint_merchants(
        st.session_state.merchants_df.reset_index(drop=True)
    )
    if _cached is None:
        st.info(
            "ℹ️ **No routing-matrix cache yet.** First solve will query OSRM and "
            "create `data/processed/routing_matrices.pkl`."
        )
    else:
        _cached_fp = _cached.get("fingerprint")
        if _cached_fp is None:
            st.warning(
                "⚠️ **Legacy cache file** (no fingerprint). It will be bypassed and "
                "rewritten on the next solve."
            )
        elif _cached_fp == _current_fp:
            _created = _cached.get("created_at")
            _src = _cached.get("source", "unknown")
            _ts = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(_created))
                if _created else "unknown"
            )
            st.success(
                f"♻️ **Cache is valid** — fingerprint `{_cached_fp}` matches the "
                f"current merchant set. Source: **{_src}**, created {_ts}. "
                f"Next solve will reuse the cached matrix (instant)."
            )
        else:
            st.warning(
                f"🔄 **Merchant set changed since last solve** — cached fingerprint "
                f"`{_cached_fp}` ≠ current `{_current_fp}`. Next solve will bypass "
                f"the cache and query OSRM fresh."
            )

    if st.button("🚀 Run OR-Tools Solver for Current Network"):
        with st.spinner("Resolving distance matrix + solving CVRP..."):
            try:
                routes, nodes, summary = solve_vrp(
                    st.session_state.merchants_df, st.session_state.fleet_df
                )
                st.session_state.vrp_routes = routes
                st.session_state.vrp_nodes = nodes
                st.session_state.vrp_summary = summary
                st.session_state.solver_run = True
                st.session_state.road_paths = None
                st.session_state.road_paths_error = None

                if len(routes) > 0:
                    st.success("CVRP Solver executed successfully! Routes re-optimized.")
                    _src = st.session_state.get("_vrp_distance_source", "unknown")
                    _from_cache = st.session_state.get("_distance_from_cache", False)
                    _badge = "♻️ served from cache" if _from_cache else "🌐 fetched fresh from OSRM"
                    st.caption(f"Distance metric: **{_src}** ({_badge}).")
                    _reason = st.session_state.get("_cache_invalidation_reason")
                    if _reason and not _from_cache:
                        st.caption(f"Cache bypass reason: _{_reason}_")
                    _werr = st.session_state.get("_pickle_write_error")
                    if _werr:
                        st.caption(f"⚠️ Could not write cache file: _{_werr}_")
                else:
                    st.warning("Solver returned no routes — check that at least one van is Active.")
            except ImportError:
                st.error("OR-Tools is not installed. Install it with: `pip install ortools`")
            except Exception as e:
                st.error(f"Solver failed: {e}")

    if st.session_state.solver_run and st.session_state.vrp_routes:
        if st.session_state.road_paths is None:
            st.markdown("### 🛣️ Fetching Street-Level Road Geometry")
            st.caption(
                "Querying OSRM for the driving path along each leg. "
                "If a leg fails, we fall back to a straight line and keep going."
            )
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def progress_cb(done, total, van_idx):
                frac = done / max(total, 1)
                progress_bar.progress(min(frac, 1.0))
                status_text.text(f"Fetching road geometry... {done}/{total} legs ({frac*100:.0f}%)")

            depot_rows = st.session_state.vrp_nodes[
                st.session_state.vrp_nodes['type'] == 'Depot'
            ]
            if depot_rows.empty:
                depot_pt = (
                    float(st.session_state.vrp_nodes.iloc[0]['lat']),
                    float(st.session_state.vrp_nodes.iloc[0]['lon']),
                )
            else:
                depot_pt = (
                    float(depot_rows.iloc[0]['lat']),
                    float(depot_rows.iloc[0]['lon']),
                )

            try:
                van_paths, path_stats = build_van_paths_with_roads(
                    st.session_state.vrp_routes,
                    st.session_state.vrp_nodes,
                    depot_pt,
                    progress_cb=progress_cb,
                )
                st.session_state.road_paths = van_paths
                st.session_state.road_paths_error = None
                progress_bar.progress(1.0)
                status_text.text("Done.")
                if path_stats["fallback_legs"] > 0:
                    st.warning(
                        f"Road geometry fetched: {path_stats['road_legs']} legs via OSRM, "
                        f"{path_stats['fallback_legs']} legs fell back to straight lines."
                    )
                else:
                    st.success(
                        f"All {path_stats['road_legs']} legs fetched from OSRM successfully."
                    )
            except Exception as e:
                st.session_state.road_paths_error = str(e)
                st.error(f"Could not fetch road geometry: {e}")
        else:
            st.success("✅ Street-level road geometry is already cached for this solution.")

    st.markdown("### 📊 Fleet Dispatch & Inventory Summary")
    if st.session_state.solver_run and st.session_state.vrp_summary is not None \
            and not st.session_state.vrp_summary.empty:
        src = st.session_state.get("_vrp_distance_source", "unknown")
        st.caption(f"Distances below are **{src}**.")
        display_cols = [
            "Van ID", "Driver", "Assigned Stops", "Total Load (Cartons)",
            "Capacity (Cartons)", "Estimated Distance (km)", "Route Sequence"
        ]
        st.dataframe(
            st.session_state.vrp_summary[display_cols],
            use_container_width=True
        )
    else:
        st.info("Run the OR-Tools solver above to generate an optimized dispatch plan.")

    with st.expander("📋 View Full Fleet Registry & Capacities"):
        st.dataframe(st.session_state.fleet_df, use_container_width=True)

    with st.expander("📋 View Current Merchant Node Registry Table"):
        st.dataframe(st.session_state.merchants_df, use_container_width=True)

    # ---------- Cache inspector ----------
    with st.expander("🗄️ Inspect routing matrix cache"):
        _c = _read_routing_cache()
        if _c is None:
            st.info("Cache file does not exist yet.")
        else:
            _mat = _c.get("distance_m")
            st.markdown(
                f"- **File**: `{ROUTING_CACHE_PATH}`\n"
                f"- **Fingerprint**: `{_c.get('fingerprint', 'MISSING')}`\n"
                f"- **Source**: `{_c.get('source', 'unknown')}`\n"
                f"- **Shape**: `{_mat.shape if isinstance(_mat, np.ndarray) else 'invalid'}`\n"
                f"- **Created at**: `{_c.get('created_at', 'unknown')}`\n"
            )
            _cur_fp = _fingerprint_merchants(
                st.session_state.merchants_df.reset_index(drop=True)
            )
            if _c.get("fingerprint") == _cur_fp:
                st.success("✅ Matches current merchant set — cache will be reused.")
            else:
                st.warning("⚠️ Does not match current merchant set — will be invalidated on next solve.")

    # ---------- Static network map with rich hover ----------
    st.markdown("### 📍 Lagos Corridor Network Map")
    st.caption(
        "**Hover a merchant marker** to see its name, coordinates, and demand. "
        "**Hover anywhere else** on the map to read lat/lon."
    )

    map_data = _augment_for_tooltip(st.session_state.merchants_df)
    map_data['color'] = map_data['type'].apply(
        lambda t: [255, 69, 0, 200] if t == 'Depot' else [30, 144, 255, 180]
    )

    merchant_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius="size * 3",
        pickable=True,
        auto_highlight=True,
    )

    hover_layer = build_hover_layer(st.session_state.merchants_df)

    # Tooltip fields exist on BOTH layers so nothing renders as {undefined}.
    shared_tooltip = {
        "html": (
            "<b>{display_name}</b>"
            "<div>Lat: {lat_str}<br/>Lon: {lon_str}</div>"
            "<div>Demand: {demand_str} cartons</div>"
        ),
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    view_state = pdk.ViewState(
        latitude=float(map_data['lat'].mean()),
        longitude=float(map_data['lon'].mean()),
        zoom=14.0,
        pitch=30,
    )

    # Hover grid FIRST (bottom), merchants SECOND (top) so merchants win the pick.
    static_deck = pdk.Deck(
        layers=[hover_layer, merchant_layer],
        initial_view_state=view_state,
        tooltip=shared_tooltip,
        map_style=None,
    )
    st.pydeck_chart(static_deck, use_container_width=True)

    if st.session_state.solver_run and st.session_state.vrp_routes is not None \
            and st.session_state.road_paths is not None:
        st.markdown("---")
        st.markdown("### 🎬 Live Fleet Route Animation (Street-Following)")
        st.markdown(
            "Each van drives along its **actual OR-Tools optimized route**, "
            "**following real streets** via OSRM geometry, stopping at each "
            "merchant in sequence, then returning to the depot."
        )

        colA, colB, colC = st.columns(3)
        with colA:
            show_paths = st.checkbox("Show planned route lines", value=True)
        with colB:
            frames = st.slider("Frames per route", 40, 250, 120, 10)
        with colC:
            frame_delay = st.slider("Frame delay (sec)", 0.02, 0.3, 0.08, 0.01)

        if st.button("▶️ Play Street-Following Van Animation"):
            van_paths = st.session_state.road_paths
            nodes = st.session_state.vrp_nodes
            active_fleet = st.session_state.fleet_df[
                st.session_state.fleet_df['Status'] == 'Active'
            ].reset_index(drop=True)

            palette = [
                [0, 255, 127, 220],
                [255, 165, 0, 220],
                [255, 20, 147, 220],
                [0, 191, 255, 220],
                [255, 215, 0, 220],
                [180, 100, 255, 220],
            ]
            van_colors = [palette[i % len(palette)] for i in range(len(van_paths))]

            bg_data = _augment_for_tooltip(nodes)
            bg_data['color'] = bg_data['type'].apply(
                lambda t: [255, 69, 0, 200] if t == 'Depot' else [30, 144, 255, 160]
            )
            bg_layer = pdk.Layer(
                "ScatterplotLayer",
                data=bg_data,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius="size * 2.5",
                pickable=True,
            )

            path_records = []
            for v_idx, pts in enumerate(van_paths):
                if len(pts) < 2:
                    continue
                path_records.append({
                    "path": [[lo, la] for (la, lo) in pts],
                    "color": van_colors[v_idx],
                    "van": (
                        active_fleet.loc[v_idx, "Van ID"]
                        if v_idx < len(active_fleet)
                        else f"Van {v_idx+1}"
                    ),
                })

            path_layer = pdk.Layer(
                "PathLayer",
                data=path_records,
                get_path="path",
                get_color="color",
                get_width=3,
                width_min_pixels=2,
                pickable=True,
            )

            animation_placeholder = st.empty()
            status_placeholder = st.empty()

            van_stops = []
            for route in st.session_state.vrp_routes:
                stops = [nodes.iloc[i]['name'] for i in route]
                stops.append("Depot (returning)")
                van_stops.append(stops)

            for frame in range(frames + 1):
                progress = frame / frames
                anim_rows = []

                for v_idx, polyline in enumerate(van_paths):
                    pt = interpolate_along_polyline(polyline, progress)
                    if pt is None:
                        continue
                    stops_list = van_stops[v_idx]
                    target_idx = min(
                        int(progress * len(stops_list)),
                        len(stops_list) - 1
                    )
                    target_name = stops_list[target_idx] if stops_list else "Depot"

                    anim_rows.append({
                        "lat": pt[0],
                        "lon": pt[1],
                        "color": van_colors[v_idx],
                        "size": 10,
                        "van_name": (
                            active_fleet.loc[v_idx, "Van ID"]
                            if v_idx < len(active_fleet)
                            else f"Van {v_idx+1}"
                        ) + f" → {target_name}",
                        # Dummy fields so the shared tooltip resolves on van points too
                        "lat_str": f"{pt[0]:.5f}",
                        "lon_str": f"{pt[1]:.5f}",
                        "display_name": (
                            active_fleet.loc[v_idx, "Van ID"]
                            if v_idx < len(active_fleet)
                            else f"Van {v_idx+1}"
                        ),
                        "demand_str": "",
                    })

                if not anim_rows:
                    break

                anim_df = pd.DataFrame(anim_rows)
                van_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=anim_df,
                    get_position=["lon", "lat"],
                    get_color="color",
                    get_radius="size * 4",
                    pickable=True,
                )

                anim_hover = build_hover_layer(nodes)

                # Order: hover grid (bottom) → merchants → paths → vans (top)
                layers = [anim_hover, bg_layer]
                if show_paths and path_records:
                    layers.append(path_layer)
                layers.append(van_layer)

                anim_deck = pdk.Deck(
                    layers=layers,
                    initial_view_state=pdk.ViewState(
                        latitude=float(bg_data['lat'].mean()),
                        longitude=float(bg_data['lon'].mean()),
                        zoom=14.0,
                        pitch=30,
                    ),
                    tooltip={
                        "html": (
                            "<b>{display_name}</b>"
                            "<div>Lat: {lat_str}<br/>Lon: {lon_str}</div>"
                            "<div>Demand: {demand_str} cartons</div>"
                        ),
                        "style": {"backgroundColor": "black", "color": "white"},
                    },
                    map_style=None,
                )
                animation_placeholder.pydeck_chart(anim_deck, use_container_width=True)
                time.sleep(frame_delay)

            status_placeholder.success(
                "✅ All vans completed their street-following routes and returned to the depot."
            )

    elif st.session_state.solver_run and st.session_state.road_paths is None:
        st.info(
            "Road geometry has not yet been loaded. Scroll up to the "
            "'Fetching Street-Level Road Geometry' section — it should run automatically "
            "after solving, but you can also re-run the solver if it didn't."
        )


# ============================================================
# MAIN VIEW: SIMPY SIMULATION
# ============================================================
elif app_mode == "Operational Simulation (SimPy)":
    st.title("⚡ Lagos Operational Chaos Simulation (SimPy Engine)")
    st.markdown(
        "Simulates **your optimized VRP plan** under stochastic Lagos traffic, "
        "breakdowns, and merchant cash rejections. Also runs a **round-robin "
        "baseline** so you can see the value of optimization."
    )

    if not st.session_state.solver_run or not st.session_state.vrp_routes:
        st.warning(
            "⚠️ No VRP solution found. Go to **Overview & Fleet Routes** "
            "and click **Run OR-Tools Solver** first."
        )
        st.stop()

    nodes = st.session_state.vrp_nodes
    vrp_routes = st.session_state.vrp_routes
    dist_matrix_m = st.session_state.get("_dist_matrix_m")
    dist_source = st.session_state.get("_vrp_distance_source", "unknown")

    st.caption(f"Travel legs use the same matrix as the solver: **{dist_source}**.")

    with st.expander("⚙️ Simulation & Cost Parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            shift_min = st.number_input("Shift length (min)", 240, 900, 480, 30)
            speed = st.number_input("Avg driving speed (km/h)", 5.0, 60.0, 18.0, 1.0)
            peak_mult = st.slider("Peak-hour traffic multiplier", 1.0, 4.0, 2.4, 0.1)
        with c2:
            svc_mean = st.number_input("Service mean (min)", 1.0, 20.0, 7.0, 0.5)
            svc_std = st.number_input("Service std (min)", 0.0, 10.0, 2.0, 0.5)
            break_base = st.slider("Breakdown base prob / stop", 0.0, 0.3, 0.02, 0.01)
        with c3:
            rej_base = st.slider("Rejection base prob / stop", 0.0, 0.4, 0.05, 0.01)
            hourly_cost = st.number_input("Driver cost (₦/hr)", 500.0, 10000.0, 2500.0, 100.0)
            fuel_km = st.number_input("Fuel cost (₦/km)", 10.0, 1000.0, 120.0, 10.0)

    cfg = SimConfig(
        shift_minutes=shift_min,
        avg_speed_kmh=speed,
        peak_multiplier=peak_mult,
        service_mean_min=svc_mean,
        service_std_min=svc_std,
        breakdown_base_p=break_base,
        rejection_base_p=rej_base,
        hourly_driver_cost=hourly_cost,
        fuel_cost_per_km=fuel_km,
        on_time_sla_min=shift_min,
    )

    st.markdown("### 🚀 Run Simulation")
    cc1, cc2, cc3 = st.columns([1.2, 1, 1])
    run_single = cc1.button("▶️ Run Single Day (VRP vs Baseline)")
    run_mc = cc2.button("🎲 Run Monte Carlo (50 runs)")
    mc_target = cc3.selectbox("Monte Carlo plan", ["VRP Optimized", "Baseline Round-Robin"])

    if run_single:
        with st.spinner("Simulating optimized plan..."):
            opt = simulate_routes(
                vrp_routes, nodes, st.session_state.fleet_df, cfg,
                seed=42, dist_matrix_m=dist_matrix_m,
            )
        with st.spinner("Simulating round-robin baseline..."):
            baseline_routes = _assign_round_robin(nodes, st.session_state.fleet_df)
            base = simulate_routes(
                baseline_routes, nodes, st.session_state.fleet_df, cfg,
                seed=42, dist_matrix_m=dist_matrix_m,
            )

        o, b = opt["totals"], base["totals"]

        st.markdown("### 📊 Optimized vs Baseline")
        cmp_df = pd.DataFrame({
            "Metric": [
                "Deliveries completed",
                "Rejections",
                "Breakdowns",
                "Total distance (km)",
                "Longest shift (min)",
                "Success rate (%)",
                "Fuel cost (₦)",
                "Breakdown cost (₦)",
                "Rejection cost (₦)",
                "Driver cost (₦)",
                "Overtime cost (₦)",
                "**Total cost (₦)**",
            ],
            "VRP Optimized": [
                o["deliveries"], o["rejections"], o["breakdowns"],
                round(o["distance_km"], 1), round(o["shift_min"], 1),
                round(o["success_rate"], 1),
                f"₦{o['fuel_cost']:,.0f}", f"₦{o['breakdown_cost']:,.0f}",
                f"₦{o['rejection_cost']:,.0f}", f"₦{o['driver_cost']:,.0f}",
                f"₦{o['overtime_cost']:,.0f}", f"₦{o['total_cost']:,.0f}",
            ],
            "Baseline Round-Robin": [
                b["deliveries"], b["rejections"], b["breakdowns"],
                round(b["distance_km"], 1), round(b["shift_min"], 1),
                round(b["success_rate"], 1),
                f"₦{b['fuel_cost']:,.0f}", f"₦{b['breakdown_cost']:,.0f}",
                f"₦{b['rejection_cost']:,.0f}", f"₦{b['driver_cost']:,.0f}",
                f"₦{b['overtime_cost']:,.0f}", f"₦{b['total_cost']:,.0f}",
            ],
        })
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        dcost = b["total_cost"] - o["total_cost"]
        ddist = b["distance_km"] - o["distance_km"]
        dshift = b["shift_min"] - o["shift_min"]
        d_deliv = o["deliveries"] - b["deliveries"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cost saved by optimization",
                  f"₦{dcost:,.0f}",
                  f"{'✅' if dcost > 0 else '❌'} vs baseline")
        m2.metric("Distance saved", f"{ddist:.1f} km",
                  f"{'✅' if ddist > 0 else '❌'}")
        m3.metric("Shift time saved", f"{dshift:.1f} min",
                  f"{'✅' if dshift > 0 else '❌'}")
        m4.metric("Extra deliveries", f"{d_deliv:+d}",
                  "VRP vs baseline")

        st.markdown("### 🗓️ Delivery Timelines")

        def render_gantt(df, title):
            if df.empty:
                st.info(f"{title}: no events recorded.")
                return
            g = df.copy()
            g["Start_dt"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(g["Start"], unit="m")
            g["Finish_dt"] = g["Start_dt"] + pd.to_timedelta(g["Duration"], unit="m")
            color_map = {
                "Travel": "#4A90E2",
                "Service": "#27AE60",
                "Rejection": "#E67E22",
                "Breakdown": "#C0392B",
            }
            fig = px.timeline(
                g, x_start="Start_dt", x_end="Finish_dt", y="Van",
                color="Type", color_discrete_map=color_map,
                hover_data={"Task": True, "Detail": True,
                            "Start_dt": False, "Finish_dt": False,
                            "Duration": ":.1f"},
                title=title,
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=max(220, 90 * g["Van"].nunique()),
                              xaxis_title="Minutes into shift",
                              legend_title="")
            st.plotly_chart(fig, use_container_width=True)

        tab_vrp, tab_base = st.tabs(["🟢 VRP Optimized", "🔴 Baseline Round-Robin"])
        with tab_vrp:
            render_gantt(opt["gantt_df"], "VRP Optimized — Per-Van Timeline")
            st.markdown("**Per-van breakdown:**")
            st.dataframe(opt["per_van"], use_container_width=True, hide_index=True)
        with tab_base:
            render_gantt(base["gantt_df"], "Baseline Round-Robin — Per-Van Timeline")
            st.markdown("**Per-van breakdown:**")
            st.dataframe(base["per_van"], use_container_width=True, hide_index=True)

    if run_mc:
        plan_label = "VRP Optimized" if mc_target == "VRP Optimized" else "Baseline Round-Robin"
        routes_to_use = (
            vrp_routes if mc_target == "VRP Optimized"
            else _assign_round_robin(nodes, st.session_state.fleet_df)
        )
        with st.spinner(f"Running 50 Monte-Carlo iterations on {plan_label}..."):
            mc_summary, mc_df = monte_carlo_simulation(
                routes_to_use, nodes, st.session_state.fleet_df, cfg,
                n_runs=50, dist_matrix_m=dist_matrix_m,
            )

        st.markdown(f"### 🎲 Monte Carlo Distribution — {plan_label}")
        g1, g2, g3 = st.columns(3)
        g1.metric("Median deliveries", f"{mc_summary['median_deliveries']:.0f}")
        g2.metric("Median total cost", f"₦{mc_summary['median_cost']:,.0f}")
        g3.metric("Mean success rate", f"{mc_summary['mean_success_rate']:.1f}%")

        g4, g5, g6 = st.columns(3)
        g4.metric("P(≥1 delivery missed)", f"{mc_summary['p_missed_delivery']:.0f}%")
        g5.metric("P(≥1 breakdown)", f"{mc_summary['p_breakdown']:.0f}%")
        g6.metric("P(overtime)", f"{mc_summary['p_overtime']:.0f}%")

        st.caption(f"P95 shift length across 50 runs: **{mc_summary['p95_shift_min']:.1f} min**")

        h1, h2 = st.columns(2)
        with h1:
            fig_d = px.histogram(mc_df, x="deliveries", nbins=12,
                                 title="Distribution: Deliveries Completed")
            st.plotly_chart(fig_d, use_container_width=True)
        with h2:
            fig_c = px.histogram(mc_df, x="total_cost", nbins=12,
                                 title="Distribution: Total Cost (₦)")
            st.plotly_chart(fig_c, use_container_width=True)

        fig_s = px.histogram(mc_df, x="shift_min", nbins=15,
                             title="Distribution: Longest Van Shift (min)")
        fig_s.add_vline(x=cfg.shift_minutes, line_dash="dash",
                        line_color="red", annotation_text="SLA")
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Active Dispatch Plan (from VRP Solver)")
    if st.session_state.vrp_summary is not None:
        st.dataframe(st.session_state.vrp_summary, use_container_width=True)

    st.caption(
        "⚠️ Breakdown and rejection probabilities are illustrative "
        "assumptions. Replace with historical Lagos data for production use."
    )


# ============================================================
# MAIN VIEW: XGBOOST RISK PREDICTOR
# ============================================================
elif app_mode == "AI Risk Predictor (XGBoost)":
    st.title("🤖 Proactive Delivery Failure Risk Predictor")
    st.markdown(
        "Input route and environmental parameters to score delivery failure "
        "probability using our trained XGBoost model."
    )

    col1, col2 = st.columns(2)
    with col1:
        traffic_mult = st.slider("Traffic Congestion Multiplier", 1.0, 3.5, 1.8, 0.1)
        distance = st.slider("Distance from Depot (km)", 1.0, 20.0, 6.5, 0.5)
        demand = st.slider("Order Size (Cartons)", 2, 15, 6)
    with col2:
        route_pos = st.slider("Route Sequence Position", 1, 10, 3)
        is_peak = st.selectbox(
            "Time Window",
            ["Off-Peak Hours", "Peak Gridlock Hours (7-10 AM / 4-7 PM)"]
        )

    peak_val = 1 if "Peak" in is_peak else 0

    if st.button("🔮 Predict Failure Risk"):
        if "model" in assets:
            model = assets["model"]
            features = pd.DataFrame([{
                'traffic_multiplier': traffic_mult,
                'distance_to_depot': distance,
                'demand_cartons': demand,
                'route_position': route_pos,
                'is_peak_hour': peak_val
            }])
            prob = model.predict_proba(features)[0][1] * 100

            st.markdown("---")
            if prob > 50:
                st.error(
                    f"⚠️ **High Delivery Risk Detected:** {prob:.1f}% failure probability. "
                    "Recommendation: Defer delivery or add buffer time."
                )
            elif prob > 25:
                st.warning(
                    f"⚡ **Moderate Risk:** {prob:.1f}% failure probability. "
                    "Monitor driver closely."
                )
            else:
                st.success(
                    f"✅ **Low Risk:** {prob:.1f}% failure probability. Safe to dispatch."
                )
        else:
            heuristic_prob = min(
                95.0, (traffic_mult * 10) + (distance * 1.5) + (peak_val * 20)
            )
            st.warning(f"⚠️ Estimated Failure Risk: **{heuristic_prob:.1f}%**.")
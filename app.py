import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import random
import simpy
import plotly.express as px
import pydeck as pdk
import xgboost as xgb

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Lagos FMCG Distribution Optimizer",
    page_icon="🚚",
    layout="wide"
)

# --- AUTO-GENERATE MODEL IF CORRUPTED/MISSING ---
def ensure_model_exists():
    os.makedirs("data/models", exist_ok=True)
    model_path = "data/models/risk_xgb_model.pkl"
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 100:
        np.random.seed(42)
        n = 1000
        X = pd.DataFrame({
            'traffic_multiplier': np.random.uniform(1, 3, n),
            'distance_to_depot': np.random.uniform(1, 15, n),
            'demand_cartons': np.random.randint(2, 12, n),
            'route_position': np.random.randint(1, 8, n),
            'is_peak_hour': np.random.choice([0, 1], n)
        })
        y = (X['traffic_multiplier'] * 0.4 + X['is_peak_hour'] * 0.5 > 1.5).astype(int)
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3)
        model.fit(X, y)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

ensure_model_exists()

# --- CACHED DATA & MODEL LOADERS ---
@st.cache_resource
def load_assets():
    assets = {}
    routing_path = "data/processed/routing_matrices.pkl"
    if os.path.exists(routing_path):
        try:
            with open(routing_path, "rb") as f:
                assets["routing"] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load routing matrices: {e}")

    model_path = "data/models/risk_xgb_model.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                assets["model"] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load XGBoost model: {e}")

    return assets

assets = load_assets()

# --- INITIALIZE SESSION STATE FOR MERCHANTS & FLEET ---
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


# --- VRP SOLVER (OR-TOOLS) ---
def haversine_km(a, b):
    R = 6371.0
    lat1, lon1 = np.radians(a['lat']), np.radians(a['lon'])
    lat2, lon2 = np.radians(b['lat']), np.radians(b['lon'])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))


def solve_vrp(merchants_df, fleet_df):
    """Solve a Capacitated VRP over the current merchant + fleet data.
    Returns:
        routes: list[list[int]] — one ordered list of node indices per van (excluding depot)
        nodes: DataFrame used for solving
        summary: DataFrame with per-van metrics
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    nodes = merchants_df.reset_index(drop=True)
    n = len(nodes)

    # Distance matrix (meters, integer required by OR-Tools)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = haversine_km(nodes.iloc[i], nodes.iloc[j])
    dist_int = (dist * 1000).astype(int)

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

    # Capacity dimension
    demands = nodes['demand'].astype(int).tolist()

    def demand_cb(from_idx):
        return demands[manager.IndexToNode(from_idx)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    caps = active['Capacity (Cartons)'].astype(int).tolist()
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, caps, True, "Capacity"
    )

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
            # close loop back to depot
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
        # Fallback: round-robin assignment so animation still works
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


def build_van_paths(routes, nodes):
    """Convert VRP routes into list of (lat, lon) waypoints per van,
    starting and ending at the depot."""
    depot_rows = nodes[nodes['type'] == 'Depot']
    if depot_rows.empty:
        depot_pt = (float(nodes.iloc[0]['lat']), float(nodes.iloc[0]['lon']))
    else:
        depot_pt = (float(depot_rows.iloc[0]['lat']),
                    float(depot_rows.iloc[0]['lon']))

    van_paths = []
    for route in routes:
        pts = [depot_pt]
        for node_i in route:
            pts.append((float(nodes.iloc[node_i]['lat']),
                        float(nodes.iloc[node_i]['lon'])))
        pts.append(depot_pt)  # return to depot
        van_paths.append(pts)
    return van_paths, depot_pt


# --- HOVER-COORDINATE LAYER HELPER ---
def build_hover_layer(merchants_df, pad=0.02, resolution=60):
    """Transparent pickable grid that reports lat/lon on hover."""
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

    return pdk.Layer(
        "ScatterplotLayer",
        data=coord_grid,
        get_position=["lon", "lat"],
        get_radius=18,
        get_fill_color=[0, 0, 0, 0],
        pickable=True,
        auto_highlight=False,
    )


# --- REAL SIMPY DISCRETE-EVENT SIMULATION ENGINE ---
def run_simpy_simulation(merchants_df, fleet_df):
    env = simpy.Environment()

    stats = {
        "completed": 0,
        "failed": 0,
        "breakdowns": 0,
        "traffic_delay": 0.0,
        "service_delay": 0.0,
        "rejection_delay": 0.0,
        "breakdown_delay": 0.0
    }

    active_vans = fleet_df[fleet_df['Status'] == 'Active']
    merchant_stops = merchants_df[merchants_df['type'] == 'Merchant'].to_dict('records')

    def van_route_process(env, van_id, stops):
        for stop in stops:
            travel_time = float(np.random.gamma(2, 3.5))
            stats["traffic_delay"] += travel_time
            yield env.timeout(travel_time)

            if random.random() < 0.06:
                breakdown_time = float(np.random.uniform(10, 25))
                stats["breakdowns"] += 1
                stats["breakdown_delay"] += breakdown_time
                yield env.timeout(breakdown_time)

            if random.random() < 0.12:
                rejection_time = float(np.random.uniform(5, 15))
                stats["failed"] += 1
                stats["rejection_delay"] += rejection_time
                yield env.timeout(rejection_time)
                continue

            service_time = float(np.random.normal(7, 2))
            actual_service = max(2.0, service_time)
            stats["service_delay"] += actual_service
            stats["completed"] += 1
            yield env.timeout(actual_service)

    if not active_vans.empty and merchant_stops:
        num_vans = len(active_vans)
        for i, van in active_vans.iterrows():
            van_name = van['Van ID']
            assigned_stops = merchant_stops[i::num_vans]
            if assigned_stops:
                env.process(van_route_process(env, van_name, assigned_stops))

        env.run(until=480)

    return stats


# --- SIDEBAR CONTROLS ---
st.sidebar.title("🚚 Control Panel")
st.sidebar.markdown("**Yaba / Ikeja FMCG Corridor**")
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
                    st.success(f"Added {new_name}!")
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
                        st.success(f"Updated {selected_m}!")
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
                    st.success(f"Deleted {del_m}!")
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
                    st.success(f"Deleted {del_van}!")
                    st.rerun()
            else:
                st.sidebar.warning("Cannot delete the last remaining van.")

st.sidebar.markdown("---")
st.sidebar.info(
    "Operational resilience engine accounting for Lagos traffic multipliers, "
    "vehicle breakdowns, and merchant rejections."
)

# --- MAIN APP VIEWS ---
if app_mode == "Overview & Fleet Routes":
    st.title("🗺️ Yaba Corridor Fleet Routing & Network")
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

    # --- SOLVER RUN ---
    st.markdown("### ⚙️ VRP Optimization Engine")
    if st.button("🚀 Run OR-Tools Solver for Current Network"):
        with st.spinner("Running Google OR-Tools CVRP optimization..."):
            try:
                routes, nodes, summary = solve_vrp(
                    st.session_state.merchants_df, st.session_state.fleet_df
                )
                st.session_state.vrp_routes = routes
                st.session_state.vrp_nodes = nodes
                st.session_state.vrp_summary = summary
                st.session_state.solver_run = True
                if len(routes) > 0:
                    st.success("CVRP Solver executed successfully! Routes re-optimized across active vans.")
                else:
                    st.warning("Solver returned no routes — check that at least one van is Active and merchants exist.")
            except ImportError:
                st.error(
                    "OR-Tools is not installed. Install it with: "
                    "`pip install ortools`"
                )
            except Exception as e:
                st.error(f"Solver failed: {e}")

    # --- DISPATCH TABLE ---
    st.markdown("### 📊 Fleet Dispatch & Inventory Summary")
    if st.session_state.solver_run and st.session_state.vrp_summary is not None \
            and not st.session_state.vrp_summary.empty:
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

    # --- STATIC NETWORK MAP WITH HOVER COORDS ---
    st.markdown("### 📍 Yaba Corridor Network Map")
    st.caption("Hover anywhere on the map to read latitude / longitude in the tooltip.")

    map_data = st.session_state.merchants_df.copy()
    map_data['color'] = map_data['type'].apply(
        lambda t: [255, 69, 0, 200] if t == 'Depot' else [30, 144, 255, 180]
    )
    map_data['coord_str'] = map_data.apply(
        lambda r: f"({r['lat']:.5f}, {r['lon']:.5f})", axis=1
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

    view_state = pdk.ViewState(
        latitude=float(map_data['lat'].mean()),
        longitude=float(map_data['lon'].mean()),
        zoom=14.0,
        pitch=30,
    )

    # Hover layer added LAST so it's on top and intercepts hover events
    static_deck = pdk.Deck(
        layers=[merchant_layer, hover_layer],
        initial_view_state=view_state,
        tooltip={
            "html": (
                "<b>Lat:</b> {lat_str}<br/>"
                "<b>Lon:</b> {lon_str}"
            ),
            "style": {"backgroundColor": "steelblue", "color": "white"}
        },
        map_style=None,
    )
    st.pydeck_chart(static_deck, use_container_width=True)

    # --- LIVE ANIMATION FOLLOWING VRP ROUTE ---
    if st.session_state.solver_run and st.session_state.vrp_routes is not None:
        st.markdown("---")
        st.markdown("### 🎬 Live Fleet Route Animation")
        st.markdown(
            "Each van follows its **actual OR-Tools optimized route**, "
            "stopping at each merchant **in sequence**, then returning to the depot."
        )

        show_paths = st.checkbox("Show planned route lines", value=True)
        speed = st.slider("Animation speed (frames per leg)", 8, 40, 20)
        frame_delay = st.slider("Frame delay (seconds)", 0.05, 0.5, 0.15, 0.05)

        if st.button("▶️ Play Van Transit Simulation"):
            van_paths, depot_pt = build_van_paths(
                st.session_state.vrp_routes, st.session_state.vrp_nodes
            )
            nodes = st.session_state.vrp_nodes
            active_fleet = st.session_state.fleet_df[
                st.session_state.fleet_df['Status'] == 'Active'
            ].reset_index(drop=True)

            # Assign colors per van
            palette = [
                [0, 255, 127, 220],
                [255, 165, 0, 220],
                [255, 20, 147, 220],
                [0, 191, 255, 220],
                [255, 215, 0, 220],
                [180, 100, 255, 220],
            ]
            van_colors = [palette[i % len(palette)] for i in range(len(van_paths))]

            # Draw planned route paths (one PathLayer record per van)
            path_records = []
            for v_idx, pts in enumerate(van_paths):
                if len(pts) < 2:
                    continue
                path_records.append({
                    "path": [[lo, la] for (la, lo) in pts],
                    "color": van_colors[v_idx],
                    "van": active_fleet.loc[v_idx, "Van ID"] if v_idx < len(active_fleet) else f"Van {v_idx+1}"
                })

            # Static background layers (merchants + depot)
            bg_data = nodes.copy()
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

            animation_placeholder = st.empty()
            status_placeholder = st.empty()

            max_legs = max((len(p) - 1) for p in van_paths) if van_paths else 0

            for leg in range(max_legs):
                for f in range(speed + 1):
                    t = f / speed
                    anim_rows = []

                    for v_idx, pts in enumerate(van_paths):
                        if leg + 1 >= len(pts):
                            # Van already finished — park it at the depot
                            anim_rows.append({
                                "lat": depot_pt[0],
                                "lon": depot_pt[1],
                                "color": [120, 120, 120, 150],
                                "size": 6,
                                "van_name": (
                                    active_fleet.loc[v_idx, "Van ID"]
                                    if v_idx < len(active_fleet)
                                    else f"Van {v_idx+1}"
                                ) + " (Done)",
                            })
                            continue

                        (la1, lo1) = pts[leg]
                        (la2, lo2) = pts[leg + 1]
                        cur_lat = la1 + (la2 - la1) * t
                        cur_lon = lo1 + (lo2 - lo1) * t

                        # Figure out which node we're heading to
                        target_node_idx = st.session_state.vrp_routes[v_idx][leg] \
                            if leg < len(st.session_state.vrp_routes[v_idx]) else None
                        if target_node_idx is not None:
                            target_name = nodes.iloc[target_node_idx]['name']
                        else:
                            target_name = "Depot (returning)"

                        anim_rows.append({
                            "lat": cur_lat,
                            "lon": cur_lon,
                            "color": van_colors[v_idx],
                            "size": 10,
                            "van_name": (
                                active_fleet.loc[v_idx, "Van ID"]
                                if v_idx < len(active_fleet)
                                else f"Van {v_idx+1}"
                            ) + f" → {target_name}",
                        })

                    anim_df = pd.DataFrame(anim_rows)

                    van_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=anim_df,
                        get_position=["lon", "lat"],
                        get_color="color",
                        get_radius="size * 4",
                        pickable=True,
                    )

                    # Invisible hover layer on top
                    anim_hover = build_hover_layer(nodes)

                    layers = [bg_layer]
                    if show_paths and path_records:
                        layers.append(pdk.Layer(
                            "PathLayer",
                            data=path_records,
                            get_path="path",
                            get_color="color",
                            get_width=3,
                            width_min_pixels=2,
                            pickable=True,
                        ))
                    layers.append(van_layer)
                    layers.append(anim_hover)

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
                                "<b>{van_name}</b><br/>"
                                "Lat: {lat}<br/>"
                                "Lon: {lon}"
                            ),
                            "style": {"backgroundColor": "black", "color": "white"}
                        },
                        map_style=None,
                    )
                    animation_placeholder.pydeck_chart(anim_deck, use_container_width=True)
                    time.sleep(frame_delay)

            status_placeholder.success(
                "✅ All vans have completed their OR-Tools optimized routes and returned to the depot."
            )


elif app_mode == "Operational Simulation (SimPy)":
    st.title("⚡ Lagos Operational Chaos Simulation (SimPy Engine)")
    st.markdown(
        "Executing a live discrete-event simulation modeling stochastic traffic "
        "gridlocks, van breakdowns, and merchant cash rejections."
    )

    if st.button("🚀 Run Live SimPy Simulation Experiment"):
        with st.spinner("Running discrete-event simulation..."):
            sim_stats = run_simpy_simulation(
                st.session_state.merchants_df, st.session_state.fleet_df
            )
            time.sleep(0.5)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Completed Deliveries", f"{sim_stats['completed']} Stops", "Fully Processed")
            col2.metric("Failed / Rejected", f"{sim_stats['failed']} Stops", "Cash-flow Delay")
            col3.metric("Vehicle Breakdowns", f"{sim_stats['breakdowns']} Incidents", "Fleet Maintenance")
            total_delay = (
                sim_stats['traffic_delay'] + sim_stats['service_delay']
                + sim_stats['rejection_delay'] + sim_stats['breakdown_delay']
            )
            col4.metric("Total Network Delay", f"{total_delay:.1f} mins", "Stochastic Impact")

            st.success("SimPy discrete-event simulation completed successfully!")

            chart_data = pd.DataFrame({
                "Disruption Factor": [
                    "Traffic Congestion", "Merchant Service Time",
                    "Rejection Handling", "Vehicle Breakdowns"
                ],
                "Delay Impact (Mins)": [
                    round(sim_stats['traffic_delay'], 1),
                    round(sim_stats['service_delay'], 1),
                    round(sim_stats['rejection_delay'], 1),
                    round(sim_stats['breakdown_delay'], 1),
                ]
            })
            fig = px.bar(
                chart_data, x="Disruption Factor", y="Delay Impact (Mins)",
                color="Disruption Factor",
                title="SimPy Delay Contribution Breakdown (Minutes)"
            )
            st.plotly_chart(fig, use_container_width=True)
   
import os
import pickle
import numpy as np
import networkx as nx
import osmnx as ox
from shapely.geometry import Point

def prepare_routing_graph(G):
    """
    Imputes speed limits and calculates travel times for all road segments,
    handling both real OSM graphs and offline synthetic grids gracefully.
    """
    print("Preparing routing graph travel times...")
    try:
        # Attempt standard OSMnx imputation (works for real OSM data)
        G = ox.routing.add_edge_speeds(G)
        G = ox.routing.add_edge_travel_times(G)
    except Exception as e:
        print(f"Notice: Using fallback speed imputation for synthetic grid ({e}).")
        # Fallback for synthetic graphs where edges already have attributes or need defaults
        for u, v, data in G.edges(data=True):
            if 'length' not in data:
                data['length'] = 300.0  # default 300 meters per edge
            if 'travel_time' not in data:
                speed_kph = data.get('speed_kph', 30.0)
                data['travel_time'] = (data['length'] / 1000.0) / speed_kph * 3600.0
    return G

def compute_distance_time_matrices(G, merchants_gdf, depot_coords=(6.5100, 3.3790)):
    """
    Computes N x N distance (meters) and duration (seconds) matrices between 
    the depot and all merchants.
    """
    # 1. Prepare graph with travel times
    G = prepare_routing_graph(G)
    
    # 2. Extract merchant coordinates and find their nearest graph nodes
    merchant_nodes = []
    for geom in merchants_gdf.geometry:
        pt = geom.centroid if geom.geom_type == 'Polygon' else geom
        node = ox.distance.nearest_nodes(G, X=pt.x, Y=pt.y)
        merchant_nodes.append(node)
        
    # 3. Find nearest node for the Depot (Warehouse)
    depot_node = ox.distance.nearest_nodes(G, X=depot_coords[1], Y=depot_coords[0])
    
    # Combine depot (index 0) and all merchants into a single list of nodes
    all_nodes = [depot_node] + merchant_nodes
    n_locations = len(all_nodes)
    
    print(f"Computing {n_locations} x {n_locations} All-Pairs Shortest Path matrices...")
    
    # Initialize matrices
    dist_matrix = np.zeros((n_locations, n_locations))
    time_matrix = np.zeros((n_locations, n_locations))
    
    # Compute shortest path lengths using Dijkstra's algorithm via NetworkX
    all_lengths_dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='length'))
    all_lengths_time = dict(nx.all_pairs_dijkstra_path_length(G, weight='travel_time'))
    
    for i, u in enumerate(all_nodes):
        for j, v in enumerate(all_nodes):
            if u in all_lengths_dist and v in all_lengths_dist[u]:
                dist_matrix[i, j] = all_lengths_dist[u][v]
                time_matrix[i, j] = all_lengths_time[u][v]
            else:
                dist_matrix[i, j] = np.inf
                time_matrix[i, j] = np.inf
                
    print("Matrices computed successfully!")
    print(f" - Distance Matrix Shape: {dist_matrix.shape}")
    print(f" - Duration Matrix Shape: {time_matrix.shape}")
    
    # Package output data structure
    routing_data = {
        "distance_matrix": dist_matrix,  # in meters
        "duration_matrix": time_matrix,  # in seconds
        "nodes": all_nodes,
        "depot_node": depot_node
    }
    
    # Save processed matrices for Step 1.3 (VRP Solver)
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/routing_matrices.pkl", "wb") as f:
        pickle.dump(routing_data, f)
    print("Saved routing matrices to data/processed/routing_matrices.pkl")
    
    return dist_matrix, time_matrix, all_nodes

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data_gen.osm_fetcher import get_yaba_network, extract_merchant_pois
    
    graph = get_yaba_network()
    merchants = extract_merchant_pois(graph, n_merchants=30)
    dist_mat, time_mat, nodes = compute_distance_time_matrices(graph, merchants)
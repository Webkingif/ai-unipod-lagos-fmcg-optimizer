import os
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import numpy as np
import networkx as nx

# Configure OSMnx settings & use a reliable public mirror endpoint
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.timeout = 60
ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api/interpreter"

def get_yaba_network():
    """
    Downloads the drivable street network graph for Yaba via mirror. 
    Falls back to building a local synthetic grid if the network is unreachable.
    """
    west, south, east, north = 3.3650, 6.4950, 3.3900, 6.5250
    bbox = (west, south, east, north)
    
    print(f"Attempting download from Overpass mirror for Yaba corridor...")
    try:
        G = ox.graph_from_bbox(bbox, network_type="drive")
        print(f"Graph downloaded successfully! Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
        return G
    except Exception as e:
        print(f"\n[Notice] Overpass API unreachable ({e}). Switching to Offline Synthetic Yaba Grid...")
        return generate_synthetic_yaba_grid()

def generate_synthetic_yaba_grid():
    """
    Creates a realistic synthetic grid graph representing Yaba's road network locally.
    Ensures your code runs anywhere, anytime, without external internet dependencies.
    """
    # Create a grid graph (e.g., 8x8 street intersections)
    G = nx.grid_2d_graph(8, 8, create_using=nx.MultiDiGraph)
    
    # Assign realistic lat/lon coordinates to grid nodes centered around Yaba
    center_lat, center_lon = 6.5100, 3.3790
    lat_step, lon_step = 0.003, 0.003
    
    pos = {}
    for r, c in G.nodes():
        node_lat = center_lat + (r - 4) * lat_step
        node_lon = center_lon + (c - 4) * lon_step
        G.nodes[(r, c)]['y'] = node_lat
        G.nodes[(r, c)]['x'] = node_lon
        pos[(r, c)] = (node_lon, node_lat)
        
    # Reindex nodes with integer IDs for compatibility
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # Add required edge attributes for distance and speed
    for u, v, data in G.edges(data=True):
        data['length'] = 300.0  # meters per block
        data['speed_kph'] = 30.0
        data['travel_time'] = (data['length'] / 1000.0) / data['speed_kph'] * 3600.0
        
    print(f"Synthetic Yaba Grid generated! Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
    return G

def extract_merchant_pois(G, n_merchants=40):
    """
    Extracts or generates merchant nodes snapped directly to the road graph.
    """
    node_ids = list(G.nodes())
    selected_nodes = np.random.choice(node_ids, size=min(n_merchants, len(node_ids)), replace=False)
    
    points = [Point(G.nodes[node]['x'], G.nodes[node]['y']) for node in selected_nodes]
    gdf_pois = gpd.GeoDataFrame(
        {'osmid': selected_nodes, 'shop': ['convenience'] * len(selected_nodes)},
        geometry=points,
        crs="EPSG:4326"
    )
    print(f"Loaded {len(gdf_pois)} merchant distribution targets.")
    return gdf_pois

if __name__ == "__main__":
    graph = get_yaba_network()
    merchants = extract_merchant_pois(graph, n_merchants=40)
    
    # Quick plot test
    fig, ax = plt.subplots(figsize=(8, 6))
    x = [G_node['x'] for _, G_node in graph.nodes(data=True)]
    y = [G_node['y'] for _, G_node in graph.nodes(data=True)]
    ax.scatter(x, y, c='#cccccc', s=10, label='Road Nodes')
    
    m_x = [geom.x for geom in merchants.geometry]
    m_y = [geom.y for geom in merchants.geometry]
    ax.scatter(m_x, m_y, c='#ff3333', s=40, zorder=5, label='Merchants')
    
    plt.legend()
    plt.title("Lagos FMCG Distribution Network - Yaba (Offline-Resilient)")
    plt.show()
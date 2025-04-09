import pandas as pd

import os
from androguard.misc import AnalyzeAPK
import networkx as nx
from collections import deque
from androguard.core.bytecodes.apk import APK
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.core.analysis.analysis import Analysis
import matplotlib.pyplot as plt


# Load features
feature_df = pd.read_csv('/home/xiaowei_3machine/MalScan-master/Feature_Ranking.csv')
selected_features = feature_df['Feature'].tolist()
split = 400
selected_features = selected_features[:split]

# Separate different types of features
opcode_features = [f for f in selected_features if '_freq' in f]  # Opcode frequency
api_features = [f for f in selected_features if '_call' in f]    # API calls
permission_features = [f for f in selected_features if '_perm' in f]  # Permissions


# Extract complete call graph
def extract_call_graph(apk_path):
    apk = APK(apk_path)
    dvm = DalvikVMFormat(apk.get_dex())
    analysis = Analysis(dvm)
    analysis.create_xref()
    call_graph = analysis.get_call_graph()
    
    G = nx.DiGraph()
    for edge in call_graph.edges:
        caller, callee = edge
        # Handle caller
        if isinstance(caller, tuple):
            # Assume tuple format: (class_name, method_name, descriptor)
            caller_full_name = f"{caller[0]}->{caller[1]}{caller[2]}"
        else:
            caller_full_name = f"{caller.get_class_name()}->{caller.get_name()}{caller.get_descriptor()}"
        
        # Handle callee
        if isinstance(callee, tuple):
            # Assume tuple format: (class_name, method_name, descriptor)
            callee_full_name = f"{callee[0]}->{callee[1]}{callee[2]}"
        else:
            callee_full_name = f"{callee.get_class_name()}->{callee.get_name()}{callee.get_descriptor()}"
        
        G.add_edge(caller_full_name, callee_full_name)
    
    return G, apk, dvm, analysis


# Load sensitive API list
def load_sensitive_api_list(txt_path):
    with open(txt_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

# Get sensitive nodes
def get_sensitive_nodes(G, sensitive_api_list):
    return [node for node in G.nodes if node in sensitive_api_list]


# BFS simplify graph
'''
    Deprecated, not using this method
'''
def bfs_distance(G, start, max_hops):
    visited = set()
    queue = deque([(start, 0)])
    while queue:
        node, hops = queue.popleft()
        if hops > max_hops:
            break
        if node not in visited:
            visited.add(node)
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, hops + 1))
    return visited


def simplify_fcg_with_edges(G, sensitive_nodes, N=1):
    """
    Simplify the call graph, retaining sensitive API nodes and their direct callers and callees, optionally extending to N-hop neighborhood.
    
    Parameters:
        G (nx.DiGraph): The complete function call graph
        sensitive_nodes (list): List of sensitive API nodes
        N (int): Number of hops to extend, default is 1
    Returns:
        simplified_G (nx.DiGraph): Simplified call graph
    """
    nodes_to_keep = set()
    edges_to_keep = set()
    
    for sensitive_node in sensitive_nodes:
        # Retain sensitive node
        nodes_to_keep.add(sensitive_node)
        
        # Retain direct predecessors (callers)
        for pred in G.predecessors(sensitive_node):
            nodes_to_keep.add(pred)
            edges_to_keep.add((pred, sensitive_node))
        
        # Retain direct successors (callees)
        for succ in G.successors(sensitive_node):
            nodes_to_keep.add(succ)
            edges_to_keep.add((sensitive_node, succ))
        
        # Optional: Extend to N-hop neighborhood
        if N > 0:
            # N-hop neighborhood of predecessors
            queue = deque([(sensitive_node, 0)])
            visited = {sensitive_node}
            while queue:
                node, hops = queue.popleft()
                if hops >= N:
                    continue
                for pred in G.predecessors(node):
                    if pred not in visited:
                        nodes_to_keep.add(pred)
                        edges_to_keep.add((pred, node))
                        visited.add(pred)
                        queue.append((pred, hops + 1))
            
            # N-hop neighborhood of successors
            queue = deque([(sensitive_node, 0)])
            visited = {sensitive_node}
            while queue:
                node, hops = queue.popleft()
                if hops >= N:
                    continue
                for succ in G.successors(node):
                    if succ not in visited:
                        nodes_to_keep.add(succ)
                        edges_to_keep.add((node, succ))
                        visited.add(succ)
                        queue.append((succ, hops + 1))
    
    # Generate simplified graph
    simplified_G = G.subgraph(nodes_to_keep).copy()
    return simplified_G



def extract_node_features(simplified_G, apk, dvm, analysis, opcode_features, api_features, permission_features):
    # Extract permission features (global)
    permissions = apk.get_permissions()
    perm_vector = [1 if perm in permissions else 0 for perm in permission_features]

    # Initialize node features
    node_features = {}
    
    for node in simplified_G.nodes():
        # Find method object
        method = None
        for m in dvm.get_methods():
            if node == f"{m.get_class_name()}->{m.get_name()}{m.get_descriptor()}":
                method = m
                break
        
        # Opcode frequency features
        if method:
            opcodes = method.get_instructions()
            opcode_counts = {op: 0 for op in opcode_features}
            for inst in opcodes:
                op_name = inst.get_name()
                feature_name = f"{op_name}_freq"
                if feature_name in opcode_counts:
                    opcode_counts[feature_name] += 1
            opcode_vector = [opcode_counts[op] for op in opcode_features]
        else:
            opcode_vector = [0] * len(opcode_features)

        # API call features
        if method:
            method_analysis = analysis.get_method_analysis(method)
            xrefs = method_analysis.get_xref_to()
            called_apis = set()
            for xref in xrefs:
                # xref is a tuple (class, method, offset)
                _, called_method, _ = xref  # Extract method part
                called_apis.add(f"{called_method.get_class_name()}->{called_method.get_name()}{called_method.get_descriptor()}")
            api_vector = [1 if api in called_apis else 0 for api in api_features]
        else:
            api_vector = [0] * len(api_features)

        # Combine feature vectors
        node_features[node] = opcode_vector + api_vector
    
    return node_features, perm_vector



import dgl
import torch

def build_dgl_graph(simplified_G, node_features, perm_vector, label, perm_length=None):
    """
    Parameters:
    - simplified_G: Simplified function call graph (NetworkX graph object)
    - node_features: Dictionary of node features, key is node, value is feature vector
    - perm_vector: Permission features as global variable
    - label: Graph-level family label (integer)
    - perm_length: Expected length of permission features (optional, for padding empty values)
    Returns:
    - g: DGL graph object containing graph structure and node features
    - graph_labels: Dictionary containing graph-level labels
    """
    # Check for empty graph
    if not simplified_G.nodes():
        print("Warning: No nodes in the graph.")
        return None, None

    # Convert directly from NetworkX graph to DGL graph
    g = dgl.from_networkx(simplified_G)

    # Process node features
    if node_features:
        node_feature_matrix = []
        # Get feature dimension (assuming node_features is not empty)
        feature_dim = len(next(iter(node_features.values())))
        for node in simplified_G.nodes():
            if node in node_features:
                feat = node_features[node]
                if len(feat) != feature_dim:
                    print(f"Error: Inconsistent feature length for node {node}: {len(feat)} vs {feature_dim}")
                    return None, None
                node_feature_matrix.append(feat)
            else:
                print(f"Warning: Node {node} missing in node_features, using zeros")
                node_feature_matrix.append([0] * feature_dim)
        g.ndata['features'] = torch.tensor(node_feature_matrix, dtype=torch.float)

    # Process perm_vector
    if not perm_vector:
        if perm_length is None:
            raise ValueError("perm_vector is empty and perm_length not provided")
        print("Warning: perm_vector is empty, using default zeros")
        perm_vector = [0] * perm_length

    # Create graph-level labels
    graph_labels = {
        'perm': torch.tensor([perm_vector], dtype=torch.float),
        'label': torch.tensor([label], dtype=torch.long)
    }

    return g, graph_labels


def test():
    # Parameter settings
    sensitive_api_txt = '/home/xiaowei_3machine/MalScan-master/Data/sensitive_apis.txt'  # Sensitive API list path
    N = 3  # BFS hop threshold
    
    output_dir = '/home/xiaowei_3machine/MalScan-master/Data'
    os.makedirs(output_dir, exist_ok=True)

    apk_path = '/storage/xiaowei_data/family_data/adpush/1E8192AEB59CF60C40B07B818552E8DA48E6C6BDB17E5FEB5B00694A7604D5EB.apk'
    
    # Load sensitive API
    print("Loading sensitive API list...")
    sensitive_api_list = load_sensitive_api_list(sensitive_api_txt)

    G, apk, dvm, analysis = extract_call_graph(apk_path)
    print("Identifying sensitive API nodes...")
    sensitive_nodes = get_sensitive_nodes(G, sensitive_api_list)
    
    print("Simplifying function call graph...")
    simplified_G = simplify_fcg_with_edges(G, sensitive_nodes, N)
    print(f"Simplified call graph - Nodes: {simplified_G.number_of_nodes()}, Edges: {simplified_G.number_of_edges()}")

    # Check if the simplified graph is empty
    if not simplified_G.nodes():
        print(f"Warning: Simplified graph for APK 0A7B9367F2B4ECEEC1B1D92A90D6FF16174AD2B54DA8570EBE1EB31F5E34CE61 has no nodes, skipping")
    
    # Extract features
    node_features, perm_vector = extract_node_features(
        simplified_G, apk, dvm, analysis, 
        opcode_features, api_features, permission_features
    )

    # Build DGL graph
    g, graph_labels = build_dgl_graph(simplified_G, node_features, perm_vector, 1)
    output_file = '/home/xiaowei_3machine/MalScan-master/Data/test_graph.dgl'
    if g is not None and graph_labels is not None:
        dgl.save_graphs(output_file, [g], graph_labels)
        print(f"Graph saved to {output_file}")
    
    
def inspect_dgl_graph(file_path):
    """
    Load and inspect the structure and data of a DGL graph
    :param file_path: Path to the .dgl file
    """
    try:
        # Load graph
        graphs, labels = dgl.load_graphs(file_path)
        if not graphs:
            print("Error: No graphs loaded")
            return
        
        # Assume each file contains only one graph
        g = graphs[0]
        
        # 1. Inspect basic graph structure
        print("\n=== Graph Structure ===")
        print(f"Number of nodes: {g.num_nodes()}")
        print(f"Number of edges: {g.num_edges()}")
        print(f"In-degree stats: min={g.in_degrees().min().item()}, max={g.in_degrees().max().item()}")
        print(f"Out-degree stats: min={g.out_degrees().min().item()}, max={g.out_degrees().max().item()}")

        # 2. Inspect node features
        print("\n=== Node Features (g.ndata) ===")
        if 'features' in g.ndata:
            features = g.ndata['features']
            print(f"Node feature shape: {features.shape}")
            print(f"Feature type: {features.dtype}")
            print(f"Sample feature (first node): {features[0]}")
        else:
            print("No node features 'features' found")


        # 3. Inspect graph-level data
        print("\n=== Graph-Level Data (graph_data) ===")
        if labels:
            for key, value in labels.items():
                print(f"{key}: {value}")
                print(f"  Shape: {value.shape}, Type: {value.dtype}")
        else:
            print("No graph-level data found")

        # 4. Visualize graph structure (optional, suitable for small graphs)
        visualize_graph(g)

    except Exception as e:
        print(f"Error loading or processing {file_path}: {str(e)}")


def visualize_graph(g):
    """
    Convert DGL graph to NetworkX graph and visualize it
    :param g: DGL graph object
    """
    try:
        # Convert to NetworkX graph
        nx_graph = g.to_networkx()
        
        # Draw the graph
        plt.figure(figsize=(8, 6))
        nx.draw(nx_graph, with_labels=False, node_size=50, node_color='skyblue', edge_color='gray')
        plt.title("Graph Visualization")
        plt.show()
    except Exception as e:
        print(f"Visualization error: {str(e)}")


if __name__ == "__main__":
    # test()
    # inspect_dgl_graph("/home/xiaowei_3machine/MalScan-master/Data/test_graph.dgl")
    
    # Parameter settings
    sensitive_api_txt = '/home/xiaowei_3machine/MalScan-master/Data/sensitive_apis.txt'  # Sensitive API list path
    N = 3  # BFS hop threshold
    
    output_dir = '/storage/xiaowei_data/DWFS-Obfuscation'
    os.makedirs(output_dir, exist_ok=True)

    # Load sensitive API
    print("Loading sensitive API list...")
    sensitive_api_list = load_sensitive_api_list(sensitive_api_txt)

    # APK file paths
    family_dirs = [
        '/storage/xiaowei_data/family_data/adpush',
        '/storage/xiaowei_data/family_data/artemis',
        '/storage/xiaowei_data/family_data/dzhtny',
        '/storage/xiaowei_data/family_data/igexin',
        '/storage/xiaowei_data/family_data/kuguo',
        '/storage/xiaowei_data/family_data/leadbolt',
        '/storage/xiaowei_data/family_data/openconnection',
        '/storage/xiaowei_data/family_data/spyagent'
    ]

    # Family label mapping
    family_paths = {}
    label_map = {}
    label_counter = 0

    # Traverse family path list
    for family_dir in family_dirs:
        if os.path.isdir(family_dir):  # Ensure the path is a directory
            family_name = os.path.basename(family_dir)  # Extract family name from path
            print("family name=", family_name)
            apk_files = [os.path.join(family_dir, f) for f in os.listdir(family_dir) if f.endswith('.apk')]
            family_paths[family_name] = apk_files
            label_map[family_name] = label_counter
        label_counter += 1


    # Process APKs in the main loop:
    for family_name, apk_files in family_paths.items():
        label = label_map[family_name]
        for apk_path in apk_files:
          
            apk_id = os.path.splitext(os.path.basename(apk_path))[0]
            dist = os.path.join(output_dir, family_name)
            output_file = os.path.join(dist, f"{apk_id}.dgl")
            
            if os.path.exists(output_file):
                print(f"{apk_id}.dgl already exists, skipping")
                continue
            
            print(f"Processing {apk_path}...")
            
            try:
                # Extract graph and features
                G, apk, dvm, analysis = extract_call_graph(apk_path)
                print("Identifying sensitive API nodes...")
                sensitive_nodes = get_sensitive_nodes(G, sensitive_api_list)
                
                print("Simplifying function call graph...")
                simplified_G = simplify_fcg_with_edges(G, sensitive_nodes, N)
                print(f"Simplified call graph - Nodes: {simplified_G.number_of_nodes()}, Edges: {simplified_G.number_of_edges()}")

                # Check if the simplified graph is empty
                if not simplified_G.nodes():
                    print(f"Warning: Simplified graph for APK {apk_id} has no nodes, skipping")
                    continue
                
                # Extract features
                node_features, perm_vector = extract_node_features(
                    simplified_G, apk, dvm, analysis, 
                    opcode_features, api_features, permission_features
                )
            
                
                # Build DGL graph
                g, graph_labels = build_dgl_graph(simplified_G, node_features, perm_vector, label)
                if g is not None and graph_labels is not None:
                    dgl.save_graphs(output_file, [g], graph_labels)
                    print(f"Graph saved to {output_file}")
            
            except Exception as e:
                print(f"Error processing {apk_path}: {str(e)}")
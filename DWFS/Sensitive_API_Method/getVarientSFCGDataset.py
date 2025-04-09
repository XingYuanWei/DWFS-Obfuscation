import pandas as pd
import os
from androguard.misc import AnalyzeAPK
import networkx as nx
from collections import deque
from androguard.core.bytecodes.apk import APK
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.core.analysis.analysis import Analysis
import matplotlib.pyplot as plt
import dgl
import torch

# Load features
feature_df = pd.read_csv('/home/xiaowei_3machine/MalScan-master/Feature_Ranking.csv')
selected_features = feature_df['Feature'].tolist()
split = 400
selected_features = selected_features[:split]

# Separate feature types
opcode_features = [f for f in selected_features if '_freq' in f]  # Opcode frequencies
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
        if isinstance(caller, tuple):
            caller_full_name = f"{caller[0]}->{caller[1]}{caller[2]}"
        else:
            caller_full_name = f"{caller.get_class_name()}->{caller.get_name()}{caller.get_descriptor()}"
        
        if isinstance(callee, tuple):
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

# Simplify call graph
def simplify_fcg_with_edges(G, sensitive_nodes, N=1):
    nodes_to_keep = set()
    edges_to_keep = set()
    
    for sensitive_node in sensitive_nodes:
        nodes_to_keep.add(sensitive_node)
        
        for pred in G.predecessors(sensitive_node):
            nodes_to_keep.add(pred)
            edges_to_keep.add((pred, sensitive_node))
        
        for succ in G.successors(sensitive_node):
            nodes_to_keep.add(succ)
            edges_to_keep.add((sensitive_node, succ))
        
        if N > 0:
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
    
    simplified_G = G.subgraph(nodes_to_keep).copy()
    return simplified_G

# Extract node features
def extract_node_features(simplified_G, apk, dvm, analysis, opcode_features, api_features, permission_features):
    permissions = apk.get_permissions()
    perm_vector = [1 if perm in permissions else 0 for perm in permission_features]

    node_features = {}
    
    for node in simplified_G.nodes():
        method = None
        for m in dvm.get_methods():
            if node == f"{m.get_class_name()}->{m.get_name()}{m.get_descriptor()}":
                method = m
                break
        
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

        if method:
            method_analysis = analysis.get_method_analysis(method)
            xrefs = method_analysis.get_xref_to()
            called_apis = set()
            for xref in xrefs:
                _, called_method, _ = xref
                called_apis.add(f"{called_method.get_class_name()}->{called_method.get_name()}{called_method.get_descriptor()}")
            api_vector = [1 if api in called_apis else 0 for api in api_features]
        else:
            api_vector = [0] * len(api_features)

        node_features[node] = opcode_vector + api_vector
    
    return node_features, perm_vector

# Build DGL graph
def build_dgl_graph(simplified_G, node_features, perm_vector, label, perm_length=None):
    if not simplified_G.nodes():
        print("Warning: No nodes in the graph.")
        return None, None

    g = dgl.from_networkx(simplified_G)

    if node_features:
        node_feature_matrix = []
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

    if not perm_vector:
        if perm_length is None:
            raise ValueError("perm_vector is empty and perm_length not provided")
        print("Warning: perm_vector is empty, using default zeros")
        perm_vector = [0] * perm_length

    graph_labels = {
        'perm': torch.tensor([perm_vector], dtype=torch.float),
        'label': torch.tensor([label], dtype=torch.long)
    }

    return g, graph_labels

# Main program
if __name__ == "__main__":
    # Parameters
    sensitive_api_txt = '/home/xiaowei_3machine/MalScan-master/Data/sensitive_apis.txt'
    N = 3  # BFS hop count
    output_dir = '/storage/xiaowei_data/DWFS-Obfuscation_Data'
    os.makedirs(output_dir, exist_ok=True)

    # Load sensitive APIs
    print("Loading sensitive API list...")
    sensitive_api_list = load_sensitive_api_list(sensitive_api_txt)

    # Family paths as a list
    family_dirs = [
        '/storage/xiaowei_data/purity_data_obfuscate/adpush',
        '/storage/xiaowei_data/purity_data_obfuscate/artemis',
        '/storage/xiaowei_data/purity_data_obfuscate/dzhtny',
        '/storage/xiaowei_data/purity_data_obfuscate/igexin',
        '/storage/xiaowei_data/purity_data_obfuscate/kuguo',
        '/storage/xiaowei_data/purity_data_obfuscate/leadbolt',
        '/storage/xiaowei_data/purity_data_obfuscate/openconnection',
        '/storage/xiaowei_data/purity_data_obfuscate/spyagent'
    ]

    # Obfuscation methods
    obf_methods = [
        'Rebuild_NewAlignment_NewSignature_CallIndirection',
        'Rebuild_NewAlignment_NewSignature_ClassRename',
        'Rebuild_NewAlignment_NewSignature_ConstStringEncryption',
        'Rebuild_NewAlignment_NewSignature_MethodRename'
    ]

    # Family label mapping
    label_map = {}
    label_counter = 0
    for family_dir in family_dirs:
        if os.path.isdir(family_dir):
            family_name = os.path.basename(family_dir)
            label_map[family_name] = label_counter
            label_counter += 1

    # Main loop: Process families and obfuscation methods
    for family_dir in family_dirs:
        if not os.path.isdir(family_dir):
            print(f"Warning: Family directory {family_dir} does not exist, skipping")
            continue

        family_name = os.path.basename(family_dir)
        label = label_map[family_name]

        for obf_method in obf_methods:
            obf_path = os.path.join(family_dir, obf_method)
            if not os.path.isdir(obf_path):
                print(f"Warning: Obfuscation directory {obf_path} does not exist, skipping")
                continue

            print(f"\nProcessing family: {family_name} | Obfuscation method: {obf_method}")
            apk_files = [os.path.join(obf_path, f) for f in os.listdir(obf_path) if f.endswith(".apk")]

            for apk_path in apk_files:
                apk_id = os.path.splitext(os.path.basename(apk_path))[0]
                save_dir = os.path.join(output_dir, family_name, obf_method)
                os.makedirs(save_dir, exist_ok=True)
                output_file = os.path.join(save_dir, f"{apk_id}.dgl")

                if os.path.exists(output_file):
                    print(f"{apk_id}.dgl already exists, skipping")
                    continue

                print(f"Processing APK: {apk_path}...")

                try:
                    # Extract call graph
                    G, apk, dvm, analysis = extract_call_graph(apk_path)
                    print("Identifying sensitive API nodes...")
                    sensitive_nodes = get_sensitive_nodes(G, sensitive_api_list)
                    
                    print("Simplifying function call graph...")
                    simplified_G = simplify_fcg_with_edges(G, sensitive_nodes, N)
                    print(f"Simplified graph - Nodes: {simplified_G.number_of_nodes()}, Edges: {simplified_G.number_of_edges()}")

                    if not simplified_G.nodes():
                        print(f"Warning: Simplified graph for APK {apk_id} has no nodes, skipping")
                        continue

                    # Extract features
                    node_features, perm_vector = extract_node_features(
                        simplified_G, apk, dvm, analysis,
                        opcode_features, api_features, permission_features
                    )

                    # Build and save DGL graph
                    g, graph_labels = build_dgl_graph(
                        simplified_G, node_features, perm_vector, label,
                        perm_length=len(permission_features)
                    )
                    if g is not None and graph_labels is not None:
                        dgl.save_graphs(output_file, [g], graph_labels)
                        print(f"Graph saved to: {output_file}")
                except Exception as e:
                    print(f"Error processing {apk_path}: {str(e)}")

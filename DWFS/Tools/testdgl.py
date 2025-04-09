import dgl
import torch
import networkx as nx

# Create a simple NetworkX graph
G = nx.DiGraph()
G.add_edges_from([(0, 1), (1, 2)])
node_features = {0: [1.0, 2.0], 1: [3.0, 4.0], 2: [5.0, 6.0]}
perm_vector = [0, 1, 0]
label = 1
output_file = "test_graph.dgl"

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


# Build and save the graph
g, graph_labels = build_dgl_graph(G, node_features, perm_vector, label)
if g is not None:
    dgl.save_graphs(output_file, [g], graph_labels)

# Load the graph
graphs, labels = dgl.load_graphs(output_file)
g_loaded = graphs[0]
perm_loaded = labels['perm'][0]
label_loaded = labels['label'][0]

# Verify results
print(f"Loaded perm: {perm_loaded}")
print(f"Loaded label: {label_loaded}")
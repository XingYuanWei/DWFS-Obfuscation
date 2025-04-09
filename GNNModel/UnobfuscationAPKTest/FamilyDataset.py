from torch.utils.data import Dataset, DataLoader, random_split
import dgl
import torch
from pathlib import Path

# Define malware family labels
FAMILY_LABELS = {
    "adpush": 0,
    "artemis": 1,
    "dzhtny": 2,
    "igexin": 3,
    "kuguo": 4,
    "leadbolt": 5,
    "openconnection": 6,
    "spyagent": 7
}

def load_graphs_from_directory(graph_dir: str, family_label: int):
    """
    Load all graph files from a directory and extract features and labels
    :param graph_dir: Directory containing graph files
    :param family_label: Family label (e.g., 0: adpush, 1: artemis, ...)
    :return: List of graphs, list of permission features, and list of corresponding labels
    """
    graph_dir = Path(graph_dir)
    graphs = []
    perm_features = []
    labels = []

    for graph_file in graph_dir.iterdir():
        if graph_file.suffix == ".dgl":
            try:
                graph_list, graph_data = dgl.load_graphs(str(graph_file))
                g = graph_list[0]  # Only one graph is saved each time
                perm = graph_data['perm'][0]  # Graph-level permission features
                label = graph_data['label'].item()  # Graph-level label
                
                graphs.append(g)
                perm_features.append(perm)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {graph_file}: {str(e)}")
    
    return graphs, perm_features, torch.tensor(labels, dtype=torch.long)



def load_data(parent_dir: str, num_families=None):
    """
    Load data for a specified number of families and merge them
    :param parent_dir: Parent directory containing family folders
    :param num_families: Optional, specify the number of families to load
    :return: Merged list of graphs, permission features, labels, and paths
    """
    graphs = []
    perm_features = []
    labels = []
    graph_paths = []

    # Traverse specified malware families
    family_names = list(FAMILY_LABELS.keys())
    if num_families is not None:
        family_names = family_names[:num_families]
    
    for family_name in family_names:
        family_dir = Path(parent_dir) / family_name
        if not family_dir.exists():
            print(f"Warning: Family directory {family_dir} does not exist, skipping")
            continue
        
        family_graphs, family_perm_features, family_labels = load_graphs_from_directory(family_dir, FAMILY_LABELS[family_name])
        graphs.extend(family_graphs)
        perm_features.extend(family_perm_features)
        labels.extend(family_labels)
        # Get list of paths
        graph_paths.extend([str(graph_file) for graph_file in family_dir.rglob("*.dgl")])
    
    return graphs, perm_features, torch.tensor(labels, dtype=torch.long), graph_paths


class GraphDataset(Dataset):
    def __init__(self, graphs, perm_features, labels, graph_paths):
        """
        Initialize the dataset
        :param graphs: List of graphs
        :param perm_features: List of permission features
        :param labels: List of labels
        :param graph_paths: List of graph file paths
        """
        self.graphs = graphs
        self.perm_features = perm_features
        self.labels = labels
        self.graph_paths = graph_paths
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        """
        Return a single sample: graph, permission features, label, and path
        """
        return self.graphs[idx], self.perm_features[idx], self.labels[idx], self.graph_paths[idx]

def collate_fn(batch):
    """
    Custom collate function for batch processing (graph, perm_feature, label, path)
    :param batch: A batch of data
    :return: Batched graph, permission features, labels, and paths
    """
    graphs, perm_features, labels, paths = zip(*batch)
    # Add self-loops to avoid nodes with zero in-degree
    graphs = [add_self_loops_if_needed(graph) for graph in graphs]
    
    batched_graph = dgl.batch(graphs)  # Merge multiple graphs
    batched_perm_features = torch.stack(perm_features)  # Merge permission features
    batched_labels = torch.stack(labels)  # Merge labels
    
    return batched_graph, batched_perm_features, batched_labels, paths

def split_dataset(dataset, train_ratio=0.8):
    """
    Split the dataset into training and test sets based on a ratio
    :param dataset: The dataset
    :param train_ratio: Proportion of the training set
    :return: Training set and test set
    """
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

def get_data_loaders(parent_dir: str, batch_size=4, train_ratio=0.8, num_families=None):
    """
    Load data, split into training and test sets, and return corresponding data loaders
    :param parent_dir: Parent directory containing family folders
    :param batch_size: Batch size for the data loaders
    :param train_ratio: Proportion of the training set
    :param num_families: Optional, specify the number of families to load
    :return: Data loaders for training and test sets
    """
    # Load data
    graphs, perm_features, labels, graph_paths = load_data(parent_dir, num_families=num_families)
    
    # Check if data is empty
    if not graphs:
        raise ValueError("No graph data loaded, please check the directory structure or file existence")
    
    # Construct dataset
    dataset = GraphDataset(graphs, perm_features, labels, graph_paths)
    
    # Split dataset
    train_dataset, test_dataset = split_dataset(dataset, train_ratio)
    
    # Construct data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader

def add_self_loops_if_needed(graph):
    """
    Add self-loops to avoid nodes with zero in-degree
    """
    if graph.in_degrees().min() == 0:  # If there are nodes with zero in-degree
        graph = dgl.add_self_loop(graph)  # Add self-loops
    return graph


def test():
    file_path = "/storage/xiaowei_data/DWFS-Obfuscation/adpush/1BA90D155E07A936A378A06584B400BC7FFA9FB65CCAF6362DDAF4EA32D7ABA6.dgl"
    graphs, graph_data = dgl.load_graphs(file_path)
    g = graphs[0]
    print(graph_data['perm'])
    print("Graph data keys:", list(graph_data.keys()))
    print("Node features:", g.ndata['features'].shape if 'features' in g.ndata else "Not found")


if __name__ == "__main__":
    
    # test()
    parent_dir = "/storage/xiaowei_data/DWFS-Obfuscation"  # Parent directory containing family folders
    batch_size = 16
    train_ratio = 0.8
    num_families = 4  # Example: Load 4 families

    try:
        train_loader, test_loader = get_data_loaders(parent_dir, batch_size=batch_size, train_ratio=train_ratio, num_families=num_families)
        
        # Example: Iterate through the training set
        for batched_graph, perm_features, labels, paths in train_loader:
            print("Batched Graph:", batched_graph)
            print("Node Features Shape:", batched_graph.ndata['features'].shape)
            print("Permission Features Shape:", perm_features.shape)
            print("Labels:", labels)
            print("Paths:", paths)
            break  # Only print the first batch
    except Exception as e:
        print(f"Error loading data: {str(e)}")
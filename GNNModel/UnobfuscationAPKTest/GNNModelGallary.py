import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn

# GCN Model
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, perm_length):
        super(GCNModel, self).__init__()
        self.conv1 = dglnn.GraphConv(input_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim + perm_length, num_classes)

    def forward(self, g, h, perm_features):
        h = self.conv1(g, h)
        h = torch.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        combined = torch.cat([hg, perm_features], dim=1)
        return self.classify(combined)  

# GAT Model
class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=8, perm_length=100):
        super(GATModel, self).__init__()
        self.conv1 = dglnn.GATConv(input_dim, hidden_dim, num_heads=num_heads)
        self.conv2 = dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads=num_heads)
        # Fully connected layer input dimension is hidden_dim * num_heads + perm_length
        self.classify = nn.Linear(hidden_dim * num_heads + perm_length, num_classes)

    def forward(self, g, h, perm_features):
        h = self.conv1(g, h)
        h = torch.relu(h)
        h = h.view(h.shape[0], -1)  # Flatten to (num_nodes, num_heads * hidden_dim)
        h = self.conv2(g, h)
        h = h.view(h.shape[0], -1)  # Flatten again
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')  # Aggregate node features into graph-level representation
        # Concatenate graph-level features with permission features
        combined = torch.cat([hg, perm_features], dim=1)
        return self.classify(combined)

# GraphSAGE Model
class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, perm_length):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = dglnn.SAGEConv(input_dim, hidden_dim, 'mean')
        self.conv2 = dglnn.SAGEConv(hidden_dim, hidden_dim, 'mean')
        # Fully connected layer input dimension is hidden_dim + perm_length
        self.classify = nn.Linear(hidden_dim + perm_length, num_classes)

    def forward(self, g, h, perm_features):
        h = self.conv1(g, h)
        h = torch.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')  # Aggregate node features into graph-level representation
        # Concatenate graph-level features with permission features
        combined = torch.cat([hg, perm_features], dim=1)
        return self.classify(combined)

# TAGConv Model
class TAGConvModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, perm_length):
        super(TAGConvModel, self).__init__()
        self.conv1 = dglnn.TAGConv(input_dim, hidden_dim)
        self.conv2 = dglnn.TAGConv(hidden_dim, hidden_dim)
        # Fully connected layer input dimension is hidden_dim + perm_length
        self.classify = nn.Linear(hidden_dim + perm_length, num_classes)

    def forward(self, g, h, perm_features):
        h = self.conv1(g, h)
        h = torch.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')  # Aggregate node features into graph-level representation
        # Concatenate graph-level features with permission features
        combined = torch.cat([hg, perm_features], dim=1)
        return self.classify(combined)

# DotGAT Model
class DotGATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads, perm_length):
        super(DotGATModel, self).__init__()
        self.conv1 = dglnn.DotGatConv(input_dim, hidden_dim, num_heads) 
        self.conv2 = dglnn.DotGatConv(hidden_dim, hidden_dim, num_heads)
        # Fully connected layer input dimension is hidden_dim + perm_length
        self.classify = nn.Linear(hidden_dim + perm_length, num_classes)

    def forward(self, g, h, perm_features):
        h = self.conv1(g, h)
        h = torch.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')  # Aggregate node features into graph-level representation
        # Concatenate graph-level features with permission features
        combined = torch.cat([hg, perm_features], dim=1)
        return self.classify(combined)

# Get the corresponding GNN model
def get_gnn_model(model_type, input_dim, hidden_dim, num_classes, num_heads=8, perm_length=3):
    """
    Return the corresponding GNN model based on the model type.

    Args:
        model_type (str): Model type ('GAT', 'GraphSAGE', 'TAGConv', 'DotGAT', 'GCN')
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        num_classes (int): Number of classes
        num_heads (int, optional): Number of attention heads for GAT models, default is 8
        perm_length (int, optional): Length of permission features, default is 3

    Returns:
        nn.Module: The corresponding GNN model
    """
    if model_type == 'GAT':
        return GATModel(input_dim, hidden_dim, num_classes, num_heads, perm_length)
    elif model_type == 'GraphSAGE':
        return GraphSAGEModel(input_dim, hidden_dim, num_classes, perm_length)
    elif model_type == 'TAGConv':
        return TAGConvModel(input_dim, hidden_dim, num_classes, perm_length)
    elif model_type == 'DotGAT':
        return DotGATModel(input_dim, hidden_dim, num_classes, num_heads, perm_length)
    elif model_type == 'GCN':
        return GCNModel(input_dim, hidden_dim, num_classes, perm_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
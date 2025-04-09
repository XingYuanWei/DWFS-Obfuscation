import dgl
import torch
import csv

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ObfuscationGNNModelGallary import get_gnn_model

# Define family labels
FAMILY_LABELS = {"adpush": 0, "artemis": 1, "dzhtny": 2, "igexin": 3, "kuguo": 4, "leadbolt": 5, "openconnection": 6, "spyagent": 7}

class ObfuscatedDataset(Dataset):
    def __init__(self, family_dir: str, obfuscation_method: str, family_labels: dict):
        self.graphs = []
        self.perm_features = []
        self.labels = []
        self.graph_paths = []
        self.family_dir = Path(family_dir)
        self.obfuscation_method = obfuscation_method
        self.family_labels = family_labels
        self.load_graphs()

    def load_graphs(self):
        method_dir = self.family_dir / self.obfuscation_method
        if not method_dir.exists():
            print(f"Warning: Obfuscation method directory {method_dir} does not exist, skipping")
            return
        family_name = self.family_dir.name
        label = self.family_labels[family_name]
        for graph_file in method_dir.iterdir():
            if graph_file.suffix == ".dgl":
                try:
                    graph_list, graph_data = dgl.load_graphs(str(graph_file))
                    g = graph_list[0]
                    perm = graph_data['perm'][0]
                    self.graphs.append(g)
                    self.perm_features.append(perm)
                    self.labels.append(label)
                    self.graph_paths.append(str(graph_file))
                except Exception as e:
                    print(f"Error loading {graph_file}: {str(e)}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.perm_features[idx], torch.tensor(self.labels[idx], dtype=torch.long), self.graph_paths[idx]



class CombinedObfuscatedDataset(Dataset):
    def __init__(self, obfuscated_dir, family_labels, obfuscation_methods):
        self.graphs = []
        self.perm_features = []
        self.labels = []
        self.graph_paths = []
        for family in family_labels.keys():
            family_dir = Path(obfuscated_dir) / family
            for method in obfuscation_methods:
                dataset = ObfuscatedDataset(str(family_dir), method, family_labels)
                self.graphs.extend(dataset.graphs)
                self.perm_features.extend(dataset.perm_features)
                self.labels.extend(dataset.labels)
                self.graph_paths.extend(dataset.graph_paths)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.perm_features[idx], torch.tensor(self.labels[idx], dtype=torch.long), self.graph_paths[idx]

def evaluate_combined_dataset(obfuscated_dir, family_labels, obfuscation_methods, batch_size=16):
    dataset = CombinedObfuscatedDataset(obfuscated_dir, family_labels, obfuscation_methods)
    if len(dataset) == 0:
        print("No data found, exiting...")
        return {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    accuracy, precision, recall, f1 = evaluate_on_obfuscated(model, loader, device)
    print(f"Combined evaluation results:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def add_self_loops_if_needed(graph):
    if graph.in_degrees().min() == 0:
        graph = dgl.add_self_loop(graph)
    return graph

def collate_fn(batch):
    graphs, perm_features, labels, paths = zip(*batch)
    graphs = [add_self_loops_if_needed(graph) for graph in graphs]
    batched_graph = dgl.batch(graphs)
    batched_perm_features = torch.stack(perm_features)
    batched_labels = torch.stack(labels)
    return batched_graph, batched_perm_features, batched_labels, paths

from sklearn.metrics import confusion_matrix
def evaluate_on_obfuscated(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batched_graph, perm_features, labels, paths in loader:
            batched_graph = batched_graph.to(device)
            perm_features = perm_features.to(device)
            labels = labels.to(device)
            features = batched_graph.ndata['features']
            logits = model(batched_graph, features, perm_features)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    return accuracy, precision, recall, f1

def evaluate_obfuscated_dataset(obfuscated_dir: str, family_labels: dict, obfuscation_methods: list, batch_size=16):
    results = {}
    for family in family_labels.keys():
        family_dir = Path(obfuscated_dir) / family
        for method in obfuscation_methods:
            print(f"Evaluating {family} with {method}...")
            dataset = ObfuscatedDataset(str(family_dir), method, family_labels)
            if len(dataset) == 0:
                print(f"No data found for {family} - {method}, skipping...")
                continue
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            accuracy, precision, recall, f1 = evaluate_on_obfuscated(model, loader, device)
            results[(family, method)] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            print(f"Family: {family}, Obfuscation Method: {method}")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
    return results

def save_results_to_csv(results, output_csv: str):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Family', 'Obfuscation Method', 'Accuracy', 'Precision', 'Recall', 'F1'])
        for (family, method), metrics in results.items():
            writer.writerow([family, method, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']])
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_type = 'GraphSAGE'
    model = get_gnn_model(model_type=model_type, input_dim=395, hidden_dim=128, num_classes=8, num_heads=8, perm_length=3)
    model.load_state_dict(torch.load('/home/xiaowei_3machine/MalScan-master/GraphSAGE_model.pth'))
    model = model.to(device)
    model.eval()

    # Obfuscation data parameters
    obfuscated_dir = "/storage/xiaowei_data/DWFS-Obfuscation_Data"
    obfuscation_methods =  [ 'Rebuild_NewAlignment_NewSignature_CallIndirection',
                            'Rebuild_NewAlignment_NewSignature_ClassRename',
                            'Rebuild_NewAlignment_NewSignature_ConstStringEncryption',
                            'Rebuild_NewAlignment_NewSignature_MethodRename'
                            ]  
    output_csv = '/home/xiaowei_3machine/MalScan-master/Output/' + model_type + 'obfuscated_results_combine.csv'

    # Evaluate and save results
    # results = evaluate_obfuscated_dataset(obfuscated_dir, FAMILY_LABELS, obfuscation_methods)
    # save_results_to_csv(results, output_csv)
    # Call combined evaluation
    results = evaluate_combined_dataset(obfuscated_dir, FAMILY_LABELS, obfuscation_methods)
    save_results_to_csv({'combined': results}, output_csv)
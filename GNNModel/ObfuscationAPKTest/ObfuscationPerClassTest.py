import dgl
import torch
import csv

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ObfuscationGNNModelGallary import get_gnn_model

# 定义家族标签
FAMILY_LABELS = {"adpush": 0, "artemis": 1, "dzhtny": 2, "igexin": 3,"kuguo": 4, "leadbolt": 5, "openconnection": 6, "spyagent": 7}

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
            print(f"警告: 混淆方法目录 {method_dir} 不存在，跳过")
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
                    print(f"加载 {graph_file} 时出错: {str(e)}")

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
        self.family_methods = []  # 记录每个样本的 (family, method)
        for family in family_labels.keys():
            family_dir = Path(obfuscated_dir) / family
            for method in obfuscation_methods:
                dataset = ObfuscatedDataset(str(family_dir), method, family_labels)
                self.graphs.extend(dataset.graphs)
                self.perm_features.extend(dataset.perm_features)
                self.labels.extend(dataset.labels)
                self.graph_paths.extend(dataset.graph_paths)
                self.family_methods.extend([(family, method)] * len(dataset))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return (
            self.graphs[idx],
            self.perm_features[idx],
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.graph_paths[idx],
            self.family_methods[idx]  # 返回 (family, method)
        )
        
        
def add_self_loops_if_needed(graph):
    if graph.in_degrees().min() == 0:
        graph = dgl.add_self_loop(graph)
    return graph


def collate_fn(batch):
    graphs, perm_features, labels, paths, family_methods = zip(*batch)
    graphs = [add_self_loops_if_needed(graph) for graph in graphs]
    batched_graph = dgl.batch(graphs)
    batched_perm_features = torch.stack(perm_features)
    batched_labels = torch.stack(labels)
    return batched_graph, batched_perm_features, batched_labels, paths, family_methods


from sklearn.metrics import confusion_matrix
def evaluate_on_obfuscated(model, loader, device, family_labels, obfuscation_methods):
    model.eval()
    all_preds = []
    all_labels = []
    all_family_methods = []
    with torch.no_grad():
        for batched_graph, perm_features, labels, paths, family_methods in loader:
            batched_graph = batched_graph.to(device)
            perm_features = perm_features.to(device)
            labels = labels.to(device)
            features = batched_graph.ndata['features']
            logits = model(batched_graph, features, perm_features)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_family_methods.extend(family_methods)

    # 计算综合混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(family_labels))))
    print("Overall Confusion Matrix:\n", cm)

    # 按家族和混淆策略分组计算指标
    results = {}
    for family in family_labels.keys():
        for method in obfuscation_methods:
            # 提取该家族和混淆策略的样本索引
            indices = [i for i, fm in enumerate(all_family_methods) if fm == (family, method)]
            if not indices:
                print(f"未找到 {family} - {method} 的数据，跳过...")
                continue
            family_labels_subset = [all_labels[i] for i in indices]
            family_preds_subset = [all_preds[i] for i in indices]

            # 如果没有预测结果，跳过
            if len(family_labels_subset) == 0:
                continue

            # 计算该子集的指标
            accuracy = accuracy_score(family_labels_subset, family_preds_subset)
            precision = precision_score(all_labels, all_preds, average='weighted', labels=[family_labels[family]], zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', labels=[family_labels[family]], zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', labels=[family_labels[family]], zero_division=0)

            results[(family, method)] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            print(f"家族: {family}, 混淆方法: {method}")
            print(f"准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}\n")

    return results

def evaluate_obfuscated_dataset(obfuscated_dir: str, family_labels: dict, obfuscation_methods: list, batch_size=16):
    dataset = CombinedObfuscatedDataset(obfuscated_dir, family_labels, obfuscation_methods)
    if len(dataset) == 0:
        print("未找到任何数据，退出...")
        return {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    results = evaluate_on_obfuscated(model, loader, device, family_labels, obfuscation_methods)
    return results


def save_results_to_csv(results, output_csv: str):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Family', 'Obfuscation Method', 'Accuracy', 'Precision', 'Recall', 'F1'])
        for (family, method), metrics in results.items():
            writer.writerow([family, method, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']])
    print(f"结果已保存到 {output_csv}")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_type = 'GAT'
    model = get_gnn_model(model_type=model_type, input_dim=395, hidden_dim=128, num_classes=8, num_heads=8, perm_length=3)
    model.load_state_dict(torch.load('/home/xiaowei_3machine/MalScan-master/GAT_model.pth'))
    model = model.to(device)
    model.eval()

    obfuscated_dir = "/storage/xiaowei_data/DWFS-Obfuscation_Data"
    obfuscation_methods = [
        'Rebuild_NewAlignment_NewSignature_CallIndirection',
        'Rebuild_NewAlignment_NewSignature_ClassRename',
        'Rebuild_NewAlignment_NewSignature_ConstStringEncryption',
        'Rebuild_NewAlignment_NewSignature_MethodRename'
    ]
    output_csv = '/home/xiaowei_3machine/MalScan-master/Output/' + model_type + '_obfuscated_results_per_family.csv'

    results = evaluate_obfuscated_dataset(obfuscated_dir, FAMILY_LABELS, obfuscation_methods)
    save_results_to_csv(results, output_csv)
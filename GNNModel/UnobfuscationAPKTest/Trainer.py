# import wandb
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from FamilyDataset import get_data_loaders
from GNNModelGallary import get_gnn_model


# Initialize wandb
# wandb.init(project='DWFS_malware')  # Set entity and project name as needed


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)
    # Normalize diagonal elements
    for i in range(len(cm)):
        cm[i, i] = cm[i, i] / cm[i].sum()  # Normalize diagonal elements of each row

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='GnBu', xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')
    
    # Save as PDF format
    plt.savefig(output_path, format='pdf')
    plt.close()



def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: GNN model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Computing device (CPU or GPU)

    Returns:
        float: Total loss for this epoch
    """
    model.train()
    total_loss = 0
    for batched_graph, perm_features, labels, paths in train_loader:
        # Move data to the specified device
        batched_graph = batched_graph.to(device)
        perm_features = perm_features.to(device)
        labels = labels.to(device)

        # Extract node features
        features = batched_graph.ndata['features']  # Note: Keep the original key name 'feature'

        # Forward pass of the model, passing graph, node features, and permission features
        logits = model(batched_graph, features, perm_features)

        # Compute loss
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def evaluate_model(model, test_loader, device, csv_writer=None, class_names=None, output_path=None, cm_output_path=None):
    model.eval()
    all_preds = []
    all_labels = []
    misclassified_data = []

    with torch.no_grad():
        for batched_graph, perm_features, labels, paths in test_loader:
            batched_graph = batched_graph.to(device)
            perm_features = perm_features.to(device)
            labels = labels.to(device)
            features = batched_graph.ndata['features']
            logits = model(batched_graph, features, perm_features)
            preds = torch.argmax(logits, dim=1)
            pred_probabilities = torch.softmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(preds)):
                deviation = abs(pred_probabilities[i][preds[i]].item() - 0.5)
                if preds[i] != labels[i]:
                    misclassified_data.append({
                        'path': paths[i],
                        'true_label': labels[i].item(),
                        'pred_label': preds[i].item(),
                        'prediction_probability': pred_probabilities[i][preds[i]].item(),
                        'deviation': deviation,
                        'error_type': 'FN' if preds[i] == 0 else 'FP'
                    })

    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='weighted')
    recall_macro = recall_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate per-class metrics
    accuracy_per_class = []
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    fpr_per_class = []
    fnr_per_class = []

    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        acc = (tp + tn) / cm.sum() if cm.sum() > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        accuracy_per_class.append(acc)
        precision_per_class.append(prec)
        recall_per_class.append(rec)
        f1_per_class.append(f1)
        fpr_per_class.append(fpr)
        fnr_per_class.append(fnr)

    # Save confusion matrix as a text file
    if cm_output_path:
        np.savetxt(cm_output_path, cm, delimiter=',', fmt='%d')
        print(f"Confusion matrix saved to {cm_output_path}")

    if class_names and output_path:
        plot_confusion_matrix(all_labels, all_preds, class_names, output_path)

    if csv_writer:
        for misclassified in misclassified_data:
            csv_writer.writerow([misclassified['path'], misclassified['true_label'],
                                 misclassified['pred_label'], misclassified['prediction_probability'],
                                 misclassified['deviation'], misclassified['error_type']])

    return {
        'accuracy': accuracy,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'accuracy_per_class': accuracy_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'fpr': fpr_per_class,
        'fnr': fnr_per_class,
        'cm': cm,
        'misclassified_data': misclassified_data
    }


def print_metrics(epoch, metrics, class_names, metrics_csv_path=None):
    print(f"\n=== Test Metrics after Epoch {epoch} ===")
    print("Overall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision']:.4f}")
    print(f"  Recall (macro): {metrics['recall']:.4f}")
    print(f"  F1-Score (macro): {metrics['f1']:.4f}")

    print("\nPer-Family Metrics:")
    cm = metrics['cm']

    # Prepare CSV data
    csv_rows = []
    csv_rows.append(['Scope', 'Family', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'TP', 'FP', 'FN', 'TN', 'FPR', 'FNR'])
    csv_rows.append(['Overall', '', metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], '', '', '', '', '', ''])

    # Output and collect metrics for each family
    for i, family_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        print(f"{family_name}:")
        print(f"  Accuracy: {metrics['accuracy_per_class'][i]:.4f}")
        print(f"  Precision: {metrics['precision_per_class'][i]:.4f}")
        print(f"  Recall: {metrics['recall_per_class'][i]:.4f}")
        print(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}")
        print(f"  TP (True Positives): {tp}")
        print(f"  FP (False Positives): {fp}")
        print(f"  FN (False Negatives): {fn}")
        print(f"  TN (True Negatives): {tn}")
        print(f"  FPR (False Positive Rate): {metrics['fpr'][i]:.4f}")
        print(f"  FNR (False Negative Rate): {metrics['fnr'][i]:.4f}")

        # Add to CSV data
        csv_rows.append(['Per-Family', family_name, metrics['accuracy_per_class'][i], metrics['precision_per_class'][i],
                         metrics['recall_per_class'][i], metrics['f1_per_class'][i], tp, fp, fn, tn,
                         metrics['fpr'][i], metrics['fnr'][i]])

    # Save metrics to CSV file
    if metrics_csv_path:
        with open(metrics_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(csv_rows)
        print(f"Metrics saved to {metrics_csv_path}")


## Save checkpoint
def save_checkpoint(epoch, model, optimizer, loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}.")


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from epoch {epoch}.")
    return epoch, loss


if __name__ == "__main__":
    # Set hyperparameters
    input_dim = 395
    hidden_dim = 128
    num_classes = 8  # Based on your original code, keep 2 classes
    perm_length = 3
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    epochs = 300

    # Initialize model
    model_type = 'GAT'
    model = get_gnn_model(model_type=model_type, input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, num_heads=8, perm_length=perm_length)
    model = model.to(device)
    model_type = model_type + 'weighted'
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Data loading
    parent_dir = "/storage/xiaowei_data/DWFS-Obfuscation"  # Keep your original path
    batch_size = 128
    train_ratio = 0.1
    train_loader, test_loader = get_data_loaders(parent_dir=parent_dir, batch_size=batch_size, train_ratio=train_ratio)

    # Checkpoint and log paths
    checkpoint_path = '/home/xiaowei_3machine/MalScan-master/Checkpoint/' + model_type + '_Model.pth'
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)

    class_names = ['AdPush', 'Artemis', 'Dzhtny', 'Igexin', 'Kuguo', 'Leadbolt', 'OpenConnection', 'SpyAgent']
    output_path = '/home/xiaowei_3machine/MalScan-master/Output/' + model_type + '_confusion_matrix.pdf'
    cm_output_path = '/home/xiaowei_3machine/MalScan-master/Output/' + model_type + '_Model_confusion_matrix.txt'  # Path to save the confusion matrix 2D array
    metrics_csv_path = '/home/xiaowei_3machine/MalScan-master/Output/' + model_type + '_Model_metrics.csv'  # Path for metrics output CSV
    

    # Create CSV file to record misclassified data
    with open('/home/xiaowei_3machine/MalScan-master/Output/' + model_type + 'family_misclassified_data.csv', mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Path', 'True Label', 'Predicted Label', 'Prediction Probability', 'Deviation', 'Error Type'])

        # Training loop
        print("Training...")
        start_time = time.time()
        for epoch in range(start_epoch, epochs):
            total_loss = train_one_epoch(model, train_loader, criterion, optimizer, device=device)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

            # Test every 10 epochs
            if (epoch + 1) % 10 == 0:
                metrics = evaluate_model(model, test_loader, device, csv_writer=csv_writer,
                                         class_names=class_names, output_path=output_path,
                                         cm_output_path=cm_output_path)
                print_metrics(epoch + 1, metrics, class_names, metrics_csv_path)

                # wandb logging (optional)
                # wandb.log({
                #     'epoch': epoch + 1,
                #     'train_loss': total_loss,
                #     'accuracy': metrics['accuracy'],
                #     'precision': metrics['precision'],
                #     'recall': metrics['recall'],
                #     'f1': metrics['f1'],
                #     'fpr': metrics['fpr'],
                #     'fnr': metrics['fnr'],
                #     'model_type': model_type
                # })

            # Save checkpoint every epoch
            save_checkpoint(epoch + 1, model, optimizer, total_loss, checkpoint_path)

        end_time = time.time()
        print(f"Program execution time: {end_time - start_time}")
        torch.save(model.state_dict(), '/home/xiaowei_3machine/MalScan-master/' + model_type + '_model.pth')
        print("Saved model!")
        # wandb.finish()
import dgl
import numpy as np
import os
import csv
from pathlib import Path

def calculate_family_statistics_to_csv(data_dir, output_csv):
    """Traverse subdirectories (family names) and calculate graph statistics, saving results to a CSV file

    Args:
        data_dir (str): Path containing all family subdirectories
        output_csv (str): Path to the output CSV file
    """
    family_stats = {}  # Dictionary to store family statistics

    # Traverse all subdirectories in the specified directory
    for family_dir in Path(data_dir).iterdir():
        if family_dir.is_dir():
            family_name = family_dir.name  # Subdirectory name is the family name
            
            # Initialize family statistics
            if family_name not in family_stats:
                family_stats[family_name] = {
                    'total_graphs': 0,
                    'total_size': 0,
                    'node_counts': [],
                    'edge_counts': [],
                    'degrees': []
                }

            # Traverse all .dgl files in the family directory
            for dgl_file in family_dir.glob('*.dgl'):
                try:
                    # Load graph data
                    graphs, _ = dgl.data.utils.load_graphs(str(dgl_file))
                    num_graphs = len(graphs)

                    # Update family statistics
                    family_stats[family_name]['total_graphs'] += num_graphs

                    for g in graphs:
                        family_stats[family_name]['node_counts'].append(g.number_of_nodes())
                        family_stats[family_name]['edge_counts'].append(g.number_of_edges())
                        family_stats[family_name]['degrees'].extend(g.in_degrees().tolist())

                    # Accumulate file size
                    family_stats[family_name]['total_size'] += dgl_file.stat().st_size

                except Exception as e:
                    print(f"Error: An exception occurred while processing file {dgl_file}: {str(e)}")
                    continue

    # Write statistics to CSV file
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'Family Name',
                'Total Graphs',
                'Total Size (GB)',
                'Node Avg',
                'Node Median',
                'Node Max',
                'Node Min',
                'Edge Avg',
                'Edge Median',
                'Edge Max',
                'Edge Min',
                'Degree Avg',
                'Degree Median',
                'Degree Max',
                'Degree Min'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header
            writer.writeheader()

            # Write each row of data
            for family_name, stats in family_stats.items():
                if not stats['node_counts']:
                    print(f"\nWarning: Family {family_name} has no valid graph data, skipping CSV write")
                    continue

                # Calculate statistics
                nodes = np.array(stats['node_counts'])
                edges = np.array(stats['edge_counts'])
                degrees = np.array(stats['degrees'])
                size_gb = stats['total_size'] / (1024 ** 3)

                writer.writerow({
                    'Family Name': family_name,
                    'Total Graphs': stats['total_graphs'],
                    'Total Size (GB)': f"{size_gb:.2f}",
                    'Node Avg': f"{nodes.mean():.2f}",
                    'Node Median': f"{np.median(nodes):.1f}",
                    'Node Max': nodes.max(),
                    'Node Min': nodes.min(),
                    'Edge Avg': f"{edges.mean():.2f}",
                    'Edge Median': f"{np.median(edges):.1f}",
                    'Edge Max': edges.max(),
                    'Edge Min': edges.min(),
                    'Degree Avg': f"{degrees.mean():.2f}",
                    'Degree Median': f"{np.median(degrees):.1f}",
                    'Degree Max': degrees.max(),
                    'Degree Min': degrees.min()
                })

        print(f"\nStatistics successfully saved to {output_csv}")

    except Exception as e:
        print(f"Error: An exception occurred while writing to CSV file: {str(e)}")


if __name__ == "__main__":
    # Set the directory path containing all .dgl files
    malware_dirs = '/storage/xiaowei_data/DWFS-Obfuscation'
    csv_path = '/home/xiaowei_3machine/MalScan-master/Output/statistic_data.csv'
    
    # Call the statistics function
    calculate_family_statistics_to_csv(malware_dirs, csv_path)
    print('Statistics completed')
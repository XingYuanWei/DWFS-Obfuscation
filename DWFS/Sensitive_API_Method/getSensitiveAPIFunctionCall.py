from androguard.misc import AnalyzeAPK
import networkx as nx
from collections import deque
import pandas as pd
from androguard.core.bytecodes.axml import AXMLPrinter
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.core.analysis.analysis import Analysis
from androguard.core.bytecodes.apk import APK

import os

"""This is the code for extracting subgraphs using BFS"""

def extract_call_graph(apk_path):
    apk = APK(apk_path)
    # Load and analyze the DEX file
    dvm = DalvikVMFormat(apk.get_dex())
    analysis = Analysis(dvm)
    dvm.create_python_export()
    analysis.create_xref()

    call_graph = analysis.get_call_graph()
    G = nx.DiGraph()

    for caller, callee in call_graph.edges:

        # Construct the full method name for the caller
        caller_class = caller.get_class_name()
        caller_name = caller.get_name()
        caller_descriptor = caller.get_descriptor()
        caller_full_name = f"{caller_class}->{caller_name}{caller_descriptor}"

        # Construct the full method name for the callee
        callee_class = callee.get_class_name()
        callee_name = callee.get_name()
        callee_descriptor = callee.get_descriptor()
        callee_full_name = f"{callee_class}->{callee_name}{callee_descriptor}"

        G.add_edge(caller_full_name, callee_full_name)
    
    return G


# Identify sensitive API nodes
def get_sensitive_nodes(G, sensitive_api_list):
    """
    Identify sensitive API nodes from the call graph.
    
    Parameters:
        G (nx.DiGraph): The complete function call graph
        sensitive_api_list (list): List of sensitive API signatures
    Returns:
        list: List of nodes in the graph that match sensitive APIs
    """
    return [node for node in G.nodes if node in sensitive_api_list]

# BFS algorithm to calculate N-hop neighborhood
def bfs_distance(G, start, max_hops):
    """
    Use BFS to calculate all nodes within N hops from the starting node.
    
    Parameters:
        G (nx.DiGraph): The complete function call graph
        start (str): Starting node (method signature)
        max_hops (int): Maximum number of hops
    Returns:
        set: Set of nodes within N hops
    """
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


def simplify_fcg(G, sensitive_nodes, N):
    """
    Simplify the call graph, retaining only sensitive nodes and their N-hop neighborhood.
    
    Parameters:
        G (nx.DiGraph): The complete function call graph
        sensitive_nodes (list): List of sensitive API nodes
        N (int): Maximum number of hops
    Returns:
        simplified_G (nx.DiGraph): Simplified call graph
    """
    nodes_to_keep = set()
    for sensitive_node in sensitive_nodes:
        nodes_to_keep.update(bfs_distance(G, sensitive_node, N))
    simplified_G = G.subgraph(nodes_to_keep).copy()
    return simplified_G



def load_sensitive_api_list(txt_path):
    """
    Read the list of sensitive APIs from a txt file.
    
    Parameters:
        txt_path (str): Path to the txt file containing sensitive API list
    Returns:
        list: List of sensitive API signatures
    """
    with open(txt_path, 'r') as file:
        sensitive_api_list = [line.strip() for line in file.readlines()]
    return sensitive_api_list



if __name__ == "__main__":
    # Paths to APK and txt files
    apk_path = '/storage/xiaowei_data/family_data/adcolony/0A18E562D029B45372F36DBFAB2AF9D4E8BD1BDCCEBEC1A851E079F5D87ABE09.apk'
    sensitive_api_txt = '/home/xiaowei_3machine/MalScan-master/Data/sensitive_apis.txt'
    N = 3  # Hop threshold

    # Load sensitive API list
    print("Loading sensitive API list...")
    sensitive_api_list = load_sensitive_api_list(sensitive_api_txt)

    # Extract the complete call graph
    print("Extracting function call graph...")
    G = extract_call_graph(apk_path)

    # Identify sensitive nodes
    print("Identifying sensitive API nodes...")
    sensitive_nodes = get_sensitive_nodes(G, sensitive_api_list)

    # Simplify the call graph
    print("Simplifying function call graph...")
    simplified_G = simplify_fcg(G, sensitive_nodes, N)

    # Output results
    print(f"Original call graph - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"Simplified call graph - Nodes: {simplified_G.number_of_nodes()}, Edges: {simplified_G.number_of_edges()}")

    # Save simplified nodes to a file
    apk_filename = os.path.basename(apk_path)
    apk_id = os.path.splitext(apk_filename)[0]
    output_file = f"{apk_id}_simplified_nodes.txt"
    with open(output_file, 'w') as f:
        for node in simplified_G.nodes():
            f.write(f"{node}\n")
    print(f"Simplified nodes saved to {output_file}")
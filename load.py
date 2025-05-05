import pickle
import json
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

DEBUG = True

def debug(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if DEBUG:
        print(f"[DEBUG {now}] {msg}")

def load_graph_and_metadata(path_prefix="citation_graph"):
    with open(f"new_build_parallel/{path_prefix}.gpickle", "rb") as f:
        graph = pickle.load(f)
    with open(f"new_build_parallel/{path_prefix}_folder_to_id.json", "r") as f:
        folder_to_id = json.load(f)
    with open(f"new_build_parallel/{path_prefix}_id_to_folder.json", "r") as f:
        id_to_folder = {int(k): v for k, v in json.load(f).items()}
    return graph, folder_to_id, id_to_folder

def analyze_graph(graph):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    num_isolated = len(list(nx.isolates(graph)))
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    avg_in_degree = sum(in_degrees.values()) / num_nodes
    avg_out_degree = sum(out_degrees.values()) / num_nodes

    debug(f"Edges: {num_edges}")
    debug(f"Isolated nodes: {num_isolated}")
    debug(f"Avg in-degree: {avg_in_degree:.2f}")
    debug(f"Avg out-degree: {avg_out_degree:.2f}")

    # Compute total degree
    degrees = {node: in_degrees[node] + out_degrees[node] for node in graph.nodes}

    # Histogram: Focus on 0–100
    plt.figure()
    plt.hist(list(degrees.values()), bins=100, range=(0, 100))
    plt.xlabel("Degree (0–100)")
    plt.ylabel("Number of Nodes")
    plt.title("Focused Degree Histogram")
    plt.savefig("plot_existing_focused.png")
    debug("Saved focused histogram to plot_existing_focused.png")

    # Identify high-degree outliers
    threshold = 100
    outliers = {node: deg for node, deg in degrees.items() if deg > threshold}

    # Optionally: Top 10 nodes by degree
    top_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]

    # Print results
    print("\n--- High-Degree Outliers (degree > 100) ---")
    for node, deg in sorted(outliers.items(), key=lambda x: x[1], reverse=True):
        print(f"Node {node} (Degree {deg})")

    print("\n--- Top 10 Nodes by Degree ---")
    for node, deg in top_degrees:
        print(f"Node {node} (Degree {deg})")

    # Optional: Save histogram with all degrees using log scale
    plt.figure()
    plt.hist(list(degrees.values()), bins=100)
    plt.yscale('log')
    plt.xlabel("Degree")
    plt.ylabel("Log Count of Nodes")
    plt.title("Full Degree Histogram (Log Scale)")
    plt.savefig("plot_existing_log.png")
    debug("Saved log-scale histogram to plot_existing_log.png")
    try:
        largest_cc = max(nx.weakly_connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc).copy()
        diameter = nx.diameter(subgraph.to_undirected())
        debug(f"Diameter of graph: {diameter}")
    except Exception as e:
        diameter = None
        debug(f"Could not compute diameter: {e}")

    return {
        "num_edges": num_edges,
        "num_nodes": num_nodes,
        "num_isolated_nodes": num_isolated,
        "avg_in_degree": avg_in_degree,
        "avg_out_degree": avg_out_degree,
        "diameter": diameter
    }

def main():
    graph, folder_to_id, id_to_folder = load_graph_and_metadata()
    stats = analyze_graph(graph)

    print("\n--- Graph Stats (Loaded) ---")
    for k, v in stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

import os
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from rapidfuzz import process, fuzz
import bibtexparser
from datetime import datetime
import subprocess
import sys
import logging
import json
from concurrent.futures import ProcessPoolExecutor
import pickle


DEBUG = True   # switch for debug logs
BIBTEXERROR = False  # set to True if bibtexparser fails to parse a .bib file
if not BIBTEXERROR:
    logging.getLogger("bibtexparser.bibdatabase").setLevel(logging.CRITICAL)
    logging.getLogger("bibtexparser.latexenc").setLevel(logging.CRITICAL)
    logging.getLogger("bibtexparser.bparser").setLevel(logging.CRITICAL)

def debug(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if DEBUG:
        print(f"[DEBUG {now}] {msg}")

def clean_latex(text):
    text = re.sub(r"\\[a-zA-Z]+\s*", "", text)
    text = re.sub(r"[{}]", "", text)
    return text.strip()

def extract_title(paper_path):
    title_path = os.path.join(paper_path, "title.txt")
    if os.path.exists(title_path):
        with open(title_path, "r", encoding='utf-8', errors='ignore') as f:
            return clean_latex(f.read().strip().lower())
    return None

def extract_citations(paper_path):
    bib_entries = []

    for file in os.listdir(paper_path):
        full_path = os.path.join(paper_path, file)

        if file.endswith(".bib"):
            try:
                with open(full_path, encoding="utf-8", errors='ignore') as bibtex_file:
                    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
                    parser.ignore_nonstandard_types = True
                    bib_database = bibtexparser.load(bibtex_file, parser=parser)
                    for entry in bib_database.entries:
                        title = entry.get("title", "")
                        if title:
                            clean_title = clean_latex(title.lower())
                            bib_entries.append(clean_title)
            except Exception as e:
                debug(f"Error parsing {file}: {e}")

        elif file.endswith(".bbl"):
            try:
                with open(full_path, "r", encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    matches = re.findall(r'title\s*=\s*[{"](.+?)[}"]', content)
                    for m in matches:
                        clean_title = clean_latex(m.lower())
                        bib_entries.append(clean_title)
            except Exception as e:
                debug(f"Error reading {file}: {e}")

    return list(set(bib_entries))

def process_folder(args):
    dataset_root, paper_folder = args
    paper_path = os.path.join(dataset_root, paper_folder)
    if not os.path.isdir(paper_path):
        return None
    title = extract_title(paper_path)
    if not title:
        return None
    citations = extract_citations(paper_path)
    return paper_folder, title, citations

def load_dataset(dataset_root):
    folder_to_title = {}
    title_to_folder = {}
    paper_refs = defaultdict(list)

    debug("Scanning dataset with multithreading...")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        args = [(dataset_root, f) for f in os.listdir(dataset_root)]
        for result in executor.map(process_folder, args):
            if result:
                folder, title, citations = result
                folder_to_title[folder] = title
                title_to_folder[title] = folder
                paper_refs[title] = citations
                debug(f"Parsed {folder} → title: {title[:60]}...")

    return folder_to_title, title_to_folder, paper_refs

folder_to_id = {}
id_to_folder = {}
NUM_NODES = 0
NUM_EDGES = 0

def build_graph(folder_to_title, title_to_folder, paper_refs):
    global folder_to_id, id_to_folder, NUM_EDGES

    graph = nx.DiGraph()
    all_titles = list(title_to_folder.keys())

    debug("Building citation graph...")

    folder_list = list(folder_to_title.keys())
    folder_to_id = {folder: idx for idx, folder in enumerate(folder_list)}
    id_to_folder = {idx: folder for folder, idx in folder_to_id.items()}

    for folder_id in folder_to_id.values():
        graph.add_node(folder_id)

    for citing_title, cited_titles in paper_refs.items():
        if citing_title not in title_to_folder:
            continue
        citing_folder = title_to_folder[citing_title]
        citing_id = folder_to_id[citing_folder]

        for ref_title in cited_titles:
            match = process.extractOne(
                ref_title, all_titles, scorer=fuzz.token_sort_ratio, score_cutoff=80
            )
            if match:
                matched_title = match[0]
                cited_folder = title_to_folder[matched_title]
                cited_id = folder_to_id[cited_folder]
                if citing_id != cited_id:
                    graph.add_edge(citing_id, cited_id)
                    debug(f"Edge: {citing_id} ({citing_folder}) -> {cited_id} ({cited_folder})")
            else:
                debug(f"No match for citation: '{ref_title[:50]}'")

    NUM_EDGES = graph.number_of_edges()
    return graph

def save_graph_and_metadata(graph, path_prefix="citation_graph"):
    os.makedirs("new", exist_ok=True)
    with open(f"new/{path_prefix}.gpickle", "wb") as f:
        pickle.dump(graph, f)
    with open(f"new/{path_prefix}_folder_to_id.json", "w") as f:
        json.dump(folder_to_id, f)
    with open(f"new/{path_prefix}_id_to_folder.json", "w") as f:
        json.dump(id_to_folder, f)
    debug(f"Saved graph and metadata with prefix 'new/{path_prefix}'")

def load_graph_and_metadata(path_prefix="citation_graph"):
    with open(f"new/{path_prefix}.gpickle", "rb") as f:
        graph = pickle.load(f)
    with open(f"new/{path_prefix}_folder_to_id.json", "r") as f:
        folder_to_id = json.load(f)
    with open(f"new/{path_prefix}_id_to_folder.json", "r") as f:
        id_to_folder = {int(k): v for k, v in json.load(f).items()}
    return graph, folder_to_id, id_to_folder

def analyze_graph(graph):
    global NUM_NODES
    NUM_NODES = graph.number_of_nodes()

    num_edges = graph.number_of_edges()
    num_isolated = len(list(nx.isolates(graph)))
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    avg_in_degree = sum(in_degrees.values()) / NUM_NODES
    avg_out_degree = sum(out_degrees.values()) / NUM_NODES

    debug(f"Edges: {num_edges}")
    debug(f"Isolated nodes: {num_isolated}")
    debug(f"Avg in-degree: {avg_in_degree:.2f}")
    debug(f"Avg out-degree: {avg_out_degree:.2f}")

    degrees = [in_degrees[node] + out_degrees[node] for node in graph.nodes]
    plt.hist(degrees, bins=50)
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.title("Degree Histogram")
    plt.savefig("plot.png")
    debug("Saved degree histogram to plot.png")

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
        "num_isolated_nodes": num_isolated,
        "avg_in_degree": avg_in_degree,
        "avg_out_degree": avg_out_degree,
        "diameter": diameter
    }

def print_adjacency(node_id, graph):
    if node_id not in id_to_folder:
        print(f"[ERROR] Node ID {node_id} does not exist.")
        return

    folder_name = id_to_folder[node_id]
    print(f"\n[Adjacency for Node {node_id} — {folder_name}]")
    neighbors = list(graph.successors(node_id))
    if not neighbors:
        print("  → No outgoing edges.")
        return

    for neighbor in neighbors:
        neighbor_folder = id_to_folder[neighbor]
        print(f"  → {neighbor} : {neighbor_folder}")

def main():
    dataset_root = "./dataset_papers"
    folder_to_title, title_to_folder, paper_refs = load_dataset(dataset_root)
    global graph
    graph = build_graph(folder_to_title, title_to_folder, paper_refs)
    save_graph_and_metadata(graph)
    stats = analyze_graph(graph)

    print("\n--- Graph Stats ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"NUM_NODES: {NUM_NODES}")
    print(f"NUM_EDGES: {NUM_EDGES}")

    if DEBUG:
        print_adjacency(0, graph)

if __name__ == "__main__":
    main()

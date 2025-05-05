import os
import random
import pickle
import json
import numpy as np
import torch
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import recall_score
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import logging

# Suppress sentence-transformers and transformers logs
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


# -------- Configuration --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "new_build_parallel"
OUTPUT_DIR = "specter_output"
LOG_FILE = os.path.join(OUTPUT_DIR, "train.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# -------- Step 1: Load Graph and Metadata --------
def load_graph_with_text():
    log("Loading graph, metadata, and text...")
    with open(f"{DATA_DIR}/citation_graph.gpickle", "rb") as f:
        graph = pickle.load(f)
    with open(f"{DATA_DIR}/citation_graph_id_to_folder.json") as f:
        id_to_folder = {int(k): v for k, v in json.load(f).items()}
    with open(f"{DATA_DIR}/citation_graph_id_to_text.json") as f:
        raw_text_data = json.load(f)
        id_to_title = {int(k): v["title"] for k, v in raw_text_data.items()}
        id_to_abstract = {int(k): v["abstract"] for k, v in raw_text_data.items()}
    log("Graph, metadata, and text loaded.")
    return graph, id_to_folder, id_to_title, id_to_abstract

# -------- Step 2: Seeding --------
def seed_everything(seed=42):
    log(f"Seeding everything with seed {seed}...")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log("Seeding complete.")

# -------- Step 3: Split and Save --------
def split_and_save_graph(graph, train_ratio=0.8):
    log("Splitting graph into train/test nodes...")
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    split_idx = int(train_ratio * len(nodes))
    train_nodes = nodes[:split_idx]
    test_nodes = nodes[split_idx:]

    train_subgraph = graph.subgraph(train_nodes).copy()

    # Save train subgraph
    with open(os.path.join(DATA_DIR, "train_subgraph.pickle"), "wb") as f:
        pickle.dump(train_subgraph, f)
    log(f"Train subgraph with {train_subgraph.number_of_nodes()} nodes and {train_subgraph.number_of_edges()} edges saved.")

    # Save test node IDs
    with open(os.path.join(DATA_DIR, "test_nodes.txt"), "w") as f:
        for node in test_nodes:
            f.write(f"{node}\n")
    log(f"{len(test_nodes)} test nodes saved.")

# -------- Step 4: Load Train Graph and Test Nodes --------
def load_split_graph():
    log("Loading train subgraph and test nodes...")
    with open(os.path.join(DATA_DIR, "train_subgraph.pickle"), "rb") as f:
        train_subgraph = pickle.load(f)
    with open(os.path.join(DATA_DIR, "test_nodes.txt"), "r") as f:
        test_nodes = [int(line.strip()) for line in f]
    log(f"Loaded train subgraph with {train_subgraph.number_of_nodes()} nodes and {train_subgraph.number_of_edges()} edges.")
    log(f"Loaded {len(test_nodes)} test nodes.")
    return train_subgraph, test_nodes

def evaluate_link_prediction(func, k, graph, train_subgraph, test_nodes):
    log(f"Evaluating link prediction with k = {k} over {len(test_nodes)} test nodes...")

    train_node_set = set(train_subgraph.nodes())
    score_sum = 0
    total = 0

    for t in tqdm(test_nodes, desc="Evaluating"):
        if t not in graph:
            continue  # skip test node not in full graph

        # Get predicted top-k train node connections
        predicted_neighbors = func(t, k)
        predicted_neighbors = [n for n in predicted_neighbors if n in train_node_set]

        # Get ground-truth connections to train graph
        real_neighbors = set(graph.neighbors(t))
        real_train_neighbors = real_neighbors.intersection(train_node_set)

        # If no true edges from t to train set: skip penalty (score = 1)
        if len(real_train_neighbors) == 0:
            score_sum += 1
        else:
            # Check if any predicted node is truly connected
            hit = any(n in real_train_neighbors for n in predicted_neighbors)
            score_sum += 1 if hit else 0

        total += 1

    avg_score = score_sum / total if total > 0 else 0
    log(f"Evaluation complete. Average score: {avg_score:.4f}")
    return avg_score


# -------- Load SPECTER Model --------
def load_specter():
    log("Loading SPECTER model...")
    model = SentenceTransformer('allenai/specter')
    model.to(DEVICE)
    log("SPECTER loaded.")
    return model

# -------- Encode All Train Nodes --------
def encode_train_embeddings(model, train_subgraph, id_to_title, id_to_abstract):
    log("Encoding train paper embeddings...")
    embeddings = {}
    for node in tqdm(train_subgraph.nodes(), desc="Encoding train"):
        title = id_to_title.get(node, "")
        abstract = id_to_abstract.get(node, "")
        text = f"{title}. {abstract}"
        emb = model.encode(text, convert_to_tensor=True, device=DEVICE)
        embeddings[node] = emb
    log("Train embeddings complete.")
    return embeddings

# -------- Top-K Prediction Function --------
def get_predictor(model, train_embeddings):
    train_ids = list(train_embeddings.keys())
    train_matrix = torch.stack([train_embeddings[n] for n in train_ids])

    def predict_topk(test_node_id, k):
        title = id_to_title.get(test_node_id, "")
        abstract = id_to_abstract.get(test_node_id, "")
        text = f"{title}. {abstract}"
        query_embedding = model.encode(text, convert_to_tensor=True, device=DEVICE)

        # Cosine similarity
        scores = util.cos_sim(query_embedding, train_matrix)[0]
        top_indices = torch.topk(scores, k=k).indices
        return [train_ids[i] for i in top_indices.tolist()]

    return predict_topk


# -------- Main --------
if __name__ == "__main__":
    seed_everything()

    graph, id_to_folder, id_to_title, id_to_abstract = load_graph_with_text()
    log(f"Full graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    log(f"Loaded {len(id_to_folder)} folder mappings, {len(id_to_title)} titles, and {len(id_to_abstract)} abstracts.")

    # Uncomment below to perform split and save
    # split_and_save_graph(graph)

    # Uncomment below to test loading the saved files
    train_graph, test_nodes = load_split_graph()
    specter = load_specter()
    # train_embeddings = encode_train_embeddings(specter, train_graph, id_to_title, id_to_abstract)
    # torch.save(train_embeddings, f"{DATA_DIR}/specter_train_embeddings.pt")
    train_embeddings = torch.load(f"{DATA_DIR}/specter_train_embeddings.pt")
    predictor = get_predictor(specter, train_embeddings)
    k_val = 20
    # Evaluate
    avg_score = evaluate_link_prediction(predictor, k=k_val, graph=graph, train_subgraph=train_graph, test_nodes=test_nodes)
    log(f"Final Recall@{k_val} style score: {avg_score:.4f}")

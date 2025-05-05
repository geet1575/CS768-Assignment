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
import argparse

# Suppress sentence-transformers and transformers logs
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


# -------- Configuration --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "new_build_parallel"
# OUTPUT_DIR = "specter_output"
# LOG_FILE = os.path.join(OUTPUT_DIR, "train.log")
DEBUG = False  # Set to True for debugging
# Set to False for production
# os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    if DEBUG:
        print(msg)

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
    if os.path.exists(os.path.join(DATA_DIR, "embeddings.pickle")):
        log("Embeddings already exist. Loading from disk...")
        with open(os.path.join(DATA_DIR, "embeddings.pickle"), "rb") as f:
            embeddings = pickle.load(f)
        return embeddings
    embeddings = {}
    # for node in tqdm(train_subgraph.nodes(), desc="Encoding train"):
    for node in train_subgraph.nodes():
        title = id_to_title.get(node, "")
        abstract = id_to_abstract.get(node, "")
        text = f"{title}. {abstract}"
        emb = model.encode(text, convert_to_tensor=True, device=DEVICE)
        embeddings[node] = emb
    log("Train embeddings complete.")
    # Save embeddings to disk
    with open(os.path.join(DATA_DIR, "embeddings.pickle"), "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="SPECTER Model Training")
    parser.add_argument('--title', type=str, required=True, help='Your title')
    parser.add_argument('--abstract', type=str, required=True, help='Your abstract')

    seed_everything()
    graph, id_to_folder, id_to_title, id_to_abstract = load_graph_with_text()
    log(f"Full graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    log(f"Loaded {len(id_to_folder)} folder mappings, {len(id_to_title)} titles, and {len(id_to_abstract)} abstracts.")
    specter = load_specter()
    embeddings = encode_train_embeddings(specter, graph, id_to_title, id_to_abstract)
    # Get input from argparse
    args = parser.parse_args()
    input_text = f"{args.title}. {args.abstract}"

    # Encode the input query
    log("Encoding input title + abstract...")
    input_embedding = specter.encode(input_text, convert_to_tensor=True, device=DEVICE)

    # Compute cosine similarities
    log("Computing cosine similarities...")
    similarities = []
    for node_id, emb in embeddings.items():
        score = util.cos_sim(input_embedding, emb).item()
        similarities.append((node_id, score))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Print results
    # print("\n=== Most Similar Papers ===")
    for node_id, score in similarities:
        # print(f"[{score:.4f}] ID: {node_id} | Title: {id_to_title.get(node_id, 'N/A')}")
        print(f"{id_to_title.get(node_id, 'N/A')}")


if __name__ == "__main__":
    main()
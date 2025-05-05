import os, json, random, pickle, shutil
from pathlib import Path
from typing import Tuple, List
import networkx as nx
import numpy as np
import torch

# ---------- 1. deterministic seeding ----------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ---------- 2. 80‑20 node split ----------
def make_train_test_split(
    graph: nx.DiGraph,
    train_ratio: float = 0.8,
    out_dir: str | Path = "splits",
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Splits *nodes* (papers) → train/test and saves:
      splits/train_nodes.json
      splits/test_nodes.json
      splits/train_graph.gpickle   (induced sub‑graph on train nodes only)
    """
    seed_everything(seed)
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    split = int(len(nodes) * train_ratio)
    train_nodes, test_nodes = nodes[:split], nodes[split:]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(train_nodes, (out_dir / "train_nodes.json").open("w"))
    json.dump(test_nodes, (out_dir / "test_nodes.json").open("w"))

    train_g = graph.subgraph(train_nodes).copy()
    nx.write_gpickle(train_g, out_dir / "train_graph.gpickle")
    return train_nodes, test_nodes

# ---------- 3. Recall@K evaluator ----------
def recall_at_k(preds: List[int], gold: List[int], k: int) -> float:
    """
    Single paper: 1 if *any* of top‑k preds is in gold list else 0.
    """
    return int(any(p in gold for p in preds[:k]))

def evaluate_model(
    model,                    # your callable: paper_id | text → List[int] (ranked)
    graph: nx.DiGraph,
    test_nodes: List[int],
    k: int = 10,
) -> float:
    """
    Feeds each test paper to `model`, collects Recall@K and returns the mean.
    """
    hits = 0
    for pid in test_nodes:
        gold = list(graph.successors(pid))  # papers it actually cites
        if not gold:            # skip isolated test papers
            continue
        preds = model(pid)      # must return ranked list of graph IDs
        hits += recall_at_k(preds, gold, k)
    return hits / len(test_nodes)

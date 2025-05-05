from datetime import datetime
import json, pickle, random, os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import networkx as nx
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx

DEBUG = True
def debug(msg: str) -> None:
    if DEBUG:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[DEBUG {now}] {msg}")

def load_graph_full(
    path_prefix: str = "citation_graph",
    build_dir: str | Path = "new_build_parallel",
) -> Tuple[
        "nx.DiGraph",                 # the citation graph
        Dict[str, int],               # folder → id
        Dict[int, str],               # id → folder
        Dict[int, dict]               # id → {"title": ..., "abstract": ...}
    ]:
    """
    Load every artefact produced by the build pipeline.

    Returns
    -------
    graph : nx.DiGraph
        The directed citation graph.
    folder_to_id : dict[str, int]
    id_to_folder : dict[int, str]
    id_to_text : dict[int, {"title": str, "abstract": str}]
    """
    build_dir = Path(build_dir)

    # Graph
    with (build_dir / f"{path_prefix}.gpickle").open("rb") as f:
        graph = pickle.load(f)
    debug(f"Loaded graph with {graph.number_of_nodes():,} nodes "
          f"and {graph.number_of_edges():,} edges")

    # Folder / ID mappings
    with (build_dir / f"{path_prefix}_folder_to_id.json").open() as f:
        folder_to_id = json.load(f)
    with (build_dir / f"{path_prefix}_id_to_folder.json").open() as f:
        id_to_folder = {int(k): v for k, v in json.load(f).items()}
    debug(f"Loaded {len(folder_to_id):,} folder↔︎id mappings")

    # Title + abstract mapping
    text_path = build_dir / f"{path_prefix}_id_to_text.json"
    if text_path.exists():
        with text_path.open() as f:
            id_to_text = {int(k): v for k, v in json.load(f).items()}
        debug(f"Loaded title/abstract for {len(id_to_text):,} papers")
    else:
        id_to_text = {}
        debug("⚠️  No id_to_text mapping found—run build_title_abstract_mapping.py")

    return graph, folder_to_id, id_to_folder, id_to_text


# ----------------------------- #
# 1. load_graph_full (verbatim) #
# ----------------------------- #
def load_graph_full(
    path_prefix: str = "citation_graph",
    build_dir: str | Path = "new_build_parallel",
    verbose: bool = True,
) -> Tuple[nx.DiGraph, Dict[str, int], Dict[int, str], Dict[int, dict]]:
    build_dir = Path(build_dir)
    with (build_dir / f"{path_prefix}.gpickle").open("rb") as f:
        graph = pickle.load(f)
    if verbose:
        print(f"Loaded graph  |V|={graph.number_of_nodes():,}  "
              f"|E|={graph.number_of_edges():,}")

    with (build_dir / f"{path_prefix}_folder_to_id.json").open() as f:
        folder_to_id = json.load(f)
    with (build_dir / f"{path_prefix}_id_to_folder.json").open() as f:
        id_to_folder = {int(k): v for k, v in json.load(f).items()}

    text_path = build_dir / f"{path_prefix}_id_to_text.json"
    if text_path.exists():
        with text_path.open() as f:
            id_to_text = {int(k): v for k, v in json.load(f).items()}
    else:
        id_to_text = {}
        if verbose:
            print("⚠️  id_to_text mapping missing – titles/abstracts unavailable")

    return graph, folder_to_id, id_to_folder, id_to_text


# ------------------------------------ #
# 2. Thin wrapper to keep things tidy  #
# ------------------------------------ #
class CitationGraphData:
    def __init__(
        self,
        graph: nx.DiGraph,
        folder_to_id: Dict[str, int],
        id_to_folder: Dict[int, str],
        id_to_text: Dict[int, dict],
    ):
        self.graph = graph
        self.folder_to_id = folder_to_id
        self.id_to_folder = id_to_folder
        self.id_to_text = id_to_text

    # convenience helpers
    def successors(self, pid: int) -> List[int]:
        return list(self.graph.successors(pid))

    def predecessors(self, pid: int) -> List[int]:
        return list(self.graph.predecessors(pid))


# --------------------------------------------------------------- #
# 3. PyTorch‑Geometric‑friendly dataset (node feats optional)     #
# --------------------------------------------------------------- #
class PyGDataset(InMemoryDataset):
    """
    Converts the NetworkX DiGraph into a single `torch_geometric.data.Data`
    object with edge_index, (optional) x, and boolean masks for train/test.
    """

    def __init__(
        self,
        root: str,
        cit_data: CitationGraphData,
        train_nodes: List[int],
        test_nodes: List[int],
        node_feats: Optional[np.ndarray] = None,
        transform=None,
        pre_transform=None,
    ):
        self.cit_data = cit_data
        self.train_nodes = train_nodes
        self.test_nodes = test_nodes
        self.node_feats = node_feats
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["citation_graph.pt"]

    def process(self):
        g_nx = self.cit_data.graph
        data = from_networkx(g_nx, group_node_attrs=None)

        if self.node_feats is not None:
            data.x = torch.from_numpy(self.node_feats).float()

        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros_like(train_mask)
        train_mask[self.train_nodes] = True
        test_mask[self.test_nodes] = True
        data.train_mask = train_mask
        data.test_mask = test_mask

        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(self.collate([data]), self.processed_paths[0])


# ---------------------------------------------------------------- #
# 4. Iterable edge‑pair loader with negative sampling               #
# ---------------------------------------------------------------- #
class EdgePairLoader(IterableDataset):
    """
    Streams (src, dst, label) where label=1 for true citation edges
    inside the *training* sub‑graph, label=0 for negatives.
    """

    def __init__(
        self,
        train_graph: nx.DiGraph,
        num_negs: int = 1,
        seed: int = 42,
    ):
        self.g = train_graph
        self.num_negs = num_negs
        self.nodes = list(train_graph.nodes())
        random.seed(seed)

    def __iter__(self):
        for src, dst in self.g.edges():
            # positive sample
            yield torch.tensor([src, dst]), torch.tensor([1], dtype=torch.float32)

            # negative samples
            for _ in range(self.num_negs):
                neg_dst = random.choice(self.nodes)
                while self.g.has_edge(src, neg_dst) or neg_dst == src:
                    neg_dst = random.choice(self.nodes)
                yield torch.tensor([src, neg_dst]), torch.tensor([0], dtype=torch.float32)


# ---------------------------------------------------------------- #
# 5. Quick usage demo                                              #
# ---------------------------------------------------------------- #
if __name__ == "__main__":
    # --- load raw artefacts ---
    g, f2id, id2f, id2txt = load_graph_full()

    cit_data = CitationGraphData(g, f2id, id2f, id2txt)

    # example 80‑20 node split (deterministic)
    rng = np.random.default_rng(42)
    nodes = np.array(list(g.nodes()))
    rng.shuffle(nodes)
    split = int(0.8 * len(nodes))
    train_nodes = nodes[:split].tolist()
    test_nodes = nodes[split:].tolist()
    train_g = g.subgraph(train_nodes).copy()

    # --- PyG dataset (no node features here) ---
    pyg_data = PyGDataset(
        root="pyg_processed",
        cit_data=cit_data,
        train_nodes=train_nodes,
        test_nodes=test_nodes,
    )

    # --- edge‑pair loader example ---
    edge_loader = DataLoader(
        EdgePairLoader(train_g, num_negs=3),
        batch_size=512,
        shuffle=True,
    )
    for (edges, labels) in edge_loader:
        print(edges.shape, labels.shape)  # (B, 2)  (B, 1)
        break


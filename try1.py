import os
import random
import pickle
import json
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


torch.set_float32_matmul_precision('high')

# -------- Configuration --------
MODEL_NAME = "allenai/scibert_scivocab_uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "new_build_parallel"
OUTPUT_DIR = "new_models"
LOG_FILE = os.path.join(OUTPUT_DIR, "train.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# -------- Step 1: Load Graph and Metadata --------
def load_graph():
    log("Loading graph and metadata...")
    with open(f"{DATA_DIR}/citation_graph.gpickle", "rb") as f:
        graph = pickle.load(f)
    with open(f"{DATA_DIR}/citation_graph_id_to_folder.json") as f:
        id_to_folder = {int(k): v for k, v in json.load(f).items()}
    log("Graph and metadata loaded.")
    return graph, id_to_folder

# -------- Step 2: Text Pair Dataset --------
class CitationDataset(Dataset):
    def __init__(self, edges, id_to_folder, tokenizer, max_len=256, negative_samples=1):
        self.tokenizer = tokenizer
        self.pairs = []
        self.labels = []

        node_ids = list(id_to_folder.keys())
        for src, tgt in tqdm(edges, desc="Building dataset"):
            src_text = read_paper_text(id_to_folder[src])
            tgt_text = read_paper_text(id_to_folder[tgt])
            if not src_text or not tgt_text:
                continue

            self.pairs.append((src_text, tgt_text))
            self.labels.append(1)

            for _ in range(negative_samples):
                negative = random.choice(node_ids)
                while (src, negative) in edges or negative == src:
                    negative = random.choice(node_ids)
                neg_text = read_paper_text(id_to_folder[negative])
                if neg_text:
                    self.pairs.append((src_text, neg_text))
                    self.labels.append(0)

        log(f"Dataset built with {len(self.pairs)} pairs.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        encoding = self.tokenizer(
            src, tgt, padding='max_length', truncation=True, max_length=256, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# -------- Step 3: Read Paper Content --------
def read_paper_text(folder_path):
    try:
        base = os.path.join("dataset_papers", folder_path)
        with open(os.path.join(base, "title.txt")) as t, open(os.path.join(base, "abstract.txt")) as a:
            return t.read().strip() + "\n" + a.read().strip()
    except:
        return None

# -------- Step 4: Lightning Module --------
class CitationPredictor(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = outputs.pooler_output
        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self(**batch)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=2e-5)

# -------- Step 5: Log Epoch End --------
class EpochLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        avg_loss = trainer.callback_metrics.get("train_loss")
        if avg_loss is not None:
            log(f"Epoch {trainer.current_epoch + 1}: train_loss = {avg_loss.item():.4f}")

# -------- Step 6: Evaluation Metric --------
def recall_at_k(model, dataset, k=10):
    model.eval()
    hits = 0
    total = 0

    # Index: paper A → list of indices in dataset where A is the source
    from collections import defaultdict
    source_index_map = defaultdict(list)
    for idx, (a, _) in enumerate(dataset.pairs):
        source_index_map[a].append(idx)

    with torch.no_grad():
        for i in range(len(dataset)):
            a, b = dataset.pairs[i]
            if dataset.labels[i] != 1:
                continue  # only evaluate on true citation pairs

            candidates = source_index_map[a]  # only test pairs with same A

            scores = []
            for j in candidates:
                input = dataset[j]
                input_tensor = {k: input[k].unsqueeze(0).to(DEVICE) for k in input if k != 'labels'}
                score = torch.sigmoid(model(**input_tensor)).item()
                scores.append((score, dataset.labels[j]))

            top_k = sorted(scores, reverse=True)[:k]
            if any(label == 1 for _, label in top_k):
                hits += 1
            total += 1

    return hits / total if total > 0 else 0

# -------- Step 7: Main --------
def evaluate_node_split_recall_at_k(model_path="new_models/citation_model.pt", k=10):
    log("🔍 Loading model...")
    model = CitationPredictor(MODEL_NAME)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

    log("📊 Loading graph...")
    graph, id_to_folder = load_graph()
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    split = int(0.8 * len(nodes))
    train_nodes = set(nodes[:split])
    test_nodes = set(nodes[split:])

    log(f"📂 Train nodes: {len(train_nodes)}, Test nodes: {len(test_nodes)}")

    # Preload and filter candidate texts from training nodes
    candidate_ids = []
    candidate_texts = []
    for nid in train_nodes:
        text = read_paper_text(id_to_folder[nid])
        if text:
            candidate_ids.append(nid)
            candidate_texts.append(text)

    log(f"✅ Loaded {len(candidate_texts)} training candidate papers.")

    missing_links_total = 0
    missing_links_hit = 0

    for test_node in tqdm(test_nodes, desc="Evaluating missing links"):
        src_text = read_paper_text(id_to_folder[test_node])
        if not src_text:
            continue

        # Get ground truth citations that point into the training set
        true_citations = [nid for nid in graph.successors(test_node) if nid in train_nodes]
        if not true_citations:
            continue  # No test-to-train citations to evaluate

        # Batch score against all training candidates
        with torch.no_grad():
            batch = tokenizer(
                [src_text] * len(candidate_texts),
                candidate_texts,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors='pt'
            ).to(DEVICE)

            logits = model(**batch)
            probs = torch.sigmoid(logits).cpu().tolist()
            results = list(zip(probs, candidate_ids))

        # Top-K predictions
        top_k = sorted(results, reverse=True)[:k]
        predicted_ids = {nid for _, nid in top_k}

        if any(nid in predicted_ids for nid in true_citations):
            missing_links_hit += 1
        missing_links_total += 1

    ratio = missing_links_hit / missing_links_total if missing_links_total > 0 else 0
    print(f"📈 Average Recall@{k} (node-split) = {ratio:.4f}")
    print(f"🔗 Missing links recovered: {missing_links_hit}/{missing_links_total}")
    return ratio


def main():
    model_path = os.path.join(OUTPUT_DIR, "citation_model.pt")
    log("🔍 Loading model...")
    model = CitationPredictor(MODEL_NAME)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

    log("📊 Loading graph and metadata...")
    graph, id_to_folder = load_graph()
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    split = int(0.8 * len(nodes))
    train_nodes = set(nodes[:split])
    test_nodes = set(nodes[split:])

    log(f"📂 Train nodes: {len(train_nodes)}, Test nodes: {len(test_nodes)}")

    # Preload candidate texts from training nodes
    candidate_texts = {}
    for nid in train_nodes:
        text = read_paper_text(id_to_folder[nid])
        if text:
            candidate_texts[nid] = text

    missing_links_total = 0
    missing_links_hit = 0

    for test_node in tqdm(test_nodes, desc="Evaluating missing links"):
        src_text = read_paper_text(id_to_folder[test_node])
        if not src_text:
            continue

        # True outgoing edges from test_node to training set
        true_citations = list(graph.successors(test_node))
        true_citations = [nid for nid in true_citations if nid in train_nodes]
        if not true_citations:
            continue  # No links to predict

        # Predict top-K targets from training set
        results = []
        with torch.no_grad():
            for tgt_id, tgt_text in candidate_texts.items():
                encoding = tokenizer(
                    src_text, tgt_text,
                    padding='max_length',
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                )
                input_tensor = {k: v.to(DEVICE) for k, v in encoding.items()}
                pred_score = torch.sigmoid(model(**input_tensor)).item()
                results.append((pred_score, tgt_id))

        top_k = sorted(results, reverse=True)[:10]
        predicted_ids = {nid for _, nid in top_k}

        if any(nid in predicted_ids for nid in true_citations):
            missing_links_hit += 1
        missing_links_total += 1

    ratio = missing_links_hit / missing_links_total if missing_links_total > 0 else 0
    print(f"🔗 Missing links predicted correctly: {missing_links_hit}/{missing_links_total} ({ratio:.4f})")

if __name__ == "__main__":
    evaluate_node_split_recall_at_k(k=10)
    # main()
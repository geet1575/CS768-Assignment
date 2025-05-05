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

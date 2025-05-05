import argparse
import json
import os
from datetime import datetime
from pathlib import Path

DEBUG = True
def debug(msg):
    if DEBUG:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[DEBUG {now}] {msg}")

def main(dataset_root: Path, prefix: str, build_dir: Path):
    # Load the existing ID → folder mapping
    id_to_folder_path = build_dir / f"{prefix}_id_to_folder.json"
    with id_to_folder_path.open() as f:
        id_to_folder = {int(k): v for k, v in json.load(f).items()}
    debug(f"Loaded {len(id_to_folder):,} folder mappings")

    id_to_text = {}
    missing = 0

    for pid, rel_path in id_to_folder.items():
        paper_dir = dataset_root / rel_path
        title_file = paper_dir / "title.txt"
        abs_file   = paper_dir / "abstract.txt"

        try:
            title = title_file.read_text(encoding="utf-8").strip()
            abstract = abs_file.read_text(encoding="utf-8").strip()
            id_to_text[pid] = {"title": title, "abstract": abstract}
        except FileNotFoundError:
            missing += 1
            debug(f"⚠️  Missing title/abstract for {paper_dir}")

    debug(f"Collected text for {len(id_to_text):,} papers "
          f"(missing {missing:,})")

    out_path = build_dir / f"{prefix}_id_to_text.json"
    out_path.write_text(json.dumps(id_to_text, ensure_ascii=False, indent=2))
    debug(f"Saved mapping → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Root directory containing the paper folders")
    parser.add_argument("--build-dir", type=Path, default=Path("new_build_parallel"),
                        help="Directory where existing *_id_to_folder.json lives")
    parser.add_argument("--prefix", type=str, default="citation_graph",
                        help="Filename prefix used for the graph artifacts")
    args = parser.parse_args()
    main(args.dataset_root, args.prefix, args.build_dir)

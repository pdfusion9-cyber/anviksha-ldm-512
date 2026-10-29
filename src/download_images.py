"""Download the Indian Dance Forms dataset (Apache-2.0) from the Hugging Face Hub.

Dataset: https://huggingface.co/datasets/tanmaykm/indian_dance_forms
License: Apache-2.0
8 classes of Indian classical dance: bharatanatyam, kathak, kathakali,
kuchipudi, manipuri, mohiniyattam, odissi, sattriya.

The downloader walks the repository tree and fetches only the missing files,
so it is safe to re-run after an interrupted download.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "tanmaykm/indian_dance_forms"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def list_tree(path: str, timeout: int = 60) -> list[dict]:
    with urllib.request.urlopen(API + path, timeout=timeout) as response:
        return json.loads(response.read())


def collect_images(split: str) -> list[str]:
    found: list[str] = []

    def walk(path: str) -> None:
        for entry in list_tree(path):
            if entry["type"] == "directory":
                walk(entry["path"])
            elif entry["path"].lower().endswith(IMAGE_SUFFIXES):
                found.append(entry["path"])

    walk(f"data/{split}")
    return found


def fetch(rel_path: str, root: Path, retries: int = 3) -> bool:
    destination = root / rel_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(RAW + rel_path, timeout=60) as response:
                payload = response.read()
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
            return True
        except Exception:
            if attempt == retries - 1:
                return False
            time.sleep(1 + attempt)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/dance_raw", type=Path)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--splits", default=",".join(SPLITS))
    args = parser.parse_args()

    root: Path = args.out
    root.mkdir(parents=True, exist_ok=True)

    remote: list[str] = []
    for split in args.splits.split(","):
        split_files = collect_images(split.strip())
        print(f"{split}: {len(split_files)} images on the hub")
        remote.extend(split_files)

    missing = [
        p for p in remote if not (root / p).exists() or (root / p).stat().st_size == 0
    ]
    print(f"missing locally: {len(missing)} of {len(remote)}")
    if not missing:
        return

    started = time.time()
    succeeded = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, ok in enumerate(pool.map(lambda p: fetch(p, root), missing), 1):
            succeeded += int(ok)
            if index % 200 == 0:
                print(f"  {index}/{len(missing)} ({time.time() - started:.0f}s)")

    print(f"downloaded {succeeded}/{len(missing)} in {time.time() - started:.0f}s")
    if succeeded != len(missing):
        raise SystemExit("some files failed to download; re-run to resume")


if __name__ == "__main__":
    main()

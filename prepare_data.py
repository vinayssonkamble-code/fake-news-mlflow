#!/usr/bin/env python
"""Step 1 - clean WELFake, remove duplicates, and write leak-free splits.

    python prepare_data.py --inspect     # REQUIRED once: eyeball samples to confirm label meaning
    python prepare_data.py               # write data/*.csv and data/split_manifest.json

Outputs
    data/train.csv, val.csv, test.csv   id,text,label,label_name
    data/new_data.csv                   id,text            (10 unseen articles, NO labels)
    data/held_back/new_data_labels.csv  id,label,label_name (kept apart, optional scoring only)
    data/split_manifest.json            counts, cleaning stats, file hashes
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

import config
from data_utils import clean_and_dedupe, make_splits
from mlflow_utils import sha256_file

COLS = ["id", "text", "label", "label_name"]


def md5_file(path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(config.RAW_CSV))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--inspect", action="store_true",
                    help="print 4 random articles per raw label and exit")
    ap.add_argument("--flip-labels", action="store_true",
                    help="use ONLY if --inspect shows the published label meaning is inverted")
    args = ap.parse_args()

    raw_path = args.raw
    try:
        raw = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"Raw file not found: {raw_path}\nDownload WELFake_Dataset.csv from "
              "https://zenodo.org/record/4561253 (CC BY 4.0) into data/raw/.")
        return 1

    md5 = md5_file(raw_path)
    if md5 != config.WELFAKE_MD5:
        print(f"WARNING: md5 {md5} differs from the Zenodo record ({config.WELFAKE_MD5}). "
              "The file may be a different version.")

    if args.inspect:
        for raw_label in (0, 1):
            sub = raw[raw["label"] == raw_label].dropna(subset=["text"]).sample(4, random_state=args.seed)
            print(f"\n===== raw label = {raw_label} "
                  f"(published meaning: {'FAKE' if raw_label == config.WELFAKE_FAKE_VALUE else 'REAL'}) =====")
            for _, r in sub.iterrows():
                print(f"- TITLE: {str(r['title'])[:100]}\n  TEXT : {str(r['text'])[:300].replace(chr(10), ' ')}\n")
        print("Check that the published meaning matches what you read.  If it is inverted, "
              "re-run prepare_data.py with --flip-labels and record this in the report.")
        return 0

    fake_value = 1 - config.WELFAKE_FAKE_VALUE if args.flip_labels else config.WELFAKE_FAKE_VALUE
    clean, stats = clean_and_dedupe(raw, fake_value, config.MIN_WORDS, config.FINGERPRINT_CHARS)
    splits = make_splits(clean, config.SPLIT_SIZES, args.seed)

    config.DATA_DIR.mkdir(exist_ok=True)
    config.HELD_BACK_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        splits[name][COLS].to_csv(config.DATA_DIR / f"{name}.csv", index=False)
    splits["new"][["id", "text"]].to_csv(config.DATA_DIR / "new_data.csv", index=False)
    splits["new"][COLS[:1] + COLS[2:]].to_csv(config.HELD_BACK_DIR / "new_data_labels.csv", index=False)

    manifest = {
        "dataset": "WELFake", "licence": "CC BY 4.0", "source": "https://zenodo.org/record/4561253",
        "raw_md5": md5, "raw_md5_expected": config.WELFAKE_MD5,
        "label_convention": {"published": "0 = fake, 1 = real",
                             "internal": "1 = FAKE (positive class), 0 = REAL",
                             "flipped_by_user": bool(args.flip_labels)},
        "seed": args.seed, "split_sizes": config.SPLIT_SIZES,
        "min_words": config.MIN_WORDS, "fingerprint_chars": config.FINGERPRINT_CHARS,
        "cleaning": stats,
        "class_counts": {k: {"REAL": int((v["label"] == 0).sum()), "FAKE": int((v["label"] == 1).sum())}
                         for k, v in splits.items()},
        "file_sha256": {f"{n}.csv": sha256_file(config.DATA_DIR / f"{n}.csv")
                        for n in ("train", "val", "test")} | {"new_data.csv": sha256_file(config.DATA_DIR / "new_data.csv")},
    }
    (config.DATA_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

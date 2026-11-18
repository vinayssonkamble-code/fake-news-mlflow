"""Central configuration for the Fake News Detection Champion-Challenger project.

The Champion rule (RULE below) is part of this file on purpose.  Commit this file
to Git BEFORE running the final experiment; compare_register.py logs the rule's
SHA-256 and the Git commit hash so the rule is traceable to a point in time that
precedes the results.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "WELFake_Dataset.csv"
HELD_BACK_DIR = DATA_DIR / "held_back"
OUT_DIR = ROOT / "outputs"

SEED = 42
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# ---------------------------------------------------------------- dataset ---
# WELFake (Zenodo record 4561253, CC BY 4.0) publishes: 0 = fake, 1 = real.
# Internally this project uses 1 = FAKE (positive class) and 0 = REAL, so that
# precision / recall / F1 describe how well FAKE articles are caught.
WELFAKE_FAKE_VALUE = 0
LABEL_NAMES = {0: "REAL", 1: "FAKE"}
LABEL_IDS = {"REAL": 0, "FAKE": 1}
WELFAKE_MD5 = "73c9675a4b3d09f86a6933d0b8d7d908"  # from the Zenodo record

SPLIT_SIZES = {"test": 1000, "val": 600, "new": 10, "train": 3000}
MIN_WORDS = 30          # drop articles with fewer body words
FINGERPRINT_CHARS = 300  # near-duplicate key: first N normalised characters

# ------------------------------------------- training (identical for A and B) ---
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EVAL_BATCH_SIZE = 32
LATENCY_WARMUP = 5      # untimed warm-up articles before latency measurement

SMOKE_SIZES = {"train": 64, "val": 32, "test": 32}
SMOKE_MAX_LEN = 64
SMOKE_EPOCHS = 1


@dataclass(frozen=True)
class ModelSpec:
    key: str
    run_name: str
    hf_id: str
    role: str
    short: str


MODELS = {
    "A": ModelSpec("A", "A_fakenews_distilbert", "distilbert-base-uncased",
                   "initial champion", "DistilBERT"),
    "B": ModelSpec("B", "B_fakenews_distilroberta", "distilroberta-base",
                   "challenger", "DistilRoBERTa"),
}
SUMMARY_RUN_NAME = "compare_fakenews"

# ---------------------------------------------- Champion rule (fixed in advance) ---
RULE = {
    "primary_metric": "macro_f1",      # higher is better
    "margin": 0.01,                    # minimum primary-metric gain that decides the result
    "tie_breaker": "avg_latency_ms",   # lower is better; used only when |delta| < margin
    "incumbent": "A",                  # wins only if everything is exactly tied
}


def rule_fingerprint() -> str:
    return hashlib.sha256(json.dumps(RULE, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Names:
    experiment: str
    registered_model: str


def names(smoke: bool = False) -> Names:
    if smoke:  # keeps smoke tests out of the real experiment and registry
        return Names("fake-news-champion-challenger-SMOKE", "fake-news-classifier-smoke")
    return Names("fake-news-champion-challenger", "fake-news-classifier")

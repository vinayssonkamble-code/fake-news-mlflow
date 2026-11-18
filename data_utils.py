
"""Cleaning, de-duplication and leak-free splitting using pandas and NumPy."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

LABEL_NAMES = {0: "REAL", 1: "FAKE"}
_WS = re.compile(r"\s+")


def normalise(s: str) -> str:
    return _WS.sub(" ", str(s).lower()).strip()


def build_text(title: str, body: str) -> str:
    return f"{title}\n{body}" if title else body


def clean_and_dedupe(raw: pd.DataFrame, fake_value: int = 0,
                     min_words: int = 30, fp_chars: int = 300):
    df = raw.copy()

    if "id" not in df.columns:
        df["id"] = (
            df["Unnamed: 0"]
            if "Unnamed: 0" in df.columns
            else np.arange(len(df))
        )

    stats = {"n_raw": int(len(df))}

    lab = pd.to_numeric(df["label"], errors="coerce")
    bad = ~lab.isin([0, 1])
    stats["n_bad_label"] = int(bad.sum())

    df = df[~bad].copy()
    df["label_raw"] = lab[~bad].astype(int)
    df["label"] = (df["label_raw"] == fake_value).astype(int)

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["body"] = df["text"].fillna("").astype(str).str.strip()

    short = df["body"].str.split().str.len().fillna(0) < min_words
    stats["n_too_short_or_empty"] = int(short.sum())
    df = df[~short].copy()

    df["fp"] = df["body"].map(
        lambda t: normalise(t)[:fp_chars]
    )

    conflict = df.groupby("fp")["label"].transform("nunique") > 1
    stats["n_label_conflict_rows"] = int(conflict.sum())
    df = df[~conflict].copy()

    before = len(df)
    df = df.drop_duplicates("fp", keep="first")
    stats["n_duplicate_rows_removed"] = int(before - len(df))

    df["text"] = [
        build_text(t, b)
        for t, b in zip(df["title"], df["body"])
    ]

    df["label_name"] = df["label"].map(LABEL_NAMES)
    df = df[
        ["id", "text", "label", "label_name", "fp"]
    ].reset_index(drop=True)

    stats["n_clean"] = int(len(df))
    stats["n_clean_fake"] = int((df["label"] == 1).sum())
    stats["n_clean_real"] = int((df["label"] == 0).sum())

    return df, stats


def _take(df: pd.DataFrame, n: int, seed: int):
    """Take a reproducible, stratified sample without scikit-learn."""
    if n < 0 or n > len(df):
        raise ValueError("Invalid requested sample size")

    rng = np.random.default_rng(seed)
    selected = []

    labels = sorted(df["label"].unique())
    counts = df["label"].value_counts()

    exact = {
        label: n * counts[label] / len(df)
        for label in labels
    }

    allocation = {
        label: int(np.floor(exact[label]))
        for label in labels
    }

    remaining = n - sum(allocation.values())

    order = sorted(
        labels,
        key=lambda label: exact[label] - allocation[label],
        reverse=True
    )

    for label in order[:remaining]:
        allocation[label] += 1

    for label in labels:
        group_indices = df.index[df["label"] == label].to_numpy()
        chosen = rng.choice(
            group_indices,
            size=allocation[label],
            replace=False
        )
        selected.extend(chosen.tolist())

    selected = np.array(selected)
    rng.shuffle(selected)

    taken = df.loc[selected].copy()
    rest = df.drop(index=selected).copy()

    return taken, rest


def make_splits(clean: pd.DataFrame, sizes: dict, seed: int = 42):
    """Create disjoint, stratified dataset splits."""
    if sum(sizes.values()) > len(clean):
        raise ValueError(
            f"requested {sum(sizes.values())} rows "
            f"but only {len(clean)} are available"
        )

    splits = {}
    rest = clean.copy()

    for name in ("test", "val", "new", "train"):
        splits[name], rest = _take(
            rest, sizes[name], seed
        )
        splits[name] = splits[name].reset_index(drop=True)

    assert_no_leakage(splits)
    return splits


def assert_no_leakage(splits: dict) -> None:
    names = list(splits)

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(splits[a]["id"]) & set(splits[b]["id"]):
                raise AssertionError(
                    f"id overlap between {a} and {b}"
                )

            if set(splits[a]["fp"]) & set(splits[b]["fp"]):
                raise AssertionError(
                    f"near-duplicate text overlap between {a} and {b}"
                )

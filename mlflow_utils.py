"""Small helpers shared by the scripts.  No torch / mlflow imports at module level."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pandas as pd


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame(df: pd.DataFrame) -> str:
    """Order-sensitive hash of (id, text, label); equal hashes => identical test inputs."""
    hashed = pd.util.hash_pandas_object(df[["id", "text", "label"]], index=False)
    return hashlib.sha256(hashed.values.tobytes()).hexdigest()


def git_commit(cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "not-a-git-repo"
    except Exception:
        return "git-unavailable"


def parse_predictions(output) -> tuple[list[str], list[float]]:
    """Normalise the output of a text-classification pyfunc model.

    Accepts a DataFrame with `label` / `score` columns, or a list of
    {"label", "score"} dicts.  Raises a clear error for anything else so that a
    change in the MLflow output format is noticed rather than silently mis-parsed.
    """
    if isinstance(output, pd.DataFrame):
        cols = {c.lower(): c for c in output.columns}
        if "label" in cols and "score" in cols:
            return output[cols["label"]].astype(str).tolist(), output[cols["score"]].astype(float).tolist()
        raise ValueError(f"Unexpected prediction columns: {list(output.columns)}")
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], dict):
        return [str(o["label"]) for o in output], [float(o["score"]) for o in output]
    raise ValueError(f"Unexpected prediction output type: {type(output).__name__}")

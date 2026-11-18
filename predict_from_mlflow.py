#!/usr/bin/env python
"""Step 4 - predict on 10 unseen articles through the Champion alias only.

    python predict_from_mlflow.py
    python predict_from_mlflow.py --with-labels   # optional: also compare with the held-back labels

This script never imports torch or transformers and never names a Hugging Face model.  It loads
models:/<registered model>@champion, so promoting a different version to `champion` changes the
model used here without touching this file.
"""
from __future__ import annotations

import argparse

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient

import config
from mlflow_utils import parse_predictions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--with-labels", action="store_true")
    args = ap.parse_args()

    name = config.names(args.smoke).registered_model
    mlflow.set_tracking_uri(config.TRACKING_URI)
    uri = f"models:/{name}@champion"

    mv = MlflowClient().get_model_version_by_alias(name, "champion")
    model = mlflow.pyfunc.load_model(uri)
    print(f"Loaded {uri} -> version {mv.version}, hf_model_id tag = {(mv.tags or {}).get('hf_model_id', 'n/a')}")

    df = pd.read_csv(config.DATA_DIR / "new_data.csv")
    labels, scores = parse_predictions(model.predict(df["text"].tolist()))

    out = pd.DataFrame({
        "id": df["id"], "predicted_label": labels, "confidence": [round(s, 4) for s in scores],
        "champion_version": mv.version, "model_uri": uri, "input_text": df["text"],
    })
    if args.with_labels:
        held = pd.read_csv(config.HELD_BACK_DIR / "new_data_labels.csv")
        out = out.merge(held[["id", "label_name"]].rename(columns={"label_name": "true_label"}), on="id")
        print(f"Agreement with held-back labels: {(out['predicted_label'] == out['true_label']).mean():.0%} "
              f"({len(out)} articles - too few for a reliable accuracy estimate)")

    config.OUT_DIR.mkdir(exist_ok=True)
    path = config.OUT_DIR / "new_data_predictions.csv"
    out.to_csv(path, index=False)
    print(out.drop(columns=["input_text", "model_uri"]).to_string(index=False))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()

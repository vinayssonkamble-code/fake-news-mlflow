#!/usr/bin/env python
"""Step 2 - fine-tune ONE model on the training split and evaluate it on the shared test split.

    python train_evaluate.py --model A
    python train_evaluate.py --model B
    python train_evaluate.py --model A --smoke      # tiny end-to-end check, separate experiment

Both models use identical data, hyper-parameters, seed, thread count and latency protocol.
The validation split is used only to keep the best epoch; the test split is touched once,
after training, and never influences training or checkpoint choice.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import mlflow.transformers
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

import config
from metrics_utils import compute_metrics, confusion, summarise_latency
from mlflow_utils import git_commit, parse_predictions, sha256_frame


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(name: str, smoke: bool) -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / f"{name}.csv")
    return df.head(config.SMOKE_SIZES[name]).copy() if smoke else df


def batches(texts, labels, bs, rng=None):
    idx = np.arange(len(texts))
    if rng is not None:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield [texts[k] for k in j], torch.tensor([labels[k] for k in j])


@torch.inference_mode()
def predict_batched(model, tok, texts, bs, max_len):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        out.extend(model(**enc).logits.argmax(-1).tolist())
    return np.array(out)


@torch.inference_mode()
def predict_single_timed(model, tok, texts, max_len, warmup):
    """One article at a time (batch size 1); times tokenisation + forward + softmax."""
    model.eval()
    for t in texts[:warmup]:  # untimed warm-up
        model(**tok(t, truncation=True, max_length=max_len, return_tensors="pt"))
    preds, probs, secs = [], [], []
    for t in texts:
        t0 = time.perf_counter()
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="pt")
        p = torch.softmax(model(**enc).logits, dim=-1)[0]
        secs.append(time.perf_counter() - t0)
        preds.append(int(p.argmax()))
        probs.append(float(p[1]))  # probability of FAKE
    return np.array(preds), np.array(probs), np.array(secs)


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=150)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["REAL", "FAKE"])
    ax.set_yticks([0, 1], labels=["REAL", "FAKE"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def log_model_compat(model, tok, max_len, input_example):
    kwargs = dict(
        transformers_model={"model": model, "tokenizer": tok},
        task="text-classification",
        input_example=input_example,
        model_config={"truncation": True, "max_length": max_len},
        pip_requirements=[
            "mlflow==3.18.0",
            "transformers==5.20.2",
            "torch==2.15.0"
        ]
    )

    try:
        return mlflow.transformers.log_model(name="model", **kwargs)
    except TypeError:
        return mlflow.transformers.log_model(artifact_path="model", **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(config.MODELS), required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="0 = PyTorch default (logged)")
    args = ap.parse_args()

    spec = config.MODELS[args.model]
    nm = config.names(args.smoke)
    max_len = config.SMOKE_MAX_LEN if args.smoke else config.MAX_LEN
    epochs = config.SMOKE_EPOCHS if args.smoke else config.EPOCHS
    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(config.SEED)
    device = "cpu"  # deliberate: the project targets a CPU-only laptop

    train, val, test = (load_split(n, args.smoke) for n in ("train", "val", "test"))
    test_hash = sha256_frame(test)

    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.hf_id, num_labels=2, id2label=config.LABEL_NAMES, label2id=config.LABEL_IDS).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(nm.experiment)

    with mlflow.start_run(run_name=spec.run_name) as run:
        mlflow.set_tags({
            "role": spec.role, "model_key": spec.key, "task": "fake-news-classification",
            "dataset": "WELFake", "dataset_licence": "CC BY 4.0", "positive_class": "FAKE",
            "hf_model_id": spec.hf_id, "test_set_sha256": test_hash,
            "git_commit": git_commit(config.ROOT), "smoke_test": str(args.smoke),
        })
        mlflow.log_params({
            "hf_model_id": spec.hf_id, "n_parameters": n_params, "seed": config.SEED,
            "max_len": max_len, "batch_size": config.BATCH_SIZE, "epochs": epochs,
            "learning_rate": config.LEARNING_RATE, "weight_decay": config.WEIGHT_DECAY,
            "warmup_ratio": config.WARMUP_RATIO, "train_n": len(train), "val_n": len(val),
            "test_n": len(test), "device": device, "torch_threads": torch.get_num_threads(),
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "mlflow_version": mlflow.__version__, "python_version": platform.python_version(),
            "platform": platform.platform(),
        })

        # ---------------------------------------------------------------- train ---
        texts, labels = train["text"].tolist(), train["label"].tolist()
        steps_per_epoch = -(-len(texts) // config.BATCH_SIZE)
        total_steps = steps_per_epoch * epochs
        opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                weight_decay=config.WEIGHT_DECAY)
        sched = get_linear_schedule_with_warmup(opt, int(config.WARMUP_RATIO * total_steps), total_steps)
        rng = np.random.default_rng(config.SEED)
        best_val, best_state, step = -1.0, None, 0
        t_train = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            running = []
            for bt, by in batches(texts, labels, config.BATCH_SIZE, rng):
                enc = tok(bt, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
                loss = model(**enc, labels=by).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                running.append(loss.item()); step += 1
                if step % 10 == 0:
                    mlflow.log_metric("train_loss_step", float(np.mean(running[-10:])), step=step)
            val_pred = predict_batched(model, tok, val["text"].tolist(), config.EVAL_BATCH_SIZE, max_len)
            vm = compute_metrics(val["label"].values, val_pred)
            mlflow.log_metrics({"train_loss": float(np.mean(running)), "val_macro_f1": vm["macro_f1"],
                                "val_accuracy": vm["accuracy"]}, step=epoch)
            print(f"[{spec.run_name}] epoch {epoch}/{epochs} loss={np.mean(running):.4f} "
                  f"val_macro_f1={vm['macro_f1']:.4f}")
            if vm["macro_f1"] > best_val:
                best_val = vm["macro_f1"]
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        mlflow.log_metrics({"train_time_s": time.perf_counter() - t_train, "best_val_macro_f1": best_val})

        # ------------------------------------------------------- test (held out) ---
        preds, probs, secs = predict_single_timed(model, tok, test["text"].tolist(), max_len,
                                                  config.LATENCY_WARMUP)
        m = compute_metrics(test["label"].values, preds)
        lat = summarise_latency(secs)
        mlflow.log_metrics({**m, **lat})

        pred_df = pd.DataFrame({
            "id": test["id"], "label": test["label"], "label_name": test["label_name"],
            "pred": preds, "pred_name": [config.LABEL_NAMES[p] for p in preds],
            "fake_probability": probs, "latency_ms": secs * 1000,
        })
        pred_df["correct"] = pred_df["label"] == pred_df["pred"]
        pred_df.to_csv("test_predictions.csv", index=False)
        mlflow.log_artifact("test_predictions.csv")

        cm = confusion(test["label"].values, preds)
        plot_confusion(cm, f"{spec.short} - confusion matrix", f"confusion_matrix_{spec.key}.png")
        mlflow.log_artifact(f"confusion_matrix_{spec.key}.png")
        mlflow.log_dict({**m, **lat, "confusion_matrix_[[TN,FP],[FN,TP]]": cm.tolist()}, "metrics.json")
        mlflow.log_artifact(str(config.DATA_DIR / "split_manifest.json"))

        # ----------------------------------------------------- log the MLflow model ---
        model.to("cpu").eval()
        info = log_model_compat(model, tok, max_len, [test["text"].iloc[0][:300]])
        model_uri = getattr(info, "model_uri", None) or f"runs:/{run.info.run_id}/model"
        mlflow.set_tag("logged_model_uri", model_uri)

        # Round-trip check: the logged model must reproduce the in-memory predictions.
        k = min(20, len(test))
        out = mlflow.pyfunc.load_model(model_uri).predict(test["text"].head(k).tolist())
        rt_labels, _ = parse_predictions(out)
        agree = float(np.mean([a == config.LABEL_NAMES[b] for a, b in zip(rt_labels, preds[:k])]))
        mlflow.log_metric("roundtrip_agreement_first20", agree)
        if agree < 1.0:
            print(f"WARNING: logged model agrees with in-memory model on only {agree:.0%} of {k} articles")

        print(json.dumps({"run_id": run.info.run_id, **m, **lat}, indent=2))
        config.OUT_DIR.mkdir(exist_ok=True)
        (config.OUT_DIR / f"run_{spec.key}{'_smoke' if args.smoke else ''}.json").write_text(
            json.dumps({"run_id": run.info.run_id, "model_uri": model_uri, **m, **lat}, indent=2))


if __name__ == "__main__":
    main()

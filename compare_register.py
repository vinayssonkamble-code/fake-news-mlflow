#!/usr/bin/env python
"""Step 3 - compare A and B in an MLflow summary run, apply the Champion rule, register both.

    python compare_register.py
    python compare_register.py --smoke

Reads both runs back from MLflow (not from local variables), refuses to compare them unless
they used the identical test set and identical hyper-parameters, applies config.RULE, logs the
`compare_fakenews` run, registers both models under one registered model name, and sets the
`champion` and `challenger` aliases.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow import MlflowClient

import config
from decision import select_champion
from metrics_utils import paired_bootstrap_delta
from mlflow_utils import git_commit

FAIR_PARAMS = ["max_len", "batch_size", "epochs", "learning_rate", "weight_decay",
               "warmup_ratio", "seed", "train_n", "val_n", "test_n"]
SHOWN = ["accuracy", "precision", "recall", "f1", "macro_f1",
         "avg_latency_ms", "p95_latency_ms", "total_inference_s"]


def latest_run(client, exp_id, run_name):
    runs = client.search_runs([exp_id], f"tags.mlflow.runName = '{run_name}' and attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit(f"No FINISHED run named {run_name!r}. Run train_evaluate.py first.")
    return runs[0]


def get_or_register(client, name, run):
    for mv in client.search_model_versions(f"name='{name}'"):
        if (mv.tags or {}).get("source_run_id") == run.info.run_id:
            return mv.version
    uri = run.data.tags.get("logged_model_uri") or f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(uri, name)
    client.set_model_version_tag(name, mv.version, "source_run_id", run.info.run_id)
    return mv.version


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    nm = config.names(args.smoke)

    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(nm.experiment)
    if exp is None:
        raise SystemExit(f"Experiment {nm.experiment!r} not found.")
    runs = {k: latest_run(client, exp.experiment_id, s.run_name) for k, s in config.MODELS.items()}
    A, B = runs["A"], runs["B"]

    # ---- fairness gate: identical test inputs and identical hyper-parameters -------------
    if A.data.tags.get("test_set_sha256") != B.data.tags.get("test_set_sha256"):
        raise SystemExit("ABORT: the two runs were evaluated on different test sets.")
    diff = [p for p in FAIR_PARAMS if A.data.params.get(p) != B.data.params.get(p)]
    if diff:
        raise SystemExit(f"ABORT: runs differ in hyper-parameters {diff}; comparison would be unfair.")

    ma, mb = ({m: r.data.metrics[m] for m in SHOWN} for r in (A, B))
    decision = select_champion(ma, mb, config.RULE)
    win, lose = decision["winner"], decision["loser"]
    spec_w, spec_l = config.MODELS[win], config.MODELS[lose]

    # informational paired bootstrap (NOT used by the rule)
    pa = pd.read_csv(client.download_artifacts(A.info.run_id, "test_predictions.csv"))
    pb = pd.read_csv(client.download_artifacts(B.info.run_id, "test_predictions.csv"))
    assert (pa["id"].values == pb["id"].values).all() and (pa["label"].values == pb["label"].values).all()
    boot = paired_bootstrap_delta(pa["label"].values, pa["pred"].values, pb["pred"].values, seed=config.SEED)

    table = pd.DataFrame({"metric": SHOWN, "A": [ma[m] for m in SHOWN], "B": [mb[m] for m in SHOWN]})
    table["B_minus_A"] = table["B"] - table["A"]

    mlflow.set_experiment(nm.experiment)
    with mlflow.start_run(run_name=config.SUMMARY_RUN_NAME) as run:
        mlflow.log_params({
            "primary_metric": config.RULE["primary_metric"], "margin": config.RULE["margin"],
            "tie_breaker": config.RULE["tie_breaker"], "incumbent": config.RULE["incumbent"],
            "run_id_A": A.info.run_id, "run_id_B": B.info.run_id,
            "rule_sha256": config.rule_fingerprint(), "git_commit": git_commit(config.ROOT),
        })
        mlflow.log_metrics({
            "delta_primary_metric": decision["delta_primary_metric"],
            **{f"A_{m}": ma[m] for m in SHOWN}, **{f"B_{m}": mb[m] for m in SHOWN},
            "bootstrap_delta_ci_low": boot["ci_low"], "bootstrap_delta_ci_high": boot["ci_high"],
        })
        mlflow.set_tags({
            "winner": config.MODELS[win].run_name, "winner_key": win, "decision_basis": decision["basis"],
            "decision_reason": decision["reason"], "task": "fake-news-classification",
            "smoke_test": str(args.smoke),
        })
        table.to_csv("comparison.csv", index=False)
        mlflow.log_artifact("comparison.csv")
        mlflow.log_dict({**decision, "bootstrap_informational": boot}, "decision.json")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150, gridspec_kw={"width_ratios": [3, 1.2]})
        qual = ["accuracy", "precision", "recall", "f1", "macro_f1"]
        x = range(len(qual))
        ax1.bar([i - 0.2 for i in x], [ma[m] for m in qual], 0.4, label=config.MODELS["A"].short)
        ax1.bar([i + 0.2 for i in x], [mb[m] for m in qual], 0.4, label=config.MODELS["B"].short)
        ax1.set_xticks(list(x), qual)
        ax1.set_ylim(max(0.0, min(min(ma[m], mb[m]) for m in qual) - 0.05), 1.0)
        ax1.set_title("Test-set quality metrics")
        ax1.legend()
        ax2.bar([config.MODELS["A"].short, config.MODELS["B"].short],
                [ma["avg_latency_ms"], mb["avg_latency_ms"]], color=["#4C78A8", "#F58518"])
        ax2.set_title("Avg latency (ms / article)")
        fig.tight_layout()
        fig.savefig("metric_comparison.png")
        plt.close(fig)
        mlflow.log_artifact("metric_comparison.png")

        # ---- registry: one registered model, two versions, two aliases --------------------
        v = {k: get_or_register(client, nm.registered_model, r) for k, r in runs.items()}
        client.set_registered_model_alias(nm.registered_model, "champion", v[win])
        client.set_registered_model_alias(nm.registered_model, "challenger", v[lose])
        for k, r in runs.items():
            client.set_model_version_tag(nm.registered_model, v[k], "role", "champion" if k == win else "challenger")
            client.set_model_version_tag(nm.registered_model, v[k], "hf_model_id", config.MODELS[k].hf_id)
            client.set_model_version_tag(nm.registered_model, v[k], "macro_f1", f"{r.data.metrics['macro_f1']:.4f}")
        mlflow.set_tags({"champion_version": str(v[win]), "challenger_version": str(v[lose])})

        result = {"summary_run_id": run.info.run_id, "registered_model": nm.registered_model,
                  "versions": v, "champion": spec_w.run_name, "challenger": spec_l.run_name,
                  "champion_uri": f"models:/{nm.registered_model}@champion", **decision,
                  "bootstrap_informational": boot}
        config.OUT_DIR.mkdir(exist_ok=True)
        (config.OUT_DIR / f"decision{'_smoke' if args.smoke else ''}.json").write_text(json.dumps(result, indent=2))

    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nDECISION:", decision["reason"])
    print(f"Champion  : {spec_w.run_name} (version {v[win]})")
    print(f"Challenger: {spec_l.run_name} (version {v[lose]})")
    print(f"Prediction URI: models:/{nm.registered_model}@champion")


if __name__ == "__main__":
    main()

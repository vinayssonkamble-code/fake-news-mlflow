# Fake News Detection System using Hugging Face and MLflow

Champion–Challenger comparison of two fine-tuned Hugging Face transformers on WELFake,
tracked and registered with MLflow.  Student: Vinaykumar (1MS25CI146).

## What has and has not been verified

| Item | Status |
|---|---|
| Cleaning, de-duplication, splitting, metrics, decision rule, output parsing (`tests/`) | 18 unit tests pass (synthetic data) |
| All scripts compile | yes |
| Fine-tuning, MLflow logging, registry aliases, prediction through `@champion` | **written but not executed** – torch, transformers and mlflow could not be installed where this code was written. Run the smoke test first. |
| Any metric, latency, run ID, version or prediction | **none exist yet** – they are produced only when you run the pipeline |

## Design

* Task: binary classification, `FAKE` = 1 (positive class), `REAL` = 0.
* Data: WELFake (Zenodo 4561253, CC BY 4.0).  Published labels: 0 = fake, 1 = real.
* Models (same hyper-parameters, seed, data, latency protocol):
  A `distilbert-base-uncased` (initial Champion) and B `distilroberta-base` (Challenger).
* Leakage control: rows are cleaned and near-duplicates (first 300 normalised characters)
  are removed – label-conflicting groups entirely – **before** the stratified split;
  splits are checked for id and text overlap.  The validation split picks the best epoch; the test
  split is used once for the final comparison.
* Rule (in `config.py`, commit before the final run): primary metric macro-F1, margin 0.01,
  tie-breaker lower average latency, incumbent A on an exact tie.
* `new_data.csv` holds 10 articles drawn from the same cleaned pool but excluded from all splits;
  their labels are kept in `data/held_back/` and are not used by the prediction script.

## Run (Windows PowerShell, CPU only)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# download WELFake_Dataset.csv from https://zenodo.org/record/4561253 into data\raw\
python prepare_data.py --inspect        # read the samples; confirm which label is FAKE
python prepare_data.py                  # add --flip-labels only if --inspect shows inverted labels

# terminal 2 - leave running
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

# 0. optional but recommended: tiny end-to-end check in a separate experiment/registry
python train_evaluate.py --model A --smoke
python train_evaluate.py --model B --smoke
python compare_register.py --smoke
python predict_from_mlflow.py --smoke

# 1. commit config.py (the rule), then the real run
git add -A; git commit -m "Fix Champion rule before final run"
python train_evaluate.py --model A
python train_evaluate.py --model B
python compare_register.py
python predict_from_mlflow.py
python predict_from_mlflow.py --with-labels   # optional sanity check against held-back labels
```

MLflow UI: http://127.0.0.1:5000.  Outputs: `outputs/new_data_predictions.csv`,
`outputs/decision.json`, `outputs/run_A.json`, `outputs/run_B.json`.

## CPU / memory notes (16 GB RAM)

Both models are 6-layer encoders (about 66–67 M and 82 M parameters).  Defaults – 3,000 training
articles, 2 epochs, batch size 16, 256 tokens – are chosen to fit a laptop CPU; articles longer than
256 tokens are truncated.  If memory is tight, lower `BATCH_SIZE` in `config.py` (identically for
both models).  Training time depends on your CPU and is not estimated here.

## Known limitations

* Label direction in WELFake must be confirmed with `--inspect`.
* WELFake merges several sources; models can learn source style rather than veracity, so test
  scores will not transfer to other publishers or time periods.
* Latency is batch-size-1 on one machine, includes tokenisation, and is noisy under system load.
* A 1,000-article test set cannot separate models that differ by less than about 1 point of F1.

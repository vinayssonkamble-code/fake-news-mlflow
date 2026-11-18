# Fake News Detection: Champion–Challenger Model Comparison

A binary fake-news classification project using Hugging Face Transformers and MLflow. Two models are fine-tuned on the same WELFake dataset, evaluated on the same test set, compared using predefined metrics, and registered in MLflow.

## 1. Project Objective

Compare two transformer models:

- Model A: DistilBERT (`distilbert-base-uncased`) — initial Champion.
- Model B: DistilRoBERTa (`distilroberta-base`) — Challenger.

The objective is to select the better model using a predefined decision rule and make predictions using the registered Champion.

## 2. Dataset

- Dataset: WELFake
- Source: https://zenodo.org/records/4561253
- License: CC BY 4.0
- Task: Binary classification.
- Label mapping: `0 = REAL`, `1 = FAKE`.

The dataset was cleaned, deduplicated and split before training.

| Split | Samples |
|---|---:|
| Training | 3,000 |
| Validation | 600 |
| Testing | 1,000 |
| Unseen prediction data | 10 |

The unseen articles are excluded from training, validation and testing. Their labels are held separately and are not used by the prediction script.

## 3. Model Configuration

Both models use the same training data, evaluation data and training configuration.

| Parameter | Value |
|---|---|
| Maximum sequence length | 128 |
| Batch size | 8 |
| Epochs | 1 |
| Learning rate | 0.00002 |
| Weight decay | 0.01 |
| Random seed | 42 |
| Device | CPU |
| Evaluation batch size | 32 |

## 4. Evaluation Results

Both models completed full training and evaluation on the 1,000-sample test set.

| Metric | Model A: DistilBERT | Model B: DistilRoBERTa |
|---|---:|---:|
| Accuracy | 0.9630 | 0.9870 |
| Precision | 0.9799 | 0.9808 |
| Recall | 0.9538 | 0.9964 |
| F1-score | 0.9667 | 0.9885 |
| Macro-F1 | 0.9625 | 0.9868 |
| Average latency (ms) | 103.8382 | 80.4251 |
| P95 latency (ms) | 174.4487 | 92.8839 |

## 5. Champion Selection Rule

The selection rule was defined before the final experiment.

1. Primary metric: Macro-F1.
2. Challenger must improve Macro-F1 by at least 0.01 to replace the incumbent.
3. If the improvement is below the required margin, average inference latency is used as the tie-breaker.
4. If both metrics are tied, the incumbent remains Champion.

Calculation:

`Model B Macro-F1 - Model A Macro-F1 = 0.9868 - 0.9625 = 0.0242`

Since the improvement of 0.0242 is greater than the required margin of 0.01, Model B is selected.

**Final decision: Model B (DistilRoBERTa) is the Champion.**

## 6. MLflow Model Registry

Registered model name: `fake-news-classifier`

| Model role | Model | Registry version | Alias |
|---|---|---:|---|
| Champion | DistilRoBERTa | 2 | `champion` |
| Challenger | DistilBERT | 1 | `challenger` |

Champion model URI:

`models:/fake-news-classifier@champion`

MLflow tracks the experiment parameters, evaluation metrics, model artifacts, predictions and comparison results.

## 7. Unseen Data Prediction

The prediction script loads the registered Champion using its MLflow alias.

Command:

`python predict_from_mlflow.py`

The script successfully loaded version 2 (`distilroberta-base`) and generated predictions for 10 unseen articles.

Output file:

`outputs/new_data_predictions.csv`

Each prediction includes the article ID, predicted label, confidence and Champion version.

## 8. Project Structure

```text
fake-news-mlflow/
├── data/
│   ├── raw/
│   ├── held_back/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── new_data.csv
├── outputs/
│   ├── decision.json
│   ├── run_A.json
│   ├── run_B.json
│   └── new_data_predictions.csv
├── tests/
├── config.py
├── data_utils.py
├── decision.py
├── metrics_utils.py
├── mlflow_utils.py
├── prepare_data.py
├── train_evaluate.py
├── compare_register.py
├── predict_from_mlflow.py
├── requirements.txt
└── README.md
```

## 9. How to Run

Use Python and install the dependencies listed in `requirements.txt`.

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start the MLflow server in a separate terminal:

```powershell
.\.venv\Scripts\python.exe -m mlflow server --host 127.0.0.1 --port 5000 --workers 1
```

Run the full experiment in the project directory:

```powershell
.\.venv\Scripts\python.exe .\train_evaluate.py --model A --threads 2
.\.venv\Scripts\python.exe .\train_evaluate.py --model B --threads 2
.\.venv\Scripts\python.exe .\compare_register.py
.\.venv\Scripts\python.exe .\predict_from_mlflow.py
```

Open the MLflow interface:

http://127.0.0.1:5000

## 10. Limitations

- The experiment uses a 1,000-sample test set from WELFake.
- Dataset-specific patterns may affect model performance; high test scores do not guarantee performance on every news source.
- Inference latency depends on the machine and system load.
- The experiment was conducted on CPU, with one training epoch.
- The 10 unseen predictions demonstrate inference through the registered Champion; they are not a replacement for an independent external evaluation.

## 11. Technologies Used

- Python
- Hugging Face Transformers
- PyTorch
- MLflow
- Pandas
- NumPy
- Matplotlib
- Git and GitHub
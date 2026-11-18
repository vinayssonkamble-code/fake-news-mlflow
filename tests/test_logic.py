"""Unit tests for the parts that do not need torch / transformers / mlflow.

    python -m unittest discover -s tests -v
"""
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_utils import assert_no_leakage, clean_and_dedupe, make_splits  # noqa: E402
from decision import select_champion  # noqa: E402
from metrics_utils import compute_metrics, paired_bootstrap_delta, summarise_latency  # noqa: E402
from mlflow_utils import parse_predictions  # noqa: E402


def synthetic_raw(n=600, seed=0):
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(5000)]
    rows = []
    for i in range(n):
        body = " ".join(rng.choice(words, 60))
        rows.append({"Unnamed: 0": i, "title": f"title {i}", "text": body, "label": int(i % 2)})
    return pd.DataFrame(rows)


class CleanAndSplit(unittest.TestCase):
    def test_label_mapping_fake_is_positive(self):
        raw = synthetic_raw(40)
        clean, _ = clean_and_dedupe(raw, fake_value=0, min_words=5)
        merged = clean.merge(raw, left_on="id", right_on="Unnamed: 0")
        self.assertTrue(((merged["label_x"] == 1) == (merged["label_y"] == 0)).all())
        self.assertTrue((clean.loc[clean.label == 1, "label_name"] == "FAKE").all())

    def test_duplicates_conflicts_missing_and_short_removed(self):
        raw = synthetic_raw(20)
        dup = raw.iloc[[0]].copy(); dup["Unnamed: 0"] = 900                       # exact duplicate
        conf = raw.iloc[[1]].copy(); conf["Unnamed: 0"] = 901; conf["label"] = 1 - conf["label"].iloc[0]
        nan_text = raw.iloc[[2]].copy(); nan_text["Unnamed: 0"] = 902; nan_text["text"] = np.nan
        short = raw.iloc[[3]].copy(); short["Unnamed: 0"] = 903; short["text"] = "too short"
        bad_label = raw.iloc[[4]].copy(); bad_label["Unnamed: 0"] = 904; bad_label["label"] = 7
        allr = pd.concat([raw, dup, conf, nan_text, short, bad_label], ignore_index=True)
        clean, st = clean_and_dedupe(allr, min_words=30)
        self.assertEqual(st["n_bad_label"], 1)
        self.assertEqual(st["n_too_short_or_empty"], 2)
        self.assertEqual(st["n_label_conflict_rows"], 2)       # both copies of the conflicting pair
        self.assertEqual(st["n_duplicate_rows_removed"], 1)
        self.assertNotIn(1, set(clean["id"]))                  # conflicting group dropped entirely
        self.assertEqual(clean["fp"].nunique(), len(clean))

    def test_splits_disjoint_sized_stratified_deterministic(self):
        clean, _ = clean_and_dedupe(synthetic_raw(600), min_words=5)
        sizes = {"test": 100, "val": 60, "new": 10, "train": 200}
        s1, s2 = make_splits(clean, sizes, 42), make_splits(clean, sizes, 42)
        for k, n in sizes.items():
            self.assertEqual(len(s1[k]), n)
            self.assertAlmostEqual(s1[k]["label"].mean(), clean["label"].mean(), delta=0.06)
            pd.testing.assert_frame_equal(s1[k], s2[k])
        assert_no_leakage(s1)
        all_ids = pd.concat([s["id"] for s in s1.values()])
        self.assertEqual(all_ids.nunique(), len(all_ids))

    def test_leakage_detector_fires(self):
        clean, _ = clean_and_dedupe(synthetic_raw(100), min_words=5)
        sp = make_splits(clean, {"test": 10, "val": 10, "new": 10, "train": 20}, 1)
        sp["train"] = pd.concat([sp["train"], sp["test"].head(1)], ignore_index=True)
        with self.assertRaises(AssertionError):
            assert_no_leakage(sp)

    def test_too_many_rows_requested(self):
        clean, _ = clean_and_dedupe(synthetic_raw(50), min_words=5)
        with self.assertRaises(ValueError):
            make_splits(clean, {"test": 40, "val": 40, "new": 10, "train": 40}, 1)

    def test_prepare_data_cli_end_to_end(self):
        """Run prepare_data.py on a synthetic CSV, with the data dir redirected to a temp folder."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw = synthetic_raw(600).rename(columns={"Unnamed: 0": ""})
            raw.to_csv(td / "raw.csv", index=False)
            code = (
                "import sys, config, prepare_data\n"
                f"from pathlib import Path\n"
                f"config.DATA_DIR = Path(r'{td}/data'); config.HELD_BACK_DIR = config.DATA_DIR/'held_back'\n"
                "config.SPLIT_SIZES = {'test': 100, 'val': 60, 'new': 10, 'train': 200}\n"
                "config.MIN_WORDS = 5\n"
                f"sys.argv = ['prepare_data.py', '--raw', r'{td}/raw.csv']\n"
                "sys.exit(prepare_data.main())\n")
            r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            d = td / "data"
            for n, k in (("train", 200), ("val", 60), ("test", 100)):
                self.assertEqual(len(pd.read_csv(d / f"{n}.csv")), k)
            new = pd.read_csv(d / "new_data.csv")
            self.assertEqual(list(new.columns), ["id", "text"])       # no labels leak into new_data
            self.assertEqual(len(new), 10)
            self.assertTrue((d / "held_back" / "new_data_labels.csv").exists())
            man = json.loads((d / "split_manifest.json").read_text())
            self.assertEqual(man["class_counts"]["test"]["REAL"] + man["class_counts"]["test"]["FAKE"], 100)
            ids = [set(pd.read_csv(d / f"{n}.csv")["id"]) for n in ("train", "val", "test")] + [set(new["id"])]
            for i in range(4):
                for j in range(i + 1, 4):
                    self.assertFalse(ids[i] & ids[j])


class Metrics(unittest.TestCase):
    def test_compute_metrics_hand_checked(self):
        # truth: 4 FAKE(1), 4 REAL(0); preds: TP=3 FN=1 FP=2 TN=2
        y = [1, 1, 1, 1, 0, 0, 0, 0]
        p = [1, 1, 1, 0, 1, 1, 0, 0]
        m = compute_metrics(y, p)
        self.assertAlmostEqual(m["accuracy"], 5 / 8)
        self.assertAlmostEqual(m["precision"], 3 / 5)
        self.assertAlmostEqual(m["recall"], 3 / 4)
        self.assertAlmostEqual(m["f1"], 2 * 0.6 * 0.75 / (0.6 + 0.75))
        f_real = 2 * (2 / 3) * (2 / 4) / ((2 / 3) + (2 / 4))
        self.assertAlmostEqual(m["macro_f1"], (m["f1"] + f_real) / 2)

    def test_latency_summary(self):
        s = summarise_latency([0.01, 0.02, 0.03])
        self.assertAlmostEqual(s["avg_latency_ms"], 20.0)
        self.assertAlmostEqual(s["total_inference_s"], 0.06)

    def test_bootstrap_identical_models_centred_on_zero(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 300)
        p = np.where(rng.random(300) < 0.9, y, 1 - y)
        b = paired_bootstrap_delta(y, p, p, n_boot=200)
        self.assertEqual((b["ci_low"], b["ci_high"]), (0.0, 0.0))


class Rule(unittest.TestCase):
    def mk(self, f1, lat):
        return {"macro_f1": f1, "avg_latency_ms": lat}

    def test_b_wins_by_margin_even_if_slower(self):
        d = select_champion(self.mk(0.90, 10), self.mk(0.92, 50))
        self.assertEqual((d["winner"], d["basis"]), ("B", "primary_metric"))

    def test_exact_margin_counts_as_win(self):
        d = select_champion(self.mk(0.90, 10), self.mk(0.91, 50))
        self.assertEqual((d["winner"], d["basis"]), ("B", "primary_metric"))

    def test_a_wins_by_margin(self):
        d = select_champion(self.mk(0.95, 50), self.mk(0.93, 10))
        self.assertEqual((d["winner"], d["basis"]), ("A", "primary_metric"))

    def test_within_margin_latency_decides(self):
        self.assertEqual(select_champion(self.mk(0.90, 30), self.mk(0.905, 12))["winner"], "B")
        self.assertEqual(select_champion(self.mk(0.905, 12), self.mk(0.90, 30))["winner"], "A")
        d = select_champion(self.mk(0.90, 12), self.mk(0.909, 30))
        self.assertEqual((d["winner"], d["basis"]), ("A", "latency_tiebreak"))

    def test_slightly_below_margin_is_not_a_win(self):
        d = select_champion(self.mk(0.90, 10), self.mk(0.9099, 50))
        self.assertEqual(d["basis"], "latency_tiebreak")
        self.assertEqual(d["winner"], "A")

    def test_exact_tie_keeps_incumbent(self):
        d = select_champion(self.mk(0.9, 20), self.mk(0.9, 20))
        self.assertEqual((d["winner"], d["basis"]), ("A", "incumbent_tie"))


class OutputParsing(unittest.TestCase):
    def test_dataframe_and_list_forms(self):
        df = pd.DataFrame({"label": ["FAKE", "REAL"], "score": [0.9, 0.8]})
        self.assertEqual(parse_predictions(df), (["FAKE", "REAL"], [0.9, 0.8]))
        self.assertEqual(parse_predictions([{"label": "REAL", "score": 0.7}]), (["REAL"], [0.7]))

    def test_unknown_format_fails_loudly(self):
        with self.assertRaises(ValueError):
            parse_predictions(pd.DataFrame({"x": [1]}))
        with self.assertRaises(ValueError):
            parse_predictions("FAKE")


class PredictionScriptContract(unittest.TestCase):
    def test_predict_script_loads_only_via_alias_and_never_imports_hf(self):
        src = (ROOT / "predict_from_mlflow.py").read_text()
        imported = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertFalse(imported & {"torch", "transformers"})
        self.assertIn('@champion', src)
        for spec_id in ("distilbert-base-uncased", "distilroberta-base"):
            self.assertNotIn(spec_id, src)


if __name__ == "__main__":
    unittest.main()

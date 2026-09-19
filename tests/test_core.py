import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
from crackzoo.metrics import binary_metrics, binarize, grid_mask, grid_metrics, boundary_f1
from crackzoo.data_audit import audit_dataset
from crackzoo.registry import command_for

ROOT = Path(__file__).resolve().parents[1]


class MetricsTests(unittest.TestCase):
    def test_hand_calculated_confusion(self):
        m = binary_metrics([[1, 1], [0, 0]], [[1, 0], [1, 0]])
        for key in ("tp", "fp", "fn", "tn"):
            self.assertEqual(m[key], 1)
        self.assertAlmostEqual(m["iou"], 1/3)
        self.assertAlmostEqual(m["dice"], .5)

    def test_encoding_and_probability_threshold(self):
        self.assertTrue(np.array_equal(binarize([[0, 255]]), binarize([[0, 1]])))
        self.assertTrue(np.array_equal(binarize([[.6, .9]], .8), [[False, True]]))

    def test_invalid_masks(self):
        for mask in ([], [[[1]]], [[float("nan")]], [[-1]], [[256]]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                binarize(mask)
        with self.assertRaises(ValueError):
            binary_metrics([[1]], [[1, 0]])

    def test_grid_partial_edges(self):
        mask = np.zeros((3, 3)); mask[2, 2] = 1
        self.assertTrue(np.array_equal(grid_mask(mask, 2, .5), [[0, 0], [0, 1]]))
        for size in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                grid_mask(mask, size)
        with self.assertRaises(ValueError):
            grid_metrics(np.zeros((3, 3)), np.zeros((4, 4)), 16)

    def test_tolerance_geometry_and_zero_radius(self):
        gt = np.zeros((5, 5)); gt[2, 2] = 1
        pred = np.zeros((5, 5)); pred[3, 3] = 1
        self.assertAlmostEqual(boundary_f1(gt, pred, 1), 1, places=6)
        self.assertEqual(boundary_f1(gt, pred, 0), 0)
        with self.assertRaises(ValueError):
            boundary_f1(gt, pred, -1)

    def test_empty_foreground_convention(self):
        empty = np.zeros((2, 2))
        self.assertEqual(binary_metrics(empty, empty)["iou"], 0)
        self.assertEqual(boundary_f1(empty, empty), 1)


class IntegrationTests(unittest.TestCase):
    def test_no_silent_missing_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"gt").mkdir(); (root/"pred").mkdir()
            for stem in ("a", "b"):
                Image.fromarray(np.zeros((4, 4), dtype="uint8")).save(root/"gt"/f"{stem}.png")
            Image.fromarray(np.zeros((4, 4), dtype="uint8")).save(root/"pred/a.png")
            result = subprocess.run([sys.executable, str(ROOT/"scripts/evaluate_masks.py"), "--gt-dir", str(root/"gt"), "--pred-dir", str(root/"pred"), "--out-csv", str(root/"out.csv")], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root/"out.csv").exists())

    def test_demo_and_split_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"demo"
            subprocess.run([sys.executable, str(ROOT/"scripts/demo.py"), "--output", str(out)], check=True, capture_output=True)
            self.assertTrue(json.loads((out/"dataset_audit.json").read_text())["ok"])
            src = out/"dataset/train/images/synthetic_train_00.png"
            dst = out/"dataset/test/images/synthetic_test_00.png"
            dst.write_bytes(src.read_bytes())
            self.assertFalse(audit_dataset(out/"dataset")["ok"])

    def test_command_paths_and_eval_entry(self):
        command = command_for("mcca", "eval", dataset="data/example", checkpoint="models/test.pth")
        self.assertEqual(command.argv[0], sys.executable)
        self.assertIn(str((Path.cwd()/"models/test.pth").resolve()), command.argv)
        self.assertEqual(command_for("crackresunet", "eval").argv[1], "evaluate.py")
        with self.assertRaises(ValueError):
            command_for("mcca", "ablation")


if __name__ == "__main__":
    unittest.main()

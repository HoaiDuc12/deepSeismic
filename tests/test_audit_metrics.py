import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facies.audit import audit_dataset, save_audit
from facies.data import Record, load_classes, load_mask, load_records
from facies.evaluate import evaluate_records, save_evaluation, select_evaluation_records
from facies.metrics import boundary_counts, boundary_metrics, confusion_matrix, confusion_metrics


ROOT = Path(__file__).resolve().parents[1]
CLASSES = load_classes(ROOT / "classes.json")


class AuditMetricsTests(unittest.TestCase):
    def make_dataset(self, path, *, copies=False, bad_mask=False, same_group=False, all_ignore=False):
        rows = [["image", "mask", "group", "split"]]
        for index, split in enumerate(("train", "val", "test")):
            image = np.full((25, 35, 3), 60 + (0 if copies else index), dtype=np.uint8)
            mask = np.zeros((25, 35), dtype=np.uint8)
            mask[3:19, 5:28] = 2
            mask[0:2] = 255
            if bad_mask and index == 1:
                mask[5, 5] = 29
            if all_ignore and index == 1:
                mask[:] = 255
            Image.fromarray(image).save(path / f"ảnh-{index}.png")
            Image.fromarray(mask).save(path / f"mask-{index}.png")
            group = "line" if same_group else f"line-{index}"
            rows.append([f"ảnh-{index}.png", f"mask-{index}.png", group, split])
        manifest = path / "dataset.csv"
        with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            csv.writer(handle).writerows(rows)
        return manifest

    def test_confusion_ignore_and_analytic_scores(self):
        truth = np.array([[0, 1, 1], [2, 2, 255]], dtype=np.uint8)
        pred = np.array([[0, 1, 2], [2, 0, 6]], dtype=np.uint8)
        matrix = confusion_matrix(truth, pred, len(CLASSES))
        scores = confusion_metrics(matrix, CLASSES)
        self.assertEqual(matrix.sum(), 5)
        self.assertAlmostEqual(scores["pixel_accuracy"], 3 / 5)
        self.assertAlmostEqual(scores["classes"][1]["precision"], 1)
        self.assertAlmostEqual(scores["classes"][1]["recall"], 0.5)
        self.assertAlmostEqual(scores["classes"][1]["f1"], 2 / 3)
        self.assertAlmostEqual(scores["classes"][2]["iou"], 1 / 3)
        self.assertIsNone(scores["classes"][6]["iou"])

    def test_false_only_class_in_union_average(self):
        matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
        matrix[1, 1] = 10
        matrix[0, 2] = 10
        scores = confusion_metrics(matrix, CLASSES)
        self.assertEqual(scores["mean_iou_facies"], 1)
        self.assertEqual(scores["mean_iou_facies_union"], 0.5)
        self.assertEqual(scores["classes"][2]["precision"], 0)
        self.assertIsNone(scores["classes"][2]["recall"])

    def test_boundary_tolerance_and_unknown_margins(self):
        truth = np.zeros((50, 70), dtype=np.uint8)
        truth[12:36, 17:48] = 2
        shifted = np.roll(truth, 1, axis=1)
        exact = boundary_metrics(boundary_counts(truth, shifted, len(CLASSES), tolerance=0), CLASSES, 0)
        tolerant = boundary_metrics(boundary_counts(truth, shifted, len(CLASSES), tolerance=1), CLASSES, 1)
        self.assertLess(exact["boundary_classes"][2]["boundary_f1"], 1)
        self.assertEqual(tolerant["boundary_classes"][2]["boundary_f1"], 1)
        truth[:] = 255
        truth[10:40, 10:60] = 2
        correct = np.full_like(truth, 2)
        counts = boundary_counts(truth, correct, len(CLASSES), tolerance=2)
        self.assertEqual(counts.sum(), 0, "Unlabeled margins must not become target boundaries")
        solid = np.full((30, 40), 2, dtype=np.uint8)
        self.assertEqual(boundary_counts(solid, solid, len(CLASSES)).sum(), 0)

    def test_audit_reports_coverage_without_promising_label_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = audit_dataset(self.make_dataset(root), ROOT / "classes.json")
            self.assertTrue(report["ready_for_training"], report["errors"])
            self.assertEqual(report["status"], "warning")
            self.assertEqual(report["summary"]["valid_sections"], 3)
            self.assertEqual(report["splits"]["val"]["classes"][2]["section_count"], 1)
            self.assertAlmostEqual(report["splits"]["train"]["ignore_fraction"], 2 / 25)
            self.assertIn(1, report["summary"]["missing_class_ids"])
            paths = save_audit(report, root / "reports")
            self.assertEqual(json.loads(paths["json"].read_text(encoding="utf-8"))["schema_version"], 1)

    def test_audit_exact_hash_duplicates_and_group_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = audit_dataset(self.make_dataset(root, copies=True, same_group=True), ROOT / "classes.json")
            codes = {row["code"] for row in report["errors"]}
            self.assertFalse(report["ready_for_training"])
            self.assertIn("duplicate_cross_split", codes)
            self.assertIn("group_leakage", codes)
            self.assertTrue(report["duplicates"][0]["cross_split"])

    def test_audit_keeps_other_sections_after_bad_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = audit_dataset(self.make_dataset(root, bad_mask=True), ROOT / "classes.json")
            self.assertFalse(report["ready_for_training"])
            self.assertEqual(report["summary"]["valid_sections"], 2)
            self.assertIn("mask_invalid", {row["code"] for row in report["errors"]})
            report = audit_dataset(self.make_dataset(root, all_ignore=True), ROOT / "classes.json")
            self.assertEqual(report["splits"]["val"]["ignore_fraction"], 1)
            self.assertFalse(report["ready_for_training"])

    def test_evaluation_requires_named_independent_holdout(self):
        record = Record(Path("x"), Path("m"), "line", "train")
        with patch("facies.evaluate.image_sha256", return_value="same"):
            with self.assertRaises(ValueError):
                select_evaluation_records([record], "test")
            with self.assertRaises(ValueError):
                select_evaluation_records([record, Record(Path("y"), Path("m"), "other", "test")], "test")
        with self.assertRaises(ValueError):
            select_evaluation_records([Record(Path("x"), Path("m"), "line")], "val")

    def test_machine_proposals_require_per_shape_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            Image.new("RGB", (30, 20)).save(image)
            annotation_path = root / "labelme.json"
            record = Record(image, annotation_path, "line")
            shapes = [
                {"label": "parallel", "shape_type": "rectangle", "points": [[2, 2], [10, 15]],
                 "flags": {"machine_prediction": True, "reviewed": True}},
                {"label": "chaotic", "shape_type": "rectangle", "points": [[18, 2], [27, 15]],
                 "flags": {"machine_prediction": True, "reviewed": False}},
            ]
            annotation = {"imageWidth": 30, "imageHeight": 20,
                          "flags": {"machine_prediction": True, "expert_edits": True, "reviewed": True},
                          "shapes": shapes}
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
            mask = load_mask(record, CLASSES)
            self.assertEqual(mask[5, 5], 2)
            self.assertEqual(mask[5, 22], 255)
            self.assertEqual(mask[0, 0], 255)
            for false_review in (False, None, "true", 1):
                shapes[0]["flags"]["reviewed"] = false_review
                annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
                self.assertTrue(np.all(load_mask(record, CLASSES) == 255))
            # A shape-level machine flag remains authoritative in an ordinary expert file.
            annotation["flags"] = {}
            shapes[0]["flags"] = {}
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
            mask = load_mask(record, CLASSES)
            self.assertEqual(mask[5, 5], 2)
            self.assertEqual(mask[5, 22], 255)
            # A machine document also requires review on shapes with no machine flag.
            annotation["flags"] = {"machine_prediction": True}
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
            self.assertTrue(np.all(load_mask(record, CLASSES) == 255))

    def test_exported_unreviewed_machine_polygons_are_not_training_labels(self):
        from facies.render import labelme_prediction
        from facies.train import _audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            Image.new("RGB", (35, 25)).save(image)
            mask = np.full((25, 35), 255, dtype=np.uint8)
            mask[4:20, 5:28] = 2
            annotation = labelme_prediction(mask, np.ones_like(mask, dtype=np.float32), CLASSES)
            self.assertTrue(annotation["shapes"])
            annotation_path = root / "labelme.json"
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
            record = Record(image, annotation_path, "line")
            self.assertTrue(np.all(load_mask(record, CLASSES) == 255))
            with self.assertRaisesRegex(ValueError, "không có pixel"):
                _audit([record], CLASSES)

    def test_evaluate_saves_per_class_and_per_section(self):
        class PerfectPredictor:
            classes = CLASSES
            def predict(self, image, **kwargs):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[3:19, 5:28] = 2
                return mask, np.ones(mask.shape, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = select_evaluation_records(load_records(self.make_dataset(root)), "test")
            report, sections = evaluate_records(PerfectPredictor(), records)
            self.assertEqual(report["mean_iou_facies"], 1)
            self.assertEqual(report["mean_boundary_f1_facies"], 1)
            paths = save_evaluation(report, sections, root / "evaluation")
            with paths["sections"].open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(CLASSES))
            self.assertIn("boundary_f1", rows[0])
            self.assertEqual(rows[2]["f1"], "1.0")


if __name__ == "__main__":
    unittest.main()

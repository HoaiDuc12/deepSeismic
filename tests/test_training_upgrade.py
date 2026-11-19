"""Independent checks for checkpoint continuation and guarded training inputs."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from facies.data import load_classes, load_records, split_records
from facies.experiment import dataset_fingerprint
from facies.model import build_model
from facies.predict import configure_device
from facies.train import segmentation_loss


ROOT = Path(__file__).resolve().parents[1]


def make_dataset(root, *, duplicate=False):
    rows = [["image", "mask", "group", "split"]]
    rng = np.random.default_rng(32)
    first = None
    for index, split in enumerate(("train", "val", "test")):
        image = rng.integers(20, 235, (35, 41, 3), dtype=np.uint8)
        if first is None:
            first = image.copy()
        if duplicate and index == 1:
            image = first
        mask = np.zeros((35, 41), dtype=np.uint8)
        mask[4:29, 6:34] = 2
        mask[:2] = 255
        Image.fromarray(image).save(root / f"section-{index}.png")
        Image.fromarray(mask).save(root / f"mask-{index}.png")
        rows.append([f"section-{index}.png", f"mask-{index}.png", f"line-{index}", split])
    path = root / "dataset.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return path


def run_train(manifest, out, epochs, extra=()):
    command = [sys.executable, "-X", "utf8", "-m", "facies.train",
               "--manifest", str(manifest), "--classes", str(ROOT / "classes.json"),
               "--out", str(out), "--epochs", str(epochs), "--width", "4", "--tile-size", "32",
               "--patches-per-image", "2", "--batch-size", "1", "--device", "cpu",
               "--cpu-threads", "1", "--patience", "0", "--amp", "off", *map(str, extra)]
    return subprocess.run(command, cwd=ROOT, capture_output=True, encoding="utf-8", timeout=90)


class TrainingUpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_device("cpu", 1)

    def test_both_architectures_preserve_odd_section_shape(self):
        image = torch.randn(1, 3, 35, 41)
        for name in ("unet", "attention_unet"):
            with self.subTest(architecture=name):
                model = build_model(name, 7, width=4)
                result = model(image)
                self.assertEqual(tuple(result.shape), (1, 7, 35, 41))
                self.assertTrue(torch.isfinite(result).all())
                result.square().mean().backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_all_ignore_and_boundary_loss_have_zero_unknown_gradient(self):
        logits = torch.randn(2, 7, 4, 6, requires_grad=True)
        all_ignore = torch.full((2, 4, 6), 255, dtype=torch.long)
        loss = segmentation_loss(logits, all_ignore, torch.ones(7), boundary_weight=2)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)
        logits = torch.randn(1, 7, 3, 4, requires_grad=True)
        target = torch.tensor([[[255, 0, 2, 2], [255, 0, 2, 2], [255, 0, 2, 2]]])
        segmentation_loss(logits, target, torch.ones(7), boundary_weight=3).backward()
        self.assertEqual(torch.count_nonzero(logits.grad[:, :, :, 0]).item(), 0)
        self.assertGreater(torch.count_nonzero(logits.grad[:, :, :, 1:]).item(), 0)

    def test_duplicate_content_cannot_train_across_splits(self):
        with tempfile.TemporaryDirectory(prefix="facies-leakage-") as directory:
            root = Path(directory)
            result = run_train(make_dataset(root, duplicate=True), root / "run", 1)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("trùng nội dung", result.stderr)
            self.assertFalse((root / "run" / "best.pt").exists())

    def test_dataset_identity_changes_when_expert_mask_changes(self):
        with tempfile.TemporaryDirectory(prefix="facies-label-identity-") as directory:
            root = Path(directory)
            parts = dict(zip(("train", "validation", "test"), split_records(load_records(make_dataset(root)))))
            classes = load_classes(ROOT / "classes.json")
            before = dataset_fingerprint(parts, classes)
            mask_path = root / "mask-0.png"
            with Image.open(mask_path) as mask:
                changed = np.array(mask)
            changed[8, 8] = 1
            Image.fromarray(changed).save(mask_path)
            self.assertNotEqual(before, dataset_fingerprint(parts, classes))

    def test_resume_matches_uninterrupted_exactly_and_records_identity(self):
        with tempfile.TemporaryDirectory(prefix="facies-resume-") as directory:
            root = Path(directory)
            manifest = make_dataset(root)
            continued, uninterrupted = root / "continued", root / "uninterrupted"
            for destination, epochs, extra in (
                (continued, 1, ()),
                (continued, 2, ("--resume", continued / "last.pt")),
                (uninterrupted, 2, ()),
            ):
                result = run_train(manifest, destination, epochs, extra)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            resumed = torch.load(continued / "last.pt", map_location="cpu", weights_only=True)
            original = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(resumed["dataset_sha256"], original["dataset_sha256"])
            self.assertEqual(resumed["active_ids"], [0, 2])
            self.assertEqual(resumed["arch"], "unet")
            for key in original["state"]:
                self.assertTrue(torch.equal(resumed["state"][key], original["state"][key]), key)
            for left, right in zip(resumed["history"], original["history"]):
                for key in ("epoch", "loss", "val_miou_facies", "val_miou_union", "lr"):
                    self.assertEqual(left[key], right[key], key)
            evaluation = root / "evaluation"
            result = subprocess.run([
                sys.executable, "-X", "utf8", "-m", "facies.evaluate", "--manifest", str(manifest),
                "--model", str(continued / "best.pt"), "--split", "test", "--out", str(evaluation),
                "--device", "cpu", "--batch-size", "1",
            ], cwd=ROOT, capture_output=True, encoding="utf-8", timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["dataset_sha256"], resumed["dataset_sha256"])
            self.assertEqual(report["architecture"], resumed["arch"])

    def test_config_rejects_unknown_keys_and_wrong_numeric_types(self):
        with tempfile.TemporaryDirectory(prefix="facies-config-") as directory:
            root = Path(directory)
            config = root / "bad.json"
            for settings in ({"epochs": True}, {"epochz": 1}, {"arch": "unrecognized"}):
                config.write_text(json.dumps(settings), encoding="utf-8")
                result = subprocess.run(
                    [sys.executable, "-X", "utf8", "-m", "facies.train", "--config", str(config)],
                    cwd=ROOT, capture_output=True, encoding="utf-8", timeout=30,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Lỗi:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()

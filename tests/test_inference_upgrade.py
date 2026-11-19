import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image

from facies.batch import process_folder
from facies.data import Record, load_mask
from facies.model import build_model
from facies.predict import Predictor, configure_device, load_predictor, prediction_provenance
from facies.render import export_results, render_preview


CLASSES = [
    {"id": 0, "name": "Nền", "color": [20, 25, 30]},
    {"id": 1, "name": "Hỗn loạn", "color": [215, 50, 100]},
    {"id": 2, "name": "Á song song", "color": [40, 160, 200]},
]


class ConstantModel(torch.nn.Module):
    def forward(self, image):
        return image.new_tensor([0.0, 7.0, 0.0]).view(1, 3, 1, 1).expand(
            image.shape[0], 3, *image.shape[-2:]
        )


class PositionalModel(torch.nn.Module):
    def forward(self, image):
        horizontal = torch.linspace(-3, 3, image.shape[-1], device=image.device)
        signal = horizontal.view(1, 1, -1).expand(image.shape[0], image.shape[-2], -1)
        return torch.stack((signal, torch.zeros_like(signal) + 12, -signal), 1)


class InferenceUpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_device("cpu", 1)

    def test_detailed_entropy_review_and_legacy_result(self):
        predictor = Predictor(ConstantModel(), CLASSES, 32, "cpu", active_ids=[0, 2])
        image = np.zeros((65, 93, 3), dtype=np.uint8)
        old_mask, old_conf = predictor.predict(image, overlap=9)
        details = predictor.predict_detailed(image, overlap=9, confidence_threshold=0.51)
        np.testing.assert_array_equal(details["raw_mask"], old_mask)
        np.testing.assert_array_equal(details["confidence"], old_conf)
        np.testing.assert_allclose(details["confidence"], 0.5, atol=1e-6)
        np.testing.assert_allclose(details["entropy"], 1.0, atol=1e-6)
        self.assertTrue(details["review_mask"].all())
        self.assertTrue((details["mask"] == 255).all())
        self.assertTrue((details["raw_mask"] == 0).all())
        details["mask"][:] = 2
        self.assertTrue((details["raw_mask"] == 0).all())
        accepted = predictor.predict_detailed(image, confidence_threshold=0)
        self.assertFalse(accepted["review_mask"].any())
        np.testing.assert_array_equal(accepted["mask"], old_mask)
        for threshold in (-0.1, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                predictor.predict_detailed(image, confidence_threshold=threshold)

    def test_horizontal_tta_aligns_before_averaging(self):
        predictor = Predictor(PositionalModel(), CLASSES, 32, "cpu", active_ids=[0, 2])
        for size in ((1, 1), (19, 71), (32, 32), (65, 93)):
            for overlap in (0, 8, 31):
                image = np.zeros((*size, 3), dtype=np.float32)
                result = predictor.predict_detailed(image, tta=True, overlap=overlap)
                self.assertEqual(result["mask"].shape, size)
                np.testing.assert_allclose(result["confidence"], 0.5, atol=2e-6)
                np.testing.assert_allclose(result["entropy"], 1.0, atol=2e-6)
                self.assertTrue(set(np.unique(result["raw_mask"])) <= {0, 2})

    def test_old_and_new_architecture_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            for arch in (None, "unet", "attention_unet"):
                model = build_model(arch or "unet", 3, width=4)
                checkpoint = {"format_version": 1, "classes": CLASSES, "width": 4, "tile_size": 32,
                              "state": model.state_dict(), "demo": True, "active_ids": [0, 2]}
                if arch is not None:
                    checkpoint["arch"] = arch
                path = Path(directory) / "model.pt"
                torch.save(checkpoint, path)
                predictor = load_predictor(path, "cpu")
                self.assertEqual(predictor.arch, arch or "unet")
                self.assertTrue(predictor.demo)
                image = np.zeros((13, 45, 3), dtype=np.uint8)
                details = predictor.predict_detailed(image, tta=True)
                self.assertEqual(details["mask"].shape, (13, 45))
                provenance = prediction_provenance(path, predictor, tta=True, confidence_threshold=0.65)
                self.assertEqual(len(provenance["model_sha256"]), 64)
                self.assertEqual(provenance["trained_class_ids"], [0, 2])
                self.assertEqual(provenance["architecture"], arch or "unet")
                self.assertTrue(provenance["horizontal_flip_tta"])

    def test_export_keeps_raw_mask_scores_and_reports_polygon_holes(self):
        with tempfile.TemporaryDirectory() as directory:
            image = np.full((90, 120, 3), 120, dtype=np.uint8)
            raw = np.zeros(image.shape[:2], dtype=np.uint8)
            raw[5:55, 5:55] = 1
            raw[15:45, 15:45] = 0  # A true hole must not be filled by a LabelMe polygon.
            raw[20:70, 70:110] = 2
            raw[80, 119] = 2
            mask = raw.copy()
            mask[65:70, 70:110] = 255
            review = mask == 255
            confidence = np.full(mask.shape, 0.75, dtype=np.float32)
            confidence[review] = 0.4
            entropy = np.full(mask.shape, 0.6, dtype=np.float32)
            before_raw, before_mask = raw.copy(), mask.copy()
            paths = export_results(
                directory, image, mask, confidence, CLASSES, min_region_area=20,
                raw_mask=raw, entropy=entropy, review_mask=review,
                provenance={"source_image": "test.png", "demo": True},
            )
            for key in ("overlay", "mask", "regions", "confidence", "uncertainty", "review",
                        "raw_mask", "labelme", "trace", "source"):
                self.assertTrue(paths[key].is_file())
            np.testing.assert_array_equal(raw, before_raw)
            np.testing.assert_array_equal(mask, before_mask)
            np.testing.assert_array_equal(np.array(Image.open(paths["mask"])), mask)
            np.testing.assert_array_equal(np.array(Image.open(paths["raw_mask"])), raw)
            np.testing.assert_array_equal(np.array(Image.open(paths["review"])) > 0, review)
            np.testing.assert_allclose(np.array(Image.open(paths["confidence"])) / 65535, confidence, atol=1/65535)
            np.testing.assert_allclose(np.array(Image.open(paths["uncertainty"])) / 65535, entropy, atol=1/65535)
            annotation = json.loads(paths["labelme"].read_text(encoding="utf-8"))
            omissions = annotation["_facies"]["omitted_regions"]
            self.assertTrue(any("holes" in item["reason"] and item["class_id"] == 1 for item in omissions))
            self.assertTrue(any(item["reason"] == "below_min_region_area" for item in omissions))
            self.assertTrue(all(shape["class_id"] == 2 for shape in annotation["shapes"]))
            # Raw machine proposals are deliberately not admissible expert labels.
            unreviewed = load_mask(Record(paths["source"], paths["labelme"], "test"), CLASSES)
            self.assertTrue((unreviewed == 255).all())
            for shape in annotation["shapes"]:
                if shape["class_id"] == 2:
                    shape["flags"]["reviewed"] = True
            paths["labelme"].write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")
            reimported = load_mask(Record(paths["source"], paths["labelme"], "test"), CLASSES)
            self.assertEqual(reimported[30, 30], 255)
            self.assertEqual(reimported[8, 8], 255)
            self.assertEqual(reimported[30, 80], 2)
            self.assertEqual(reimported[67, 80], 255)
            metadata = json.loads(paths["regions"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["review_pixel_count"], int(review.sum()))
            self.assertAlmostEqual(metadata["review_fraction"], review.mean())
            summary = next(item for item in metadata["class_summary"] if item["class_id"] == 2)
            self.assertEqual(summary["raw_pixel_count"], int((raw == 2).sum()))
            self.assertEqual(summary["review_pixel_count"], int(review.sum()))
            self.assertIn("opened_in_trace", paths["trace"].read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                export_results(directory, image, mask, confidence, CLASSES, raw_mask=raw,
                               review_mask=np.zeros(mask.shape, dtype=bool))

    def test_folder_priority_errors_and_zip(self):
        class ReviewPredictor:
            tile_size, demo, arch = 32, True, "unet"
            active_ids = np.array([0, 2], dtype=np.uint8)
            classes = CLASSES

            def predict_detailed(self, image, **kwargs):
                h, w = image.shape[:2]
                value = float(image.mean())
                confidence = np.full((h, w), 0.95 - value * 0.5, dtype=np.float32)
                review = confidence < kwargs["confidence_threshold"]
                raw_mask = np.full((h, w), 2, dtype=np.uint8)
                mask = raw_mask.copy()
                mask[review] = 255
                return {"mask": mask, "raw_mask": raw_mask, "confidence": confidence,
                        "entropy": np.full((h, w), value, dtype=np.float32), "review_mask": review}

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            inputs, output = base / "input", base / "output"
            inputs.mkdir()
            Image.new("RGB", (36, 32), (0, 0, 0)).save(inputs / "dark.png")
            Image.new("RGB", (36, 32), (255, 255, 255)).save(inputs / "white.png")
            (inputs / "corrupt.png").write_bytes(b"not an image")
            checkpoint = base / "model.pt"
            checkpoint.write_bytes(b"dummy fingerprint file")
            updates = []
            with patch("facies.batch.load_predictor", return_value=ReviewPredictor()):
                result = process_folder(
                    checkpoint, inputs, output, confidence_threshold=0.7,
                    zip_path=base / "results.zip", progress=lambda *args: updates.append(args),
                )
            self.assertEqual(result["summary"]["total_images"], 3)
            self.assertEqual(result["summary"]["successful_images"], 2)
            self.assertEqual(result["summary"]["failed_images"], 1)
            self.assertEqual(len(updates), 3)
            self.assertEqual(result["results"][0]["image"], "white.png")
            self.assertEqual(result["results"][0]["rank"], 1)
            self.assertEqual(result["results"][0]["review_fraction"], 1)
            self.assertTrue((output / result["results"][0]["paths"]["trace"]).is_file())
            with ZipFile(result["zip"]) as bundle:
                self.assertIn("index.csv", bundle.namelist())
                self.assertIn("index.json", bundle.namelist())
                self.assertIn(result["results"][0]["paths"]["raw_mask"], bundle.namelist())

    def test_preview_profile_is_lightweight_and_standard_skips_trace_assets(self):
        image = np.full((220, 340, 3), 120, dtype=np.uint8)
        mask = np.zeros((220, 340), dtype=np.uint8)
        mask[10:210, 15:320] = 1
        mask[90:108, 70:140] = 255
        confidence = np.full(mask.shape, 0.72, dtype=np.float32)
        entropy = np.full(mask.shape, 0.34, dtype=np.float32)
        review = mask == 255
        preview = render_preview(
            image, mask, confidence, CLASSES, entropy=entropy, review_mask=review,
            preview_max_side=96,
        )
        self.assertEqual(set(preview["files"]), {"overlay", "confidence", "review", "uncertainty"})
        self.assertEqual(Image.open(BytesIO(preview["files"]["overlay"])).size, (96, 62))
        self.assertEqual(preview["summary"]["width"], 340)
        self.assertEqual(preview["summary"]["height"], 220)
        self.assertAlmostEqual(preview["summary"]["review_fraction"], float(review.mean()))
        with Image.open(BytesIO(preview["files"]["review"])) as preview_review:
            self.assertEqual(set(np.unique(preview_review)), {0, 255})
        with Image.open(BytesIO(preview["files"]["confidence"])) as preview_confidence:
            self.assertEqual(preview_confidence.mode, "L")
        with tempfile.TemporaryDirectory() as directory:
            paths = export_results(
                directory, image, mask, confidence, CLASSES, raw_mask=mask,
                entropy=entropy, review_mask=review, profile="standard",
            )
            self.assertNotIn("source", paths)
            self.assertNotIn("labelme", paths)
            self.assertNotIn("trace", paths)
            self.assertTrue(paths["overlay"].is_file())
            self.assertTrue(paths["confidence"].is_file())
            self.assertTrue(paths["uncertainty"].is_file())
            with Image.open(paths["confidence"]) as exported_confidence:
                self.assertEqual(exported_confidence.size, (340, 220))
                np.testing.assert_allclose(np.array(exported_confidence) / 65535, confidence, atol=1/65535)
            with Image.open(paths["mask"]) as exported_mask:
                np.testing.assert_array_equal(np.array(exported_mask), mask)


if __name__ == "__main__":
    unittest.main()

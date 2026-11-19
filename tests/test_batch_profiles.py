import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image

from facies.batch import process_folder
from facies.data import load_classes
from facies.render import export_results


ROOT = Path(__file__).resolve().parents[1]
CLASSES = load_classes(ROOT / "classes.json")


class BatchProfileTests(unittest.TestCase):
    def test_standard_render_omits_editing_artifacts_and_review_keeps_them(self):
        image = np.zeros((40, 48, 3), dtype=np.uint8)
        mask = np.zeros((40, 48), dtype=np.uint8)
        mask[5:32, 8:38] = 2
        confidence = np.full(mask.shape, 0.8, dtype=np.float32)
        raw = mask.copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            standard = export_results(root / "standard", image, mask, confidence, CLASSES,
                                      raw_mask=raw, review_mask=np.zeros(mask.shape, bool), profile="standard")
            review = export_results(root / "review", image, mask, confidence, CLASSES,
                                    raw_mask=raw, review_mask=np.zeros(mask.shape, bool), profile="review")
            self.assertNotIn("source", standard)
            self.assertNotIn("trace", standard)
            self.assertNotIn("labelme", standard)
            self.assertIn("source", review)
            self.assertIn("trace", review)
            self.assertIn("labelme", review)
            metadata = json.loads(standard["regions"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["region_count"], 1)

    def test_batch_standard_zip_has_only_indexed_current_run_files(self):
        # This verifies archive enumeration with a lightweight fake predictor.
        import facies.batch as batch

        class FakePredictor:
            classes = CLASSES
            def predict_detailed(self, image, **kwargs):
                height, width = image.shape[:2]
                mask = np.zeros((height, width), np.uint8)
                mask[3:-3, 4:-4] = 2
                return {"mask": mask, "raw_mask": mask.copy(), "confidence": np.ones(mask.shape, np.float32),
                        "entropy": np.zeros(mask.shape, np.float32), "review_mask": np.zeros(mask.shape, bool)}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            output = root / "output"
            source.mkdir()
            Image.fromarray(np.full((32, 35, 3), 120, np.uint8)).save(source / "ảnh.png")
            output.mkdir()
            (output / "images-old").mkdir()
            (output / "images-old" / "stale.txt").write_text("previous run", encoding="utf-8")
            with patch.object(batch, "load_predictor", return_value=FakePredictor()), \
                 patch.object(batch, "prediction_provenance", return_value={"tile_size": 32, "overlap": 8}):
                result = process_folder(root / "model.pt", source, output, profile="standard",
                                        zip_path=root / "results.zip", device="cpu")
            self.assertEqual(result["summary"]["export_profile"], "standard")
            self.assertEqual(result["summary"]["successful_images"], 1)
            self.assertTrue(result["zip"].is_file())
            with ZipFile(result["zip"]) as archive:
                names = set(archive.namelist())
            row = result["results"][0]
            self.assertIsNone(row["omitted_polygon_regions"])
            self.assertEqual(set(row["paths"]), {"overlay", "mask", "regions", "confidence", "review", "raw_mask", "uncertainty"})
            self.assertEqual(names, {"index.json", "index.csv", *row["paths"].values()})


if __name__ == "__main__":
    unittest.main()

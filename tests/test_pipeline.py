import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from facies.data import Record, load_classes, load_mask, split_records
from facies.import_outlines import extract_outline_mask
from facies.predict import Predictor, configure_device
from facies.train import segmentation_loss


CLASSES = load_classes(Path(__file__).resolve().parents[1] / "classes.json")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_device("cpu", 1)

    def test_group_leakage_rejected(self):
        records = [
            Record(Path("a.png"), Path("a.json"), "line-a", "train"),
            Record(Path("b.png"), Path("b.json"), "LINE-A", "val"),
        ]
        with self.assertRaises(ValueError):
            split_records(records)

    def test_sparse_polygon_unlabeled_is_ignore(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            Image.new("RGB", (12, 12)).save(path / "input.png")
            (path / "input.json").write_text(json.dumps({
                "imageWidth": 12, "imageHeight": 12,
                "shapes": [{
                    "label": "parallel", "shape_type": "polygon",
                    "points": [[2, 2], [9, 2], [9, 9], [2, 9]],
                }],
            }))
            mask = load_mask(Record(path / "input.png", path / "input.json", "line"), CLASSES)
            self.assertEqual(mask[5, 5], 2)
            self.assertEqual(mask[0, 0], 255)

    def test_ignored_pixel_has_zero_gradient(self):
        logits = torch.randn(1, 7, 2, 2, requires_grad=True)
        labels = torch.tensor([[[255, 0], [2, 255]]])
        segmentation_loss(logits, labels, torch.ones(7)).backward()
        self.assertTrue(torch.all(logits.grad[:, :, 0, 0] == 0).item())
        self.assertTrue(torch.all(logits.grad[:, :, 1, 1] == 0).item())

    def test_stitching_and_unsupported_class_suppression(self):
        class PixelModel(torch.nn.Module):
            def forward(self, image):
                x = image[:, 0]
                return torch.stack([x, x * 0, -x, x * 0 + 25, x * 0, x * 0, x * 0], 1)

        predictor = Predictor(PixelModel(), CLASSES, 32, "cpu", active_ids=[0, 2])
        rng = np.random.default_rng(42)
        for size in ((1, 1), (29, 70), (32, 32), (64, 74), (83, 91)):
            image = rng.random((*size, 3), dtype=np.float32)
            for overlap in (0, 8, 31):
                with self.subTest(size=size, overlap=overlap):
                    mask, confidence = predictor.predict(image, overlap=overlap, batch_size=6)
                    np.testing.assert_array_equal(mask, np.where(image[:, :, 0] >= 0.5, 0, 2))
                    analytic = 1 / (1 + np.exp(-2 * np.abs(2 * image[:, :, 0] - 1)))
                    np.testing.assert_allclose(confidence, analytic, atol=3e-5)

    def test_closed_color_outline_and_open_outline(self):
        clean = np.full((64, 90, 3), 127, np.uint8)
        annotated = Image.fromarray(clean)
        drawing = ImageDraw.Draw(annotated)
        color = tuple(CLASSES[2]["color"])
        drawing.rectangle((4, 4, 40, 58), outline=color, width=2)
        drawing.text((12, 20), "SS", fill=color)
        mask, _, counts = extract_outline_mask(np.array(annotated), CLASSES, clean=clean)
        self.assertEqual(mask[30, 20], 2)
        self.assertEqual(mask[0, 0], 255)
        self.assertEqual(counts[CLASSES[2]["name"]], 1)
        bad = Image.fromarray(clean)
        ImageDraw.Draw(bad).line(((4, 4), (4, 58), (40, 58), (40, 4)), fill=color, width=2)
        with self.assertRaises(ValueError):
            extract_outline_mask(np.array(bad), CLASSES, clean=clean)

    def test_outline_overlap_rejected(self):
        clean = np.full((64, 90, 3), 127, np.uint8)
        annotated = Image.fromarray(clean)
        drawing = ImageDraw.Draw(annotated)
        drawing.rectangle((4, 4, 60, 58), outline=tuple(CLASSES[2]["color"]), width=2)
        drawing.rectangle((16, 14, 48, 45), outline=tuple(CLASSES[1]["color"]), width=2)
        with self.assertRaises(ValueError):
            extract_outline_mask(np.array(annotated), CLASSES, clean=clean)


if __name__ == "__main__":
    unittest.main()

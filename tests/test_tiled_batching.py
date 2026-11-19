import unittest

import numpy as np
import torch

from facies.predict import Predictor


class TiledBatchingTests(unittest.TestCase):
    def test_narrow_section_reuses_full_batches_without_changing_predictions(self):
        class CountingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batches = []

            def forward(self, image):
                self.batches.append(image.shape[0])
                red = image[:, 0]
                return torch.stack((red, -red), dim=1)

        image = np.random.default_rng(7).random((230, 19, 3), dtype=np.float32)
        model = CountingModel()
        classes = [{"id": 0}, {"id": 1}]
        predictor = Predictor(model, classes, 32, "cpu")
        reference = predictor.predict_detailed(image, batch_size=1, tta=True, confidence_threshold=.65)
        self.assertEqual(model.batches, [1] * 20)
        model.batches.clear()
        grouped = predictor.predict_detailed(image, batch_size=4, tta=True, confidence_threshold=.65)
        self.assertEqual(model.batches, [4, 4, 4, 4, 2, 2])
        for key in ("mask", "raw_mask", "review_mask"):
            np.testing.assert_array_equal(reference[key], grouped[key])
        for key in ("confidence", "entropy"):
            np.testing.assert_allclose(reference[key], grouped[key], atol=1e-6)


if __name__ == "__main__":
    unittest.main()

"""Optional real-browser regression: saving a proposal never approves its labels."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from facies.data import Record, load_classes, load_mask


PROJECT = Path(__file__).resolve().parents[1]
EDGE = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
BROWSER_AVAILABLE = importlib.util.find_spec("playwright") is not None and EDGE.is_file()


@unittest.skipUnless(BROWSER_AVAILABLE, "Optional Playwright and local Edge required")
class TraceProvenanceTests(unittest.TestCase):
    def test_proposal_flags_survive_save_review_undo_and_relabel(self):
        from playwright.sync_api import sync_playwright

        classes = load_classes(PROJECT / "classes.json")
        with tempfile.TemporaryDirectory(prefix="trace-flags-") as directory, sync_playwright() as playwright:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (80, 60), "#888888").save(source)
            document = {
                "flags": {"machine_prediction": True, "reviewed": True, "expert_edits": True, "custom": "keep"},
                "imageWidth": 80, "imageHeight": 60, "imagePath": source.name, "imageData": None,
                "_facies": {"requires_expert_review": True},
                "shapes": [
                    {"label": "Song song", "class_id": 2, "mean_confidence": .8, "points": [[5, 5], [25, 5], [25, 25], [5, 25]], "shape_type": "polygon", "flags": {"machine_prediction": True, "reviewed": False, "custom": "region"}},
                    {"label": "Hỗn loạn", "points": [[40, 5], [60, 5], [60, 25], [40, 25]], "shape_type": "polygon", "flags": {}},
                ],
            }
            annotation = root / "proposal.json"
            annotation.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            browser = playwright.chromium.launch(executable_path=str(EDGE), headless=True)
            try:
                page = browser.new_page(accept_downloads=True, viewport={"width": 1300, "height": 950})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto((PROJECT / "trace.html").as_uri())
                page.locator("#ref-file").set_input_files(str(source))
                page.wait_for_function("state.ref !== null && !state.loading")
                page.locator("#json-file").set_input_files(str(annotation))
                page.wait_for_function("state.shapes.length === 2 && !state.loading")
                self.assertEqual(page.locator('.shape[data-review="pending"]').count(), 2)

                def saved(name):
                    with page.expect_download() as delivery:
                        page.locator("#save").click()
                    path = root / name
                    delivery.value.save_as(str(path))
                    return path, json.loads(path.read_text(encoding="utf-8"))

                path, exported = saved("untouched.json")
                self.assertEqual(exported["flags"], document["flags"])
                self.assertEqual(exported["shapes"][0]["flags"], document["shapes"][0]["flags"])
                self.assertEqual(exported["_facies"], document["_facies"])
                mask = load_mask(Record(source, path, "line"), classes)
                self.assertTrue(np.all(mask == 255))  # Global reviewed/expert_edits cannot approve regions.

                page.locator(".shape").nth(0).click()
                page.locator("#reviewed").click()
                self.assertEqual(page.locator('.shape[data-review="reviewed"]').count(), 1)
                path, approved = saved("approved.json")
                self.assertTrue(approved["shapes"][0]["flags"]["reviewed"])
                self.assertEqual(approved["shapes"][0]["flags"]["custom"], "region")
                mask = load_mask(Record(source, path, "line"), classes)
                self.assertEqual(mask[10, 10], 2)
                self.assertEqual(mask[10, 50], 255)

                page.locator("#undo").click()
                self.assertEqual(page.locator('.shape[data-review="pending"]').count(), 2)
                _, undone = saved("undone.json")
                self.assertFalse(undone["shapes"][0]["flags"]["reviewed"])
                page.locator(".shape").nth(0).click()
                page.locator("#label").select_option("Sigma")
                page.locator("#relabel").click()
                path, edited = saved("edited.json")
                self.assertEqual(edited["flags"], document["flags"])
                self.assertEqual(edited["shapes"][0]["label"], "Sigma")
                self.assertEqual(edited["shapes"][0]["class_id"], 6)
                self.assertTrue(edited["shapes"][0]["flags"]["reviewed"])
                self.assertTrue(edited["shapes"][0]["flags"]["machine_prediction"])
                self.assertNotIn("mean_confidence", edited["shapes"][0])
                self.assertEqual(edited["shapes"][0]["model_mean_confidence_before_edit"], .8)
                self.assertEqual(load_mask(Record(source, path, "line"), classes)[10, 10], 6)

                # A newly traced expert polygon is approved only at shape level.
                for x, y in [(5, 35), (25, 35), (25, 50), (5, 50)]:
                    bounds = page.locator("#surface").bounding_box()
                    page.mouse.click(bounds["x"] + x + .5, bounds["y"] + y + .5)
                page.keyboard.press("Enter")
                path, added = saved("added.json")
                self.assertTrue(added["shapes"][-1]["flags"]["reviewed"])
                self.assertFalse(added["shapes"][-1]["flags"]["machine_prediction"])
                self.assertEqual(load_mask(Record(source, path, "line"), classes)[40, 10], 6)
                self.assertFalse(errors, errors)
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()

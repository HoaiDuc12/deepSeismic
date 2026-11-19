"""Smoke tests for the four local workspaces and their real backend contracts."""

import json
import tempfile
import unittest
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

import app
from facies.data import load_classes
from facies.model import UNet


PROJECT = Path(__file__).resolve().parents[1]
CLASSES = load_classes(PROJECT / "classes.json")


@contextmanager
def fixture_project():
    """Small deterministic checkpoint and clean images, never measured geology."""
    with tempfile.TemporaryDirectory(prefix="facies-app-") as directory:
        root = Path(directory)
        demo = root / "demo"
        demo.mkdir()
        model = UNet(len(CLASSES), width=4)
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
            model.head.bias[1] = 2
        torch.save({"format_version": 1, "arch": "unet", "classes": CLASSES, "width": 4,
                    "tile_size": 32, "state": model.state_dict(), "demo": True,
                    "active_ids": [0, 1]}, demo / "model.pt")
        pixels = np.full((40, 64, 3), 140, dtype=np.uint8)
        Image.fromarray(pixels).save(demo / "sample.png")
        (root / "classes.json").write_text(json.dumps(CLASSES, ensure_ascii=False), encoding="utf-8")
        with patch.multiple(app, ROOT=root, DEMO_MODEL=demo / "model.pt", DEMO_IMAGE=demo / "sample.png", DEFAULT_MODEL=root / "runs/facies/best.pt"):
            yield root


def new_app():
    return AppTest.from_string("import app\napp.main()").run(timeout=45)


class AppUpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_clean(self, application):
        self.assertFalse(application.exception, [item.value for item in application.exception])
        self.assertFalse(application.error, [item.value for item in application.error])

    def test_map_preview_scales_16bit_without_clipping(self):
        values = np.array([[0, 32768, 65535]], dtype=np.uint16)
        buffer = BytesIO()
        Image.fromarray(values).save(buffer, format="PNG")
        np.testing.assert_array_equal(app.map_preview(buffer.getvalue()), [[0, 128, 255]])

    def test_no_checkpoint_keeps_all_workspaces_available(self):
        with fixture_project() as root:
            (root / "demo/model.pt").unlink()
            application = new_app()
            self.assert_clean(application)
            self.assertEqual([item.label for item in application.tabs], ["Phân tướng một ảnh", "Xử lý thư mục", "Kiểm tra dữ liệu", "Thí nghiệm"])
            self.assertTrue(any("Chưa tìm thấy trọng số" in item.value for item in application.info))
            self.assertTrue(application.button(key="run_batch").disabled)

    def test_threshold_and_tta_clear_stale_single_results(self):
        with fixture_project():
            application = new_app()
            application.button(key="run_single").click().run(timeout=45)
            self.assert_clean(application)
            self.assertGreaterEqual(len(application.get("download_button")), 10)
            self.assertEqual(application.metric[2].value, "0.0%")
            files = application.session_state["facies_result"][1]["files"]
            self.assertTrue({"labelme", "trace", "raw_mask", "source", "confidence", "uncertainty", "review"}.issubset(files))
            application.slider(key="review_threshold").set_value(0.95).run()
            self.assertFalse(application.get("download_button"))
            application.checkbox(key="use_tta").check().run()
            application.button(key="run_single").click().run(timeout=45)
            self.assert_clean(application)
            self.assertEqual(application.metric[2].value, "100.0%")
            mask_bytes = application.session_state["facies_result"][1]["files"]["mask"]
            with Image.open(BytesIO(mask_bytes)) as mask:
                self.assertTrue(np.all(np.asarray(mask) == 255))

    def test_batch_uses_new_directory_and_relative_preview_paths(self):
        with fixture_project() as root:
            images = root / "inputs"
            images.mkdir()
            for name in ("a.png", "b.png"):
                (images / name).write_bytes((root / "demo/sample.png").read_bytes())
            application = new_app()
            application.text_input(key="batch_source").set_value(str(images))
            application.text_input(key="batch_output").set_value(str(root / "batch"))
            application.button(key="run_batch").click().run(timeout=60)
            self.assert_clean(application)
            result = application.session_state["batch_result"][1]
            self.assertEqual(len(result["results"]), 2)
            self.assertFalse(result["errors"])
            base = Path(result["index_csv"]).parent
            self.assertTrue((base / result["results"][0]["paths"]["overlay"]).is_file())
            self.assertTrue(any(item.label == "Tải công cụ sửa vùng cho ảnh này" for item in application.get("download_button")))

    def test_invalid_manifest_is_reported_without_success(self):
        with fixture_project():
            application = new_app()
            application.button(key="run_audit").click().run(timeout=45)
            self.assert_clean(application)
            report = application.session_state["audit_result"][1]
            self.assertFalse(report["ready_for_training"])
            self.assertTrue(report["errors"])
            self.assertFalse(application.success)
            self.assertTrue(any(item.label == "Tải báo cáo kiểm tra JSON" for item in application.get("download_button")))

    def test_legacy_report_and_missing_metrics_stay_undefined(self):
        with fixture_project() as root:
            report = {"demo": True, "best_epoch": 1,
                      "validation": {"sections": 1, "mean_iou_facies": .6,
                                     "classes": [{"id": 0, "name": "Nền", "support_pixels": 0, "iou": None, "dice": None}],
                                     "confusion_matrix": [[0]]},
                      "history": [{"epoch": 1, "loss": 2.0, "val_miou_facies": .6}]}
            (root / "demo/metrics.json").write_text(json.dumps(report), encoding="utf-8")
            application = new_app()
            self.assert_clean(application)
            self.assertGreaterEqual(len(application.dataframe), 3)
            rows = app.experiment_rows({"demo": report}, "validation")
            self.assertIsNone(rows[0]["F1 biên"])
            self.assertIsNone(rows[0]["Độ đúng pixel"])
            application.radio(key="report_split").set_value("test").run()
            self.assert_clean(application)
            self.assertTrue(any("chưa có kết quả" in item.value for item in application.info))

    def test_comparison_warns_on_different_dataset_fingerprints(self):
        with fixture_project() as root:
            report = {"demo": True, "dataset_sha256": "a" * 64, "validation": {"mean_iou_facies": .6}}
            demo_path = root / "demo/metrics.json"
            demo_path.write_text(json.dumps(report), encoding="utf-8")
            other = root / "runs/other/metrics.json"
            other.parent.mkdir(parents=True)
            other.write_text(json.dumps({**report, "dataset_sha256": "b" * 64}), encoding="utf-8")
            application = new_app()
            application.multiselect(key="selected_reports").set_value([str(demo_path), str(other)]).run()
            self.assert_clean(application)
            self.assertTrue(any("dấu vân tay dữ liệu khác nhau" in item.value for item in application.warning))
            self.assertEqual(len(application.dataframe[0].value), 2)

    def test_malformed_report_class_metrics_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps({"validation": {"classes": [1, 2]}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "không hợp lệ"):
                app.load_metric_report(path)

    def test_boundary_f1_display_joins_classes_by_id(self):
        report = {"validation": {
            "classes": [{"id": 1, "name": "Hỗn loạn", "support_pixels": 100}, {"id": 2, "name": "Song song", "support_pixels": 90}],
            "boundary_classes": [{"id": 2, "boundary_f1": .75}, {"id": 1, "boundary_f1": .6}],
        }}
        application = AppTest.from_string(f"from app import render_experiment_detail\nrender_experiment_detail({report!r}, 'validation')").run()
        self.assert_clean(application)
        self.assertEqual(application.dataframe[0].value["F1 biên"].tolist(), [.6, .75])

    def test_flat_evaluator_reports_normalize_without_inventing_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            for split, target in (("val", "validation"), ("test", "test")):
                report = {"split": split, "architecture": "attention_unet", "dataset_sha256": "a" * 64,
                          "demo": True, "classes": [{"id": 0, "name": "Nền"}], "confusion_matrix": [[10]],
                          "mean_iou_facies": .6, "sections": 2, "model_sha256": "b" * 64}
                path.write_text(json.dumps(report), encoding="utf-8")
                normalized = app.load_metric_report(path)
                self.assertEqual(normalized[target], report)
                self.assertEqual(normalized["model_sha256"], report["model_sha256"])
                self.assertEqual(normalized["dataset_sha256"], report["dataset_sha256"])
                self.assertNotIn("history", normalized)
                self.assertIsNone(app.experiment_rows({"evaluation": normalized}, target)[0]["Epoch tốt nhất"])
            for invalid in ({**report, "split": "train"}, {**report, "confusion_matrix": [[1, 2]]}, {**report, "classes": []}):
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    app.load_metric_report(path)

    def test_flat_evaluator_report_renders_in_experiment_workspace(self):
        with fixture_project() as root:
            path = root / "demo/metrics.json"
            report = {"split": "test", "demo": True, "architecture": "attention_unet",
                      "classes": [{"id": 1, "name": "Hỗn loạn", "support_pixels": 100, "iou": .5}],
                      "confusion_matrix": [[100]], "mean_iou_facies": .5, "sections": 1}
            path.write_text(json.dumps(report), encoding="utf-8")
            application = new_app()
            application.radio(key="report_split").set_value("test").run()
            self.assert_clean(application)
            self.assertEqual(application.dataframe[0].value["mIoU tướng"].tolist(), [.5])
            self.assertEqual(application.dataframe[1].value["Tướng"].tolist(), ["Hỗn loạn"])
            self.assertEqual(application.dataframe[2].value.iloc[0, 0], 100)

    def test_optional_attention_demo_selector(self):
        with fixture_project() as root:
            folder = root / "demo/attention"
            folder.mkdir()
            (folder / "model.pt").write_bytes((root / "demo/model.pt").read_bytes())
            application = new_app()
            application.selectbox(key="demo_variant").select("Attention U-Net · mô phỏng").run()
            application.button(key="run_single").click().run(timeout=45)
            self.assert_clean(application)
            signature = application.session_state["facies_result"][0][1]
            self.assertIn(str(folder / "model.pt"), signature)

    @unittest.skipUnless((PROJECT / "demo/model.pt").is_file(), "Bundled synthetic demo unavailable")
    def test_real_bundled_demo_ui_inference(self):
        application = AppTest.from_file(str(PROJECT / "app.py")).run(timeout=60)
        if application.radio(key="model_source").value != app.DEMO_MODE:
            application.radio(key="model_source").set_value(app.DEMO_MODE).run()
        application.button(key="run_single").click().run(timeout=120)
        self.assert_clean(application)
        self.assertGreaterEqual(len(application.get("download_button")), 10)
        result = application.session_state["facies_result"][1]
        self.assertTrue(result["demo"])
        self.assertTrue(0 <= result["review_fraction"] <= 1)


if __name__ == "__main__":
    unittest.main()

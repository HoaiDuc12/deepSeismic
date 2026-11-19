"""Evaluate a stored model on explicitly held-out, complete seismic sections."""

import argparse
import csv
import json
import pickle
import sys
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

from .audit import image_sha256
from .data import IGNORE_INDEX, load_image, load_mask, load_records
from .experiment import dataset_fingerprint
from .metrics import boundary_counts, boundary_metrics, confusion_matrix, confusion_metrics


def select_evaluation_records(records, split):
    """Require named validation/test assignments; never reconstruct or fall back to train."""
    if split not in {"val", "test"}:
        raise ValueError("Chỉ đánh giá tập val hoặc test độc lập.")
    groups, hashes = {}, {}
    for record in records:
        if record.split not in {"train", "val", "test"}:
            raise ValueError("Đánh giá độc lập cần cột split rõ ràng cho mọi ảnh: train, val hoặc test.")
        key = unicodedata.normalize("NFC", record.group.strip()).casefold()
        if groups.setdefault(key, record.split) != record.split:
            raise ValueError(f"Nhóm {record.group!r} xuất hiện trong nhiều tập.")
        digest = image_sha256(record.image)
        if hashes.setdefault(digest, record.split) != record.split:
            raise ValueError(f"Ảnh {record.image} trùng nội dung với ảnh ở tập khác.")
    selected = [record for record in records if record.split == split]
    if not selected:
        raise ValueError(f"Không có ảnh được chỉ định split={split}. Không dùng dữ liệu train thay thế.")
    return selected


def evaluate_records(predictor, records, *, batch_size=4, boundary_tolerance=2, overlap=None,
                     tile_size=None, preview_dir=None, previews=0, progress=None):
    """Return aggregate and per-section metrics using full original image grids."""
    if not records:
        raise ValueError("Tập đánh giá đang trống.")
    if batch_size < 1 or previews < 0:
        raise ValueError("Batch cần lớn hơn 0 và số ảnh xem trước không âm.")
    n = len(predictor.classes)
    aggregate = np.zeros((n, n), dtype=np.int64)
    edges = np.zeros((n, 4), dtype=np.int64)
    per_section = []
    for index, record in enumerate(records):
        image = load_image(record.image)
        height, width = image.shape[:2]
        truth = load_mask(record, predictor.classes, image_size=(width, height))
        if not np.any(truth != IGNORE_INDEX):
            raise ValueError(f"Ảnh không có pixel chuyên gia để đánh giá: {record.image}.")
        prediction, confidence = predictor.predict(
            image, batch_size=batch_size, overlap=overlap, tile_size=tile_size,
        )
        matrix = confusion_matrix(truth, prediction, n)
        section_edges = boundary_counts(truth, prediction, n, tolerance=boundary_tolerance)
        aggregate += matrix
        edges += section_edges
        scores = confusion_metrics(matrix, predictor.classes)
        scores.update(boundary_metrics(section_edges, predictor.classes, boundary_tolerance))
        scores.update(image=str(record.image), mask=str(record.mask), group=record.group, split=record.split,
                      width=width, height=height, ignored_pixels=int(np.count_nonzero(truth == IGNORE_INDEX)))
        per_section.append(scores)
        if preview_dir is not None and index < previews:
            from .render import export_results
            folder = Path(preview_dir) / f"{index + 1:04d}-{record.image.stem}"
            export_results(folder, image, prediction, confidence, predictor.classes)
            Image.fromarray(truth).save(folder / "expert_mask.png")
        if progress is not None:
            progress(f"{index + 1}/{len(records)} · {record.image.name}")
    report = confusion_metrics(aggregate, predictor.classes)
    report.update(boundary_metrics(edges, predictor.classes, boundary_tolerance))
    report.update(sections=len(records), ignored_pixels=sum(row["ignored_pixels"] for row in per_section),
                  groups=sorted({record.group for record in records}))
    for label, edge in zip(report["classes"], report.pop("boundary_classes")):
        label.update({name: value for name, value in edge.items() if name not in {"id", "name"}})
    return report, per_section


def save_evaluation(report, per_section, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"metrics": out / "metrics.json", "classes": out / "metrics.csv", "sections": out / "sections.csv"}
    paths["metrics"].write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with paths["classes"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["classes"][0]))
        writer.writeheader()
        writer.writerows(report["classes"])
    # One section/class per row preserves the full per-class metrics for difficult sections.
    base_columns = ["image", "group", "split", "width", "height", "labeled_pixels", "ignored_pixels",
                    "pixel_accuracy", "mean_iou_facies", "mean_boundary_f1_facies"]
    class_columns = list(report["classes"][0])
    with paths["sections"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_columns + class_columns)
        writer.writeheader()
        for section in per_section:
            base = {name: section[name] for name in base_columns}
            for label, boundary in zip(section["classes"], section["boundary_classes"]):
                writer.writerow({**base, **label, **boundary})
    return paths


def main(argv=None):
    from .predict import default_device, load_predictor
    parser = argparse.ArgumentParser(description="Đánh giá toàn lát cắt trên tập val/test có split rõ ràng.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--out", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--previews", type=int, default=0, help="Số lát lưu kèm ảnh dự đoán và mask chuyên gia.")
    args = parser.parse_args(argv)
    try:
        all_records = load_records(args.manifest)
        records = select_evaluation_records(all_records, args.split)
        predictor = load_predictor(args.model, args.device)
        fingerprint = dataset_fingerprint({
            name: [record for record in all_records if record.split == split]
            for name, split in (("train", "train"), ("validation", "val"), ("test", "test"))
        }, predictor.classes)
        report, sections = evaluate_records(
            predictor, records, batch_size=args.batch_size, boundary_tolerance=args.boundary_tolerance,
            overlap=args.overlap, tile_size=args.tile_size, preview_dir=args.out / "previews", previews=args.previews,
            progress=lambda message: print(message, flush=True),
        )
        report.update(model=str(args.model.resolve()), manifest=str(args.manifest.resolve()),
                      split=args.split, demo=bool(predictor.demo), dataset_sha256=fingerprint,
                      architecture=getattr(predictor, "arch", "unet"))
        if predictor.demo:
            report["notice"] = "Mô hình mô phỏng; điểm này chưa chứng minh chất lượng trên địa chấn thực."
        for path in save_evaluation(report, sections, args.out).values():
            print(path.resolve())
    except (OSError, ValueError, RuntimeError, KeyError, pickle.UnpicklingError) as exc:
        parser.exit(2, f"Lỗi: {exc}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()

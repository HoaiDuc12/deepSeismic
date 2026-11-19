"""Inspect annotation coverage, exact duplicates and independent survey splits."""

import argparse
import csv
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .data import IGNORE_INDEX, Record, load_classes, load_mask, split_records


def _key(value):
    return unicodedata.normalize("NFC", value.strip()).casefold()


def image_sha256(path):
    """Hash file bytes, including duplicates stored under different names."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _issue(report, severity, code, message, image=None):
    row = {"code": code, "message": str(message)}
    if image is not None:
        row["image"] = str(image)
    report[severity].append(row)


def _manifest_records(manifest, report):
    records, seen = [], set()
    try:
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = [value.strip() for value in (reader.fieldnames or [])]
            if len(set(headers)) != len(headers) or not {"image", "mask", "group"} <= set(headers):
                raise ValueError("CSV cần các cột image, mask, group duy nhất; split là tùy chọn.")
            reader.fieldnames = headers
            for row in reader:
                image, mask, group, split = (
                    (row.get(name) or "").strip() for name in ("image", "mask", "group", "split")
                )
                if None in row or not image or not mask or not group:
                    _issue(report, "errors", "manifest_row", f"Dòng {reader.line_num} thiếu image/mask/group hoặc thừa giá trị.")
                    continue
                split = {"validation": "val", "valid": "val"}.get(_key(split), _key(split))
                if split not in {"", "train", "val", "test"}:
                    _issue(report, "errors", "split_unknown", f"Dòng {reader.line_num}: split cần là train, val hoặc test.", image)
                    continue
                image, mask = ((manifest.parent / value).resolve() for value in (image, mask))
                if image in seen:
                    _issue(report, "errors", "duplicate_path", "Ảnh xuất hiện nhiều lần trong manifest.", image)
                seen.add(image)
                if image == mask:
                    _issue(report, "errors", "image_is_mask", "Ảnh gốc và mask đang là cùng một tệp.", image)
                records.append(Record(image, mask, group, split))
    except (OSError, ValueError, csv.Error) as exc:
        _issue(report, "errors", "manifest_invalid", exc)
    if not records:
        _issue(report, "errors", "manifest_empty", "Không đọc được dòng dữ liệu hợp lệ từ manifest.")
    return records


def _new_stats(classes):
    return {
        "sections": 0, "valid_sections": 0, "groups": [], "total_pixels": 0,
        "labeled_pixels": 0, "ignored_pixels": 0, "ignore_fraction": None,
        "classes": [{"id": label["id"], "name": label["name"], "pixel_count": 0,
                     "section_count": 0, "labeled_fraction": None} for label in classes],
    }


def _finalize_stats(stats):
    stats["groups"] = sorted({_key(group): group for group in stats["groups"]}.values())
    total = stats["total_pixels"]
    stats["ignore_fraction"] = stats["ignored_pixels"] / total if total else None
    for label in stats["classes"]:
        label["labeled_fraction"] = label["pixel_count"] / stats["labeled_pixels"] if stats["labeled_pixels"] else None


def _split_warnings(report, classes):
    for name, stats in report["splits"].items():
        if not stats["sections"]:
            continue
        for label in stats["classes"][1:]:
            if not label["pixel_count"]:
                _issue(report, "warnings", "class_missing", f"{name}: chưa có pixel lớp {label['id']} · {label['name']}.")
            elif label["section_count"] < 2:
                _issue(report, "warnings", "class_few_sections", f"{name}: {label['name']} chỉ có nhãn trên một lát cắt.")
        if stats["ignore_fraction"] is not None and stats["ignore_fraction"] > 0.8:
            _issue(report, "warnings", "sparse_labels", f"{name}: hơn 80% pixel chưa gán nhãn hoặc bỏ qua.")
        present = [label["pixel_count"] for label in stats["classes"][1:] if label["pixel_count"]]
        if present and max(present) / min(present) >= 20:
            _issue(report, "warnings", "class_imbalance", f"{name}: tỷ lệ pixel giữa các tướng đã có nhãn lớn nhất/nhỏ nhất từ 20 lần.")
        if name in {"val", "test"} and len(stats["groups"]) < 2:
            _issue(report, "warnings", "few_evaluation_groups", f"{name}: chỉ có một nhóm khảo sát; độ đa dạng đánh giá còn hạn chế.")
    for name in ("train", "val"):
        stats = report["splits"][name]
        if not stats["sections"] or not sum(label["pixel_count"] for label in stats["classes"][1:]):
            _issue(report, "errors", "split_no_facies", f"Tập {name} cần ít nhất một ảnh có pixel tướng hợp lệ.")
    if sum(label["pixel_count"] > 0 for label in report["splits"]["train"]["classes"]) < 2:
        _issue(report, "errors", "train_too_few_classes", "Tập train cần ít nhất hai lớp có pixel, có thể gồm nền.")
    if not report["splits"]["test"]["sections"]:
        _issue(report, "warnings", "test_missing", "Chưa có tập test độc lập. Điểm val dùng chọn mô hình chưa thay thế kiểm định cuối.")


def audit_dataset(manifest, classes_path="classes.json", *, val_fraction=0.2, seed=42):
    """Return a JSON-ready readiness report; malformed data is reported as errors."""
    manifest = Path(manifest).resolve()
    report = {
        "schema_version": 1, "manifest": str(manifest), "classes": [],
        "ready_for_training": False, "status": "error", "split_mode": None,
        "errors": [], "warnings": [], "summary": {}, "splits": {}, "sections": [], "duplicates": [],
        "dimensions": {}, "split_parameters": {"val_fraction": val_fraction, "seed": seed},
        "scope": "Kiểm tra kỹ thuật và độ phủ nhãn; chuyên gia vẫn cần xác nhận vùng và tính độc lập địa chất.",
    }
    try:
        classes = load_classes(classes_path)
    except (ValueError, OSError) as exc:
        _issue(report, "errors", "classes_invalid", exc)
        return report
    report["classes"] = classes
    report["splits"] = {name: _new_stats(classes) for name in ("train", "val", "test")}
    records = _manifest_records(manifest, report)
    explicit = any(record.split for record in records)
    report["split_mode"] = "explicit" if explicit else "grouped"
    assignments = {}
    try:
        for name, partition in zip(("train", "val", "test"), split_records(records, val_fraction, seed)):
            for record in partition:
                assignments[id(record)] = name
    except ValueError as exc:
        _issue(report, "errors", "split_invalid", exc)
        assignments = {id(record): record.split or "unassigned" for record in records}
    groups = defaultdict(set)
    for record in records:
        groups[_key(record.group)].add(assignments.get(id(record), "unassigned"))
    for group, names in groups.items():
        if len(names) > 1:
            _issue(report, "errors", "group_leakage", f"Nhóm {group!r} nằm trong nhiều tập: {', '.join(sorted(names))}.")

    hashes, sizes = defaultdict(list), Counter()
    for record in records:
        name = assignments.get(id(record), "unassigned")
        stats = report["splits"].setdefault(name, _new_stats(classes))
        stats["sections"] += 1
        stats["groups"].append(record.group)
        section = {"image": str(record.image), "mask": str(record.mask), "group": record.group,
                   "split": name, "status": "error", "errors": []}
        report["sections"].append(section)
        try:
            digest = image_sha256(record.image)
            section["sha256"] = digest
            hashes[digest].append(section)
            with Image.open(record.image) as image:
                image.load()
                width, height = image.size
                if getattr(image, "n_frames", 1) > 1:
                    _issue(report, "warnings", "multiple_frames", "Ảnh nhiều trang: chỉ dùng trang đầu.", record.image)
            section.update(width=width, height=height, aspect_ratio=width / height)
            sizes[(width, height)] += 1
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            section["errors"].append(str(exc))
            _issue(report, "errors", "image_invalid", exc, record.image)
            continue
        try:
            mask = load_mask(record, classes, image_size=(width, height))
            pixels = np.bincount(mask[mask != IGNORE_INDEX], minlength=len(classes))
            ignored = int(np.count_nonzero(mask == IGNORE_INDEX))
            labeled = int(pixels.sum())
            section.update(
                total_pixels=int(mask.size), labeled_pixels=labeled, ignored_pixels=ignored,
                ignore_fraction=ignored / mask.size, class_pixels=pixels.tolist(),
                present_class_ids=np.flatnonzero(pixels).tolist(), status="valid",
            )
            stats["total_pixels"] += mask.size
            stats["labeled_pixels"] += labeled
            stats["ignored_pixels"] += ignored
            for item, count in zip(stats["classes"], pixels):
                item["pixel_count"] += int(count)
                item["section_count"] += int(count > 0)
            if not labeled:
                raise ValueError("Ảnh không có pixel được chuyên gia gán nhãn; toàn bộ mask là 255.")
            stats["valid_sections"] += 1
        except (OSError, ValueError) as exc:
            section["status"] = "error"
            section["errors"].append(str(exc))
            _issue(report, "errors", "mask_invalid", exc, record.image)
    for digest, copies in hashes.items():
        if len(copies) < 2:
            continue
        splits = sorted({row["split"] for row in copies})
        cross_split = len(splits) > 1
        report["duplicates"].append({"sha256": digest, "images": [row["image"] for row in copies],
                                     "splits": splits, "cross_split": cross_split})
        _issue(report, "errors" if cross_split else "warnings", "duplicate_cross_split" if cross_split else "duplicate_content",
               f"{len(copies)} ảnh có SHA256 giống nhau trong {', '.join(splits)}.", copies[0]["image"])
    for stats in report["splits"].values():
        _finalize_stats(stats)
    _split_warnings(report, classes)
    all_stats = list(report["splits"].values())
    total = sum(item["total_pixels"] for item in all_stats)
    ignored = sum(item["ignored_pixels"] for item in all_stats)
    missing = [label["id"] for index, label in enumerate(classes)
               if not sum(item["classes"][index]["pixel_count"] for item in all_stats)]
    report["summary"] = {
        "sections": len(records), "valid_sections": sum(item["valid_sections"] for item in all_stats),
        "groups": len(groups), "total_pixels": total, "labeled_pixels": total - ignored,
        "ignored_pixels": ignored, "ignore_fraction": ignored / total if total else None,
        "missing_class_ids": missing,
    }
    if sizes:
        report["dimensions"] = {
            "width_min": min(size[0] for size in sizes), "width_max": max(size[0] for size in sizes),
            "height_min": min(size[1] for size in sizes), "height_max": max(size[1] for size in sizes),
            "aspect_min": min(width / height for width, height in sizes),
            "aspect_max": max(width / height for width, height in sizes),
            "unique_sizes": [{"width": w, "height": h, "sections": count} for (w, h), count in sorted(sizes.items())],
        }
        if report["dimensions"]["aspect_max"] / report["dimensions"]["aspect_min"] >= 4:
            _issue(report, "warnings", "aspect_variation", "Tỷ lệ rộng/cao thay đổi từ 4 lần; kiểm tra trục ảnh và tỷ lệ hiển thị.")
    report["ready_for_training"] = not report["errors"]
    report["status"] = "error" if report["errors"] else "warning" if report["warnings"] else "ready"
    return report


def save_audit(report, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "audit.json", "report": out / "audit.md"}
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = report["summary"]
    text = ["# Kiểm tra dữ liệu tướng địa chấn", "", f"Trạng thái: **{report['status']}**.", "",
            f"Manifest: `{report['manifest']}`", "", report["scope"], "",
            f"Số lát cắt: {summary.get('sections', 0)}; đọc nhãn hợp lệ: {summary.get('valid_sections', 0)}.", ""]
    for heading, key in (("Lỗi cần sửa", "errors"), ("Cảnh báo cần xem", "warnings")):
        text.extend([f"## {heading}", ""])
        text.extend([f"- {item['message']}" + (f" · `{item['image']}`" if item.get("image") else "") for item in report[key]] or ["Không có."])
        text.append("")
    text += ["## Độ phủ theo tập", "", "| Tập | Ảnh | Nhóm | Pixel có nhãn | Pixel bỏ qua |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in report["splits"].items():
        text.append(f"| {name} | {row['sections']} | {len(row['groups'])} | {row['labeled_pixels']} | {row['ignored_pixels']} |")
    paths["report"].write_text("\n".join(text) + "\n", encoding="utf-8")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kiểm tra độ phủ nhãn, nhóm khảo sát và ảnh trùng trước huấn luyện.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--classes", type=Path, default=Path("classes.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/audit"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    report = audit_dataset(args.manifest, args.classes, val_fraction=args.val_fraction, seed=args.seed)
    paths = save_audit(report, args.out)
    print(f"{report['status']} · {len(report['errors'])} lỗi, {len(report['warnings'])} cảnh báo.")
    print(paths["json"].resolve())
    return int(bool(report["errors"]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())

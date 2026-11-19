"""Folder inference and a reproducible queue for expert review."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from .data import load_image
from .predict import default_device, load_predictor, prediction_provenance, validate_tiles
from .render import export_results


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
EXPORT_PROFILES = ("review", "standard")
ZIP_LEVELS = {"fast": 1, "compact": 9}


def process_folder(model_path, input_dir, out_dir, *, recursive=False, device=None,
                   tile_size=None, overlap=None, batch_size=4, tta=False,
                   confidence_threshold=0.0, min_region_area=64, alpha=0.27,
                   profile="review", zip_path=None, zip_compression="fast", progress=None):
    """Process images one at a time and rank them with an explicit review heuristic.

    Ranking is 0.6 * review pixel fraction + 0.4 * mean normalized entropy.
    It estimates review priority, not error probability or geological accuracy.
    Files are loaded once in deterministic relative-path order. Output subtrees
    under the input folder are excluded. Per-image failures stay in both indices.
    The review profile includes editable annotations; standard omits that package.
    ``progress(done, total, source_relative)`` is called after each attempted image.
    """
    input_dir, out_dir = Path(input_dir).resolve(), Path(out_dir).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Không tìm thấy thư mục ảnh: {input_dir}")
    if input_dir == out_dir or input_dir.is_relative_to(out_dir):
        raise ValueError("Thư mục kết quả phải tách khỏi thư mục ảnh hoặc là thư mục con riêng.")
    if batch_size < 1 or min_region_area < 1 or not 0 <= alpha <= 1:
        raise ValueError("Batch và diện tích phải ≥1, alpha thuộc [0, 1].")
    if profile not in EXPORT_PROFILES:
        raise ValueError("Kiểu xuất cần là review hoặc standard.")
    if zip_compression not in ZIP_LEVELS:
        raise ValueError("Mức nén ZIP cần là fast hoặc compact.")
    archive = None if zip_path is None else Path(zip_path).resolve()
    if archive is not None:
        if archive.suffix.lower() != ".zip":
            raise ValueError("Tệp nén kết quả cần có phần mở rộng .zip.")
        if archive.is_relative_to(input_dir) and not archive.is_relative_to(out_dir):
            raise ValueError("Không đặt tệp ZIP trong thư mục ảnh đầu vào.")
    source_files = sorted(
        (path for path in (input_dir.rglob("*") if recursive else input_dir.iterdir())
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.is_relative_to(out_dir)),
        key=lambda path: path.relative_to(input_dir).as_posix().casefold(),
    )
    if not source_files:
        raise ValueError("Thư mục không có ảnh PNG/JPEG/BMP/TIFF/WebP hợp lệ.")
    predictor = load_predictor(model_path, device or default_device())
    provenance = prediction_provenance(
        model_path, predictor, tta=tta, confidence_threshold=confidence_threshold,
        tile_size=tile_size, overlap=overlap,
    )
    validate_tiles(provenance["tile_size"], provenance["overlap"])
    provenance["created_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["export_profile"] = profile
    results, errors = [], []
    out_dir.mkdir(parents=True, exist_ok=True)
    # Never reuse stale exports from a preceding invocation of the same folder run.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = out_dir / ("images-" + run_id)
    run_dir.mkdir()
    for index, path in enumerate(source_files, 1):
        relative = path.relative_to(input_dir).as_posix()
        stem = re.sub(r"[^\w.-]+", "_", path.stem, flags=re.UNICODE).strip(". ")[:64] or "image"
        target = run_dir / f"{index:05d}-{stem}"
        try:
            image = load_image(path)
            details = predictor.predict_detailed(
                image, tile_size=tile_size, overlap=overlap, batch_size=batch_size,
                tta=tta, confidence_threshold=confidence_threshold,
            )
            paths = export_results(
                target, image, details["mask"], details["confidence"], predictor.classes,
                min_region_area=min_region_area, alpha=alpha, raw_mask=details["raw_mask"],
                entropy=details["entropy"], review_mask=details["review_mask"],
                provenance={**provenance, "source_image": relative}, profile=profile,
            )
            metadata = json.loads(paths["regions"].read_text(encoding="utf-8"))
            labelme = metadata.get("labelme")
            mean_entropy = float(details["entropy"].mean())
            fraction = float(details["review_mask"].mean())
            results.append({
                "image": path.name, "source_relative": relative,
                "output_dir": target.relative_to(out_dir).as_posix(), "status": "ok",
                "width": image.shape[1], "height": image.shape[0],
                "mean_confidence": float(details["confidence"].mean()),
                "mean_entropy": mean_entropy, "review_fraction": fraction,
                "review_pixels": int(details["review_mask"].sum()),
                "region_count": metadata["region_count"],
                "priority_score": 0.6 * fraction + 0.4 * mean_entropy,
                "omitted_polygon_regions": len(labelme["omitted_regions"]) if labelme is not None else None,
                "paths": {key: value.relative_to(out_dir).as_posix() for key, value in paths.items()},
            })
        except (ValueError, OSError, RuntimeError) as error:
            errors.append({"image": path.name, "source_relative": relative,
                           "status": "error", "error": str(error)})
        finally:
            # Release full-size arrays before loading or predicting the next section.
            image = details = None
        if progress is not None:
            progress(index, len(source_files), relative)
    results.sort(key=lambda row: (-row["priority_score"], -row["mean_entropy"], row["source_relative"]))
    for rank, row in enumerate(results, 1):
        row["rank"] = rank
    summary = {
        "total_images": len(source_files), "successful_images": len(results), "failed_images": len(errors),
        "review_pixels": sum(row["review_pixels"] for row in results),
        "export_profile": profile,
        "priority_rule": "0.6*review_fraction + 0.4*mean_entropy (uncalibrated heuristic)",
        "provenance": provenance,
    }
    index_json, index_csv = out_dir / "index.json", out_dir / "index.csv"
    index_json.write_text(json.dumps({"summary": summary, "results": results, "errors": errors},
                                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = ["rank", "image", "source_relative", "status", "priority_score", "review_fraction",
               "mean_entropy", "mean_confidence", "review_pixels", "width", "height", "region_count",
               "omitted_polygon_regions", "output_dir", "error"]
    with index_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in [*results, *errors]:
            safe = {key: ("'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
                    for key, value in row.items()}
            writer.writerow(safe)
    if archive is not None:
        archive.parent.mkdir(parents=True, exist_ok=True)
        # Stream only files advertised in these indices into a replaceable archive.
        with NamedTemporaryFile(prefix=".facies-archive-", suffix=".tmp", dir=archive.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=ZIP_LEVELS[zip_compression]) as bundle:
                for path in (index_csv, index_json):
                    bundle.write(path, arcname=path.name)
                for row in results:
                    for relative in row["paths"].values():
                        bundle.write(out_dir / relative, arcname=relative)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    return {"results": results, "errors": errors, "summary": summary,
            "index_csv": index_csv, "index_json": index_json, "zip": archive}


def main():
    parser = argparse.ArgumentParser(description="Khoanh vùng nhiều lát cắt và xếp ưu tiên chuyên gia xem lại.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="Thư mục chỉ chứa ảnh lát cắt sạch.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--zip", dest="zip_path", type=Path)
    parser.add_argument("--zip-compression", choices=ZIP_LEVELS, default="fast",
                        help="fast rút ngắn thời gian nén, compact ưu tiên giảm dung lượng ZIP.")
    parser.add_argument("--profile", choices=EXPORT_PROFILES, default="review",
                        help="standard xuất kết quả gọn, review kèm ảnh nguồn và tệp chỉnh nhãn.")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--min-region-area", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.27)
    args = parser.parse_args()
    try:
        result = process_folder(
            args.model, args.input, args.out, recursive=args.recursive, device=args.device,
            tile_size=args.tile_size, overlap=args.overlap, batch_size=args.batch_size,
            tta=args.tta, confidence_threshold=args.confidence_threshold,
            min_region_area=args.min_region_area, alpha=args.alpha, zip_path=args.zip_path,
            profile=args.profile, zip_compression=args.zip_compression,
            progress=lambda done, total, name: print(f"[{done}/{total}] {name}", flush=True),
        )
    except (ValueError, OSError, RuntimeError, KeyError, pickle.UnpicklingError) as error:
        parser.exit(2, f"Lỗi: {error}\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(result["index_csv"])
    if result["zip"]:
        print(result["zip"])
    if result["errors"]:
        parser.exit(1, f"Có {len(result['errors'])} ảnh lỗi; xem index.json.\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()

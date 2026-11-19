"""Import a batch of registered sections with color-coded expert outlines."""

import argparse
import csv
import json
import os
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .data import IGNORE_INDEX, load_classes, load_records, split_records


IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg", ".webp"}
LOSSLESS_ANNOTATION_FORMATS = {"PNG", "TIFF", "BMP"}


def _key(value):
    return unicodedata.normalize("NFC", value.strip()).casefold()


def _images(directory):
    if not directory.is_dir():
        raise ValueError(f"Không tìm thấy thư mục {directory}.")
    paths = sorted(path.resolve() for path in directory.iterdir()
                   if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"Chưa có ảnh trong {directory}.")
    return paths


def _read_rgb(path, *, annotation=False):
    with Image.open(path) as image:
        if annotation and image.format not in LOSSLESS_ANNOTATION_FORMATS:
            raise ValueError("Cần ảnh chú giải xuất trực tiếp dưới dạng PNG, TIFF hoặc BMP giữ màu chính xác.")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Mỗi tệp cần chứa một ảnh lát cắt.")
        if "A" in image.getbands() or "transparency" in image.info:
            if image.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                raise ValueError("Hãy lưu ảnh đã gộp chú giải với lát cắt và có nền đục.")
        return np.asarray(image.convert("RGB"))


def _color_pixels(image, color):
    return (image[..., 0] == color[0]) & (image[..., 1] == color[1]) & (image[..., 2] == color[2])


def _closed_regions(stroke, min_area):
    """Yield (bounds, filled region) for single closed strokes, including their edge pixels."""
    _, components, stats, _ = cv2.connectedComponentsWithStats(stroke.astype(np.uint8), connectivity=8)
    for index, (x, y, width, height, _) in enumerate(stats[1:], start=1):
        if width < 3 or height < 3 or (width - 2) * (height - 2) < min_area:
            continue
        bounds = np.s_[y:y + height, x:x + width]
        segment = components[bounds] == index
        grid = np.pad(segment.astype(np.uint8), 1)
        # A four-connected exterior complements the eight-connected outline, including diagonals.
        cv2.floodFill(grid, None, (0, 0), 2, flags=4)
        hole = (grid[1:-1, 1:-1] == 0).astype(np.uint8)
        if cv2.countNonZero(hole) < min_area:
            continue
        _, _, hole_stats, _ = cv2.connectedComponentsWithStats(hole, connectivity=4)
        if len(hole_stats) != 2:
            raise ValueError(f"Viền tại ({x}, {y}) tạo nhiều khoang. Hãy kiểm tra vùng giao hoặc tách viền.")
        yield bounds, segment | hole.astype(bool)


def _expanded_pixels(pixels, pad):
    if pad == 0:
        return pixels
    kernel = np.ones((2 * pad + 1, 2 * pad + 1), dtype=np.uint8)
    return cv2.dilate(pixels.astype(np.uint8), kernel).astype(bool)


def _composited_ink(annotated, clean, color, candidates):
    """Recognize RGB ink alpha-blended over the registered original within 1 RGB level."""
    result = np.zeros(candidates.shape, dtype=bool)
    if not candidates.any():
        return result
    background = clean[candidates].astype(np.float32)
    observed = annotated[candidates].astype(np.float32)
    direction = np.asarray(color, dtype=np.float32) - background
    difference = observed - background
    denominator = np.einsum("ij,ij->i", direction, direction)
    alpha = np.divide(
        np.einsum("ij,ij->i", difference, direction), denominator,
        out=np.zeros_like(denominator), where=denominator > 0,
    )
    error = np.max(np.abs(difference - alpha[:, None] * direction), axis=1)
    result[candidates] = (alpha > 0) & (alpha <= 1) & (error <= 1)
    return result


def extract_outline_mask(annotated, classes, *, clean=None, text_colors=(), min_area=256,
                         markup_padding=2, outside_background=False):
    """Return (class IDs, detected markup, region counts), or reject ambiguous geometry."""
    if clean is not None and clean.shape != annotated.shape:
        raise ValueError("Ảnh sạch và ảnh chú giải khác kích thước.")
    changed = np.any(annotated != clean, axis=2) if clean is not None else None
    mask = np.zeros(annotated.shape[:2], dtype=np.uint8)
    markup = np.zeros_like(mask, dtype=bool)
    counts = {}
    residual = []
    for label in classes[1:]:
        class_id = label["id"]
        stroke = _color_pixels(annotated, label["color"])
        if changed is not None:
            stroke &= changed
        if not stroke.any():
            continue
        markup |= stroke
        count = 0
        try:
            for bounds, region in _closed_regions(stroke, min_area):
                target = mask[bounds]
                if np.any(target[region] != 0):
                    raise ValueError("Nhiều viền kín có phần trong chồng nhau hoặc lồng nhau.")
                target[region] = class_id
                count += 1
        except ValueError as exc:
            raise ValueError(f"{label['name']} · {exc}") from exc
        outside = stroke & (mask != class_id)
        if outside.any():
            y, x = divmod(int(np.argmax(outside)), outside.shape[1])
            residual.append(f"{label['name']} gần ({x}, {y})")
        if count:
            counts[label["name"]] = count
    if residual:
        raise ValueError("Có nét màu nằm ngoài viền kín tương ứng hoặc vùng quá nhỏ. " + ", ".join(residual) + ".")
    if not counts:
        raise ValueError("Không tìm thấy viền kín theo màu trong bảng lớp. Hãy kiểm tra màu viền và tướng ghi trên ảnh.")

    for color in text_colors:
        text = _color_pixels(annotated, color)
        if changed is not None:
            text &= changed
        if np.any(text & (mask == 0)):
            raise ValueError(f"Màu chữ {color} xuất hiện ngoài các vùng viền kín.")
        markup |= text
    expanded = _expanded_pixels(markup, markup_padding)
    if changed is not None:
        explained = np.zeros_like(markup)
        for label in classes[1:]:
            # Geometry still comes only from exact-color closed rings. Blended text can
            # contain no full-opacity pixels, so proximity to an exact text core is not required.
            allowed = _expanded_pixels(mask == label["id"], markup_padding) & changed
            explained |= _composited_ink(annotated, clean, label["color"], allowed)
        for color in text_colors:
            explained |= _composited_ink(annotated, clean, color, (mask != 0) & changed)
        extra = changed & ~explained
        if extra.any():
            y, x = divmod(int(np.argmax(extra)), extra.shape[1])
            raise ValueError(f"Ảnh sạch và ảnh chú giải khác nhau ngoài vết vẽ đã biết gần ({x}, {y}). "
                             "Pixel cần khớp màu mực đã khai báo hoặc màu đó pha alpha với ảnh sạch "
                             "trong sai số tối đa 1 mức RGB. Kiểm tra cặp ảnh, màu viền và màu chữ.")
        markup = changed
    else:
        chroma = annotated.max(axis=2).astype(np.int16) - annotated.min(axis=2).astype(np.int16)
        other_colors = (chroma > 8) & ~expanded
        if other_colors.any():
            y, x = divmod(int(np.argmax(other_colors)), other_colors.shape[1])
            raise ValueError(f"Có màu ngoài bảng viền/chữ gần ({x}, {y}). "
                             "Hãy thêm đúng màu chữ hoặc cung cấp ảnh sạch đã khớp tọa độ. "
                             "Chữ hoặc viền có làm mịn màu cần ảnh sạch để đối chiếu pha alpha.")
        markup = expanded
    if not outside_background:
        mask[mask == 0] = IGNORE_INDEX
    return mask, markup, counts


def _save_preview(path, annotated, mask, classes, *, clean=None):
    tint = np.array(annotated if clean is None else clean, copy=True)
    for label in classes[1:]:
        selected = mask == label["id"]
        tint[selected] = np.rint(0.50 * tint[selected] + 0.50 * np.array(label["color"], dtype=np.uint8))
    pair = [Image.fromarray(source) for source in (annotated, tint)]
    for pane in pair:
        pane.thumbnail((1200, 1000), Image.Resampling.LANCZOS)
    preview = Image.new("RGB", (2 * pair[0].width, pair[0].height))
    preview.paste(pair[0], (0, 0))
    preview.paste(pair[1], (pair[0].width, 0))
    preview.save(path)


def _group_rows(path):
    if path is None:
        return {}
    groups = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if not {"image", "group"} <= set(rows.fieldnames or []):
            raise ValueError("Bảng nhóm cần các cột image, group và có thể thêm split.")
        for row in rows:
            image, group, split = ((row.get(name) or "").strip() for name in ("image", "group", "split"))
            if not image or not group or Path(image).name != image or _key(image) in groups:
                raise ValueError(f"Tên tệp hoặc nhóm không hợp lệ ở dòng {rows.line_num} của {path}.")
            groups[_key(image)] = group, split
    return groups


def _relative_to(path, directory):
    try:
        return Path(os.path.relpath(path, directory)).as_posix()
    except ValueError:  # On Windows, relative paths do not cross drives.
        return path.as_posix()


def _write_manifest(out, accepted, groups):
    missing = [row["source"] for row in accepted if _key(Path(row["source"]).name) not in groups]
    if missing:
        raise ValueError(f"Thiếu thông tin nhóm cho {len(missing)} ảnh, ví dụ {Path(missing[0]).name}.")
    manifest = out / "dataset.csv"
    temporary = out / "dataset.csv.tmp"
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("image", "mask", "group", "split"))
            for row in accepted:
                group, split = groups[_key(Path(row["source"]).name)]
                writer.writerow((
                    _relative_to(Path(row["training_image"]), out), row["mask"], group, split,
                ))
        split_records(load_records(temporary))
        temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest.name


def _import_one(args, source, clean_paths, classes, out):
    clean_path = None
    if clean_paths is not None:
        matches = clean_paths.get(_key(source.stem), [])
        if len(matches) != 1:
            raise ValueError("Cần đúng một ảnh sạch có cùng tên gốc trong thư mục --clean.")
        clean_path = matches[0]
        if clean_path == source:
            raise ValueError("Ảnh sạch và ảnh chú giải đang trỏ đến cùng tệp.")
    annotated = _read_rgb(source, annotation=True)
    clean = _read_rgb(clean_path) if clean_path else None
    mask, markup, counts = extract_outline_mask(
        annotated, classes, clean=clean, text_colors=args.text_color, min_area=args.min_area,
        markup_padding=args.markup_padding, outside_background=args.outside_background,
    )
    if args.inpaint:
        clean = cv2.inpaint(annotated, markup.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        mask[markup] = IGNORE_INDEX
        if not np.any((mask > 0) & (mask != IGNORE_INDEX)):
            raise ValueError("Không còn pixel tướng sau khi bỏ qua vết nội suy.")
        clean_path = out / "images" / f"{source.stem}.png"
        Image.fromarray(clean).save(clean_path)
    mask_path = out / "labels" / f"{source.stem}.png"
    preview_path = out / "review" / f"{source.stem}.png"
    Image.fromarray(mask).save(mask_path)
    _save_preview(preview_path, annotated, mask, classes, clean=clean)
    return {
        "source": str(source), "status": "converted",
        "training_image": str(clean_path) if clean_path else None,
        "mask": mask_path.relative_to(out).as_posix(), "review": preview_path.relative_to(out).as_posix(),
        "regions_by_class": counts, "markup_pixels": int(markup.sum()),
        "ignored_pixels": int(np.count_nonzero(mask == IGNORE_INDEX)),
    }


def import_batch(args):
    classes = load_classes(args.classes)
    outline_colors = {tuple(label["color"]) for label in classes[1:]}
    if outline_colors & set(args.text_color):
        raise ValueError("Chỉ dùng --text-color cho màu chữ khác màu viền.")
    if not args.clean and any(max(color) - min(color) <= 8 for color in outline_colors | set(args.text_color)):
        raise ValueError("Màu viền/chữ gần thang xám không thể tách chắc chắn khỏi địa chấn khi thiếu --clean.")
    if args.min_area < 1 or not 0 <= args.markup_padding <= 32:
        raise ValueError("--min-area cần lớn hơn 0 và --markup-padding cần nằm trong khoảng 0–32.")
    sources = _images(args.annotated)
    if len({_key(source.stem) for source in sources}) != len(sources):
        raise ValueError("Ảnh chú giải cần tên gốc duy nhất để lưu mask và ảnh kiểm tra.")
    clean_paths = None
    if args.clean:
        clean_paths = {}
        for path in _images(args.clean):
            clean_paths.setdefault(_key(path.stem), []).append(path)
    groups = _group_rows(args.groups)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Hãy chọn thư mục đầu ra trống cho lượt nhập này. {out}")
    for name in ("labels", "review") + (("images",) if args.inpaint else ()):
        (out / name).mkdir()

    results, accepted = [], []
    for source in sources:
        try:
            row = _import_one(args, source, clean_paths, classes, out)
        except (ValueError, OSError, cv2.error, Image.DecompressionBombError) as exc:
            row = {"source": str(source), "status": "needs_review", "reason": str(exc)}
        else:
            accepted.append(row)
        results.append(row)
        print(f"{row['status']} · {source.name}", flush=True)

    manifest, manifest_error = None, None
    if accepted and (args.clean or args.inpaint) and args.groups:
        try:
            manifest = _write_manifest(out, accepted, groups)
        except (OSError, ValueError) as exc:
            manifest_error = str(exc)
    mode = "clean" if args.clean else "inpaint" if args.inpaint else "mask_only"
    report = {
        "mode": mode, "review_required": True,
        "manual_review": "So sánh ảnh chú giải bên trái với mask phủ màu bên phải trong thư mục review trước khi huấn luyện.",
        "converted": len(accepted), "needs_review": len(results) - len(accepted),
        "manifest": manifest, "manifest_error": manifest_error, "results": results,
    }
    if mode == "mask_only":
        report["manifest_note"] = "Chỉ xuất mask để rà soát. Cần ảnh sạch hoặc chọn --inpaint để chuẩn bị đầu vào huấn luyện."
    if not args.groups:
        report["group_note"] = "Chưa có bảng nhóm nên chưa xuất dataset.csv. Hãy nhóm theo tuyến/khối để tránh rò rỉ dữ liệu."
    if mode == "inpaint":
        report["input_quality"] = ("Ảnh mới chỉ nội suy tại các màu vẽ phát hiện được trên nền địa chấn thang xám. "
                                   "Không phát hiện chắc chắn chữ đen/xám hòa với địa chấn. "
                                   "Cần rà soát chữ sót và đánh giá bằng ảnh sạch độc lập.")
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(accepted)} ảnh đã chuyển, {report['needs_review']} ảnh cần kiểm tra · {report_path}", flush=True)
    if manifest:
        print(out / manifest, flush=True)
    if manifest_error:
        print(f"Chưa tạo được manifest · {manifest_error}", file=sys.stderr)
    return int(bool(report["needs_review"] or manifest_error))


def _rgb_argument(value):
    try:
        channels = tuple(int(part) for part in value.split(","))
        if len(channels) == 3 and all(0 <= channel <= 255 for channel in channels):
            return channels
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Dùng R,G,B với các số nguyên từ 0 đến 255.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Chuyển loạt ảnh chú giải có viền kín và mỗi tướng một màu thành mask ID.",
        epilog=("Không đọc tên tướng bằng OCR. Viền một màu và tên tướng chỉ ghi bằng chữ cần gán nhãn thủ công. "
                "Khi thiếu --clean, yêu cầu nền địa chấn thang xám và màu viền/chữ riêng biệt; "
                "không nhận biết chắc chắn chữ đen/xám. --inpaint chỉ khôi phục gần đúng và cần rà soát từng ảnh."),
    )
    parser.add_argument("--annotated", required=True, type=Path, help="Thư mục ảnh chú giải PNG, TIFF hoặc BMP.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--clean", type=Path, help="Thư mục ảnh sạch có cùng tên gốc và cùng lưới pixel.")
    mode.add_argument("--inpaint", action="store_true", help="Xóa màu vẽ đã biết bằng nội suy và bỏ qua pixel đã xử lý.")
    parser.add_argument("--classes", type=Path, default=Path("classes.json"), help="Bảng ID, tên và màu viền RGB.")
    parser.add_argument("--groups", type=Path, help="CSV image,group,split với image là tên tệp chú giải có phần mở rộng.")
    parser.add_argument("--out", type=Path, required=True, help="Thư mục đầu ra trống.")
    parser.add_argument("--text-color", type=_rgb_argument, default=[], action="append", metavar="R,G,B",
                        help="Màu chữ khác màu viền. Có thể dùng tùy chọn này nhiều lần.")
    parser.add_argument("--min-area", type=int, default=256, help="Số pixel tối thiểu ở phần trong của viền kín.")
    parser.add_argument("--markup-padding", type=int, default=2, help="Bán kính pixel để xử lý mép vết vẽ.")
    parser.add_argument("--outside-background", action="store_true", help="Dùng khi chuyên gia đã xác nhận nền ngoài viền.")
    args = parser.parse_args(argv)
    try:
        return import_batch(args)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"Lỗi · {exc}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())

"""Load aligned seismic sections, expert masks and survey group assignments."""

import csv
import json
import math
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IGNORE_INDEX = 255
IGNORE_LABEL = "__ignore__"


@dataclass(frozen=True)
class Record:
    image: Path
    mask: Path
    group: str
    split: str = ""


def _key(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip()).casefold()


def validate_classes(raw) -> list[dict]:
    """Return validated class definitions sorted by numeric ID."""
    if not isinstance(raw, list) or not 2 <= len(raw) <= IGNORE_INDEX:
        raise ValueError("Danh sách lớp phải có từ 2 đến 255 mục, gồm ID 0 cho nền.")

    classes = []
    ids, colors, labels = set(), set(), {IGNORE_LABEL}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Mỗi lớp phải có id, name và color [R, G, B].")
        class_id = item.get("id")
        name = item.get("name")
        color = item.get("color")
        aliases = item.get("aliases", [])
        if type(class_id) is not int or not 0 <= class_id < len(raw) or class_id in ids:
            raise ValueError(f"ID lớp phải duy nhất và liên tiếp từ 0. ID gặp được: {class_id!r}.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Tên lớp {class_id} bị trống hoặc không hợp lệ.")
        if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
            raise ValueError(f"aliases của lớp {class_id} phải là danh sách tên hợp lệ.")
        if (
            not isinstance(color, (tuple, list))
            or len(color) != 3
            or any(type(channel) is not int or not 0 <= channel <= 255 for channel in color)
            or tuple(color) in colors
        ):
            raise ValueError(f"Màu của lớp {class_id} phải là bộ [R, G, B] duy nhất.")
        names = {_key(label) for label in [name, *aliases]}
        if labels.intersection(names):
            raise ValueError(f"Tên hoặc bí danh trùng hoặc dành riêng cho lớp {class_id}: {name!r}.")
        labels.update(names)
        ids.add(class_id)
        colors.add(tuple(color))
        definition = {"id": class_id, "name": name.strip(), "color": list(color)}
        if aliases:
            definition["aliases"] = [alias.strip() for alias in aliases]
        classes.append(definition)

    return sorted(classes, key=lambda item: item["id"])


def load_classes(path: str | Path) -> list[dict]:
    path = Path(path)
    try:
        return validate_classes(json.loads(path.read_text(encoding="utf-8-sig")))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Không đọc được bảng lớp {path}: {exc}") from exc


def load_records(manifest: str | Path) -> list[Record]:
    """Read CSV columns image, mask, group and optional split."""
    manifest = Path(manifest).resolve()
    records = []
    images = set()
    try:
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            headers = [name.strip() for name in (rows.fieldnames or [])]
            if len(set(headers)) != len(headers) or not {"image", "mask", "group"} <= set(headers):
                raise ValueError("CSV cần các cột duy nhất image, mask, group; split là tùy chọn.")
            rows.fieldnames = headers
            for row in rows:
                if None in row:
                    raise ValueError(f"Dòng {rows.line_num} có nhiều giá trị hơn số cột.")
                image, mask, group, split = (
                    (row.get(column) or "").strip() for column in ("image", "mask", "group", "split")
                )
                if not all((image, mask, group)):
                    raise ValueError(f"Dòng {rows.line_num} thiếu image, mask hoặc group.")
                split = {"validation": "val", "valid": "val"}.get(_key(split), _key(split))
                if split not in {"", "train", "val", "test"}:
                    raise ValueError(f"Dòng {rows.line_num}: split phải là train, val hoặc test.")
                image_path, mask_path = ((manifest.parent / value).resolve() for value in (image, mask))
                for path in (image_path, mask_path):
                    if not path.is_file():
                        raise ValueError(f"Dòng {rows.line_num} không tìm thấy tệp {path}.")
                if image_path == mask_path:
                    raise ValueError(f"Dòng {rows.line_num} dùng cùng tệp cho ảnh và mask.")
                if image_path in images:
                    raise ValueError(f"Dòng {rows.line_num} lặp ảnh {image_path}.")
                images.add(image_path)
                records.append(Record(image_path, mask_path, group, split))
    except (OSError, ValueError, csv.Error) as exc:
        raise ValueError(f"Manifest {manifest}: {exc}") from exc
    if not records:
        raise ValueError(f"Manifest {manifest} chưa có lát cắt nào.")
    return records


def split_records(
    records: list[Record], val_fraction: float = 0.2, seed: int = 42
) -> tuple[list[Record], list[Record], list[Record]]:
    """Return train, validation and optional test lists without splitting any group."""
    if any(record.split for record in records):
        partitions = {"train": [], "val": [], "test": []}
        assigned = {}
        for record in records:
            if record.split not in partitions:
                raise ValueError("Khi dùng cột split, mọi dòng cần ghi train, val hoặc test.")
            group = _key(record.group)
            if assigned.setdefault(group, record.split) != record.split:
                raise ValueError(f"Nhóm {record.group!r} xuất hiện trong nhiều tập.")
            partitions[record.split].append(record)
        if not partitions["train"] or not partitions["val"]:
            raise ValueError("Cột split phải có ít nhất một nhóm train và một nhóm val.")
        return partitions["train"], partitions["val"], partitions["test"]

    groups = sorted({_key(record.group) for record in records})
    if len(groups) < 2:
        raise ValueError("Cần ít nhất hai nhóm khảo sát hoặc tuyến độc lập để chia train và val.")
    if not 0 < val_fraction < 1:
        raise ValueError("Tỷ lệ val phải lớn hơn 0 và nhỏ hơn 1.")
    random.Random(seed).shuffle(groups)
    count = max(1, min(len(groups) - 1, round(len(groups) * val_fraction)))
    val_groups = set(groups[:count])
    train, val = [], []
    for record in records:
        (val if _key(record.group) in val_groups else train).append(record)
    return train, val, []


def load_image(path) -> np.ndarray:
    """Read an image path or binary stream as float32 RGB HWC in the range [0, 1]."""
    try:
        with Image.open(path) as image:
            # Preserve the original orientation so expert pixel coordinates still align.
            if image.mode.startswith("I;16"):
                plane = np.asarray(image, dtype=np.float32) / 65535
            elif image.mode in {"I", "F"}:
                values = np.asarray(image, dtype=np.float32)
                finite = values[np.isfinite(values)]
                if not finite.size:
                    raise ValueError("Ảnh không chứa giá trị pixel hữu hạn.")
                low, high = np.percentile(finite, (0.5, 99.5))
                if high <= low:
                    low, high = finite.min(), finite.max()
                plane = np.zeros_like(values)
                if high > low:
                    plane = np.clip((values - low) / (high - low), 0, 1)
                    plane[~np.isfinite(values)] = 0
            else:
                return np.asarray(image.convert("RGB"), dtype=np.float32) / 255
            return np.repeat(plane[..., None], 3, axis=-1).astype(np.float32, copy=False)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        source = getattr(path, "name", path)
        raise ValueError(f"Không đọc được ảnh {source}: {exc}") from exc


def _polygon_mask(path: Path, classes: list[dict], size: tuple[int, int]) -> np.ndarray:
    annotation = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(annotation, dict) or not isinstance(annotation.get("shapes"), list):
        raise ValueError("JSON LabelMe phải có danh sách shapes.")
    for key, expected in zip(("imageWidth", "imageHeight"), size):
        if annotation.get(key) not in (None, expected):
            raise ValueError(f"{key} trong JSON khác kích thước ảnh sạch {size}.")

    labels = {IGNORE_LABEL: IGNORE_INDEX}
    for item in classes:
        for label in [item["name"], *item.get("aliases", [])]:
            labels[_key(label)] = item["id"]
    mask = Image.new("L", size, IGNORE_INDEX)
    draw = ImageDraw.Draw(mask)
    annotation_flags = annotation.get("flags")
    machine_document = isinstance(annotation_flags, dict) and annotation_flags.get("machine_prediction") is True
    for number, shape in enumerate(annotation["shapes"], 1):
        if not isinstance(shape, dict):
            raise ValueError(f"Vùng {number} trong JSON không hợp lệ.")
        shape_flags = shape.get("flags")
        shape_flags = shape_flags if isinstance(shape_flags, dict) else {}
        if machine_document or shape_flags.get("machine_prediction") is True:
            # Global edit flags do not turn untouched proposals into expert ground truth.
            if shape_flags.get("reviewed") is not True:
                continue
        label = shape.get("label")
        if not isinstance(label, str) or _key(label) not in labels:
            raise ValueError(f"Vùng {number} có nhãn chưa khai báo {label!r}.")
        kind = shape.get("shape_type") or "polygon"
        points = shape.get("points")
        if kind not in ("polygon", "rectangle"):
            raise ValueError(f"Vùng {number} có kiểu {kind!r}. Hãy dùng polygon hoặc rectangle.")
        if not isinstance(points, list) or (len(points) < 3 if kind == "polygon" else len(points) != 2):
            raise ValueError(f"Vùng {number} không đủ điểm hoặc sai số điểm.")
        if any(
            not isinstance(point, list)
            or len(point) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in point)
            for point in points
        ):
            raise ValueError(f"Tọa độ của vùng {number} trong JSON không hợp lệ.")
        fill = labels[_key(label)]
        if kind == "rectangle":
            (x1, y1), (x2, y2) = points
            draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), fill=fill)
        else:
            draw.polygon([tuple(point) for point in points], fill=fill)
    return np.array(mask)


def _raster_mask(path: Path, classes: list[dict], size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError(f"Mask có kích thước {image.size}, ảnh sạch có kích thước {size}.")
        if image.format == "JPEG":
            raise ValueError("Hãy lưu mask dưới dạng PNG hoặc TIFF để giữ nguyên giá trị nhãn.")
        transparent = None
        if image.mode in {"RGB", "RGBA", "LA"}:
            values = np.asarray(image)
            if image.mode in {"RGBA", "LA"}:
                alpha = values[..., -1]
                if np.any((alpha != 0) & (alpha != 255)):
                    raise ValueError("Mask có pixel bán trong suốt. Hãy xuất mask với alpha 0 hoặc 255.")
                transparent = alpha == 0
            if image.mode == "LA":
                result = values[..., 0].copy()
                if transparent is not None:
                    result[transparent] = IGNORE_INDEX
                return _validate_ids(result, len(classes))

            result = np.full(values.shape[:2], IGNORE_INDEX, dtype=np.uint8)
            rgb = values[..., :3]
            known = transparent.copy() if transparent is not None else np.zeros(values.shape[:2], dtype=bool)
            for item in classes:
                red, green, blue = item["color"]
                matches = (rgb[..., 0] == red) & (rgb[..., 1] == green) & (rgb[..., 2] == blue)
                result[matches] = item["id"]
                known |= matches
            if not known.all():
                row, column = divmod(int(np.argmin(known)), known.shape[1])
                color = tuple(map(int, rgb[row, column]))
                raise ValueError(f"Màu {color} tại ({column}, {row}) không có trong bảng lớp.")
            if transparent is not None:
                result[transparent] = IGNORE_INDEX
            return result
        if image.mode not in {"1", "L", "P", "I"} and not image.mode.startswith("I;16"):
            raise ValueError(f"Kiểu ảnh mask {image.mode!r} không hỗ trợ.")
        return _validate_ids(np.asarray(image), len(classes))


def _validate_ids(values: np.ndarray, class_count: int) -> np.ndarray:
    invalid = ((values < 0) | (values >= class_count)) & (values != IGNORE_INDEX)
    if invalid.any():
        sample = np.unique(values[invalid])[:10].tolist()
        raise ValueError(f"Mask có ID ngoài bảng lớp: {sample}.")
    return values.astype(np.uint8, copy=True)


def load_mask(
    record: Record, classes: list[dict], *, image_size: tuple[int, int] | None = None
) -> np.ndarray:
    """Read a uint8 HW mask aligned with the clean section. ID 255 is ignored."""
    path = Path(record.mask)
    try:
        if image_size is None:
            with Image.open(record.image) as image:
                image_size = image.size
        if path.suffix.casefold() == ".json":
            return _polygon_mask(path, classes, image_size)
        return _raster_mask(path, classes, image_size)
    except (OSError, TypeError, ValueError, OverflowError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Không đọc được nhãn {path} cho ảnh {record.image}: {exc}") from exc

"""Content-based dataset identity and atomic experiment reports."""

import hashlib
import json
import unicodedata
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(partitions, classes):
    rows = []
    seen = {}
    for split, records in partitions.items():
        for record in records:
            image_hash = file_sha256(record.image)
            if image_hash in seen and seen[image_hash] != split:
                raise ValueError(f"Ảnh trùng nội dung trong {seen[image_hash]} và {split}: {record.image}")
            seen[image_hash] = split
            rows.append({
                "split": split,
                "group": unicodedata.normalize("NFC", record.group.strip()).casefold(),
                "image": image_hash, "mask": file_sha256(record.mask),
            })
    payload = {"classes": classes, "sections": sorted(rows, key=lambda r: json.dumps(r, sort_keys=True))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

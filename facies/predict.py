import argparse
import hashlib
import pickle
import sys
from itertools import islice, product
from pathlib import Path

import numpy as np
import torch

from .data import load_image, validate_classes


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_device(device, cpu_threads=None):
    device = torch.device(device)
    if device.type == "cpu":
        threads = min(4, torch.get_num_threads()) if cpu_threads is None else cpu_threads
        if threads < 1:
            raise ValueError("Số luồng CPU cần lớn hơn 0.")
        torch.set_num_threads(threads)
    return device


def validate_tiles(tile_size, overlap):
    if tile_size < 32 or tile_size % 8:
        raise ValueError("Kích thước ô cần từ 32 pixel và là bội số của 8.")
    if not 0 <= overlap < tile_size:
        raise ValueError("Phần chồng lấp cần từ 0 và nhỏ hơn kích thước ô.")


def normalize(image):
    return image * 2.0 - 1.0


def _positions(length, tile, step):
    last = length - tile
    return list(range(0, last, step)) + [last]


class Predictor:
    def __init__(self, model, classes, tile_size, device, *, demo=False, active_ids=None):
        validate_tiles(tile_size, 0)
        self.model = model.eval()
        self.classes = classes
        self.tile_size = tile_size
        self.device = torch.device(device)
        self.demo = demo
        ids = list(range(len(classes))) if active_ids is None else active_ids
        if (
            not isinstance(ids, (list, tuple)) or len(ids) < 2
            or any(type(class_id) is not int or not 0 <= class_id < len(classes) for class_id in ids)
            or sorted(set(ids)) != list(ids)
        ):
            raise ValueError("Danh sách ID được huấn luyện trong checkpoint không hợp lệ.")
        self.active_ids = np.array(ids, dtype=np.uint8)
        self.active_channels = None if len(ids) == len(classes) else list(ids)

    def predict(self, image, *, tile_size=None, overlap=None, batch_size=4):
        """Backward-compatible class IDs and uncalibrated maximum probability."""
        return self._predict(image, tile_size=tile_size, overlap=overlap, batch_size=batch_size)

    def predict_detailed(self, image, *, tile_size=None, overlap=None, batch_size=4,
                         tta=False, confidence_threshold=0.0):
        """Predict with normalized entropy and an explicit manual-review mask.

        Scores are model probabilities, not calibrated geological correctness.
        Entropy is normalized by log(number of trained/active classes).
        Horizontal TTA mirrors each tile, reverses that prediction, then averages
        probabilities before stitching. Vertical flips and rotations are not used.
        """
        if not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("Ngưỡng xem lại phải thuộc [0, 1].")
        raw_mask, confidence, entropy = self._predict(
            image, tile_size=tile_size, overlap=overlap, batch_size=batch_size,
            tta=tta, detailed=True,
        )
        review = confidence < confidence_threshold
        mask = raw_mask.copy()
        mask[review] = 255
        return {"mask": mask, "raw_mask": raw_mask, "confidence": confidence,
                "entropy": entropy, "review_mask": review}

    def _tile_probabilities(self, image, rows, cols, tile, batch_size, tta):
        # Fill batches across row boundaries, including when the section is one tile wide.
        positions = product(rows, cols)
        while batch := list(islice(positions, batch_size)):
            chips = np.stack([image[y:y + tile, x:x + tile] for y, x in batch])
            chips *= 2.0
            chips -= 1.0
            tensor = torch.from_numpy(chips).permute(0, 3, 1, 2).to(self.device)
            logits = self.model(tensor).float()
            if self.active_channels is not None:
                logits = logits[:, self.active_channels]
            probabilities = logits.softmax(1)
            if tta:
                flipped = self.model(tensor.flip(-1)).float()
                if self.active_channels is not None:
                    flipped = flipped[:, self.active_channels]
                probabilities.add_(flipped.softmax(1).flip(-1)).mul_(0.5)
            yield from probabilities.cpu().numpy()

    @torch.inference_mode()
    def _predict(self, image, *, tile_size=None, overlap=None, batch_size=4,
                 tta=False, detailed=False):
        tile = self.tile_size if tile_size is None else tile_size
        overlap = tile // 4 if overlap is None else overlap
        validate_tiles(tile, overlap)
        if batch_size < 1:
            raise ValueError("Batch cần lớn hơn 0.")
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or not all(image.shape[:2]):
            raise ValueError("Cần ảnh RGB có ít nhất một pixel.")
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32, copy=False)
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("Giá trị ảnh float cần nằm trong khoảng [0, 1].")
        height, width = image.shape[:2]
        padded = image
        if height < tile or width < tile:
            padded = np.pad(
                image, ((0, max(0, tile - height)), (0, max(0, tile - width)), (0, 0)), mode="edge",
            )
        step = tile - overlap
        rows = _positions(padded.shape[0], tile, step)
        cols = _positions(padded.shape[1], tile, step)
        window = np.maximum(np.hanning(tile).astype(np.float32), 0.1)
        window = np.outer(window, window)
        # Keep only the rows that future tiles can still overlap.
        votes = np.zeros((len(self.active_ids), tile, padded.shape[1]), dtype=np.float32)
        total = np.zeros((tile, padded.shape[1]), dtype=np.float32)
        mask = np.empty((height, width), dtype=np.uint8)
        confidence = np.empty((height, width), dtype=np.float32)
        entropy = np.empty((height, width), dtype=np.float32) if detailed else None
        predictions = self._tile_probabilities(padded, rows, cols, tile, batch_size, tta)
        for row_index, y in enumerate(rows):
            for x in cols:
                votes[:, :, x:x + tile] += next(predictions) * window
                total[:, x:x + tile] += window
            end = rows[row_index + 1] if row_index + 1 < len(rows) else height
            count = end - y
            completed = votes[:, :count, :width]
            winners = completed.argmax(0)[None]
            mask[y:end] = self.active_ids[winners[0]]
            confidence[y:end] = np.take_along_axis(completed, winners, axis=0)[0] / total[:count, :width]
            if not np.isfinite(confidence[y:end]).all():
                raise ValueError("Mô hình tạo xác suất không hữu hạn. Hãy kiểm tra checkpoint.")
            if detailed:
                probabilities = completed / total[:count, :width]
                entropy[y:end] = -np.sum(
                    probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=0,
                ) / np.log(len(self.active_ids))
            if end < height:
                votes[:, :-count] = votes[:, count:].copy()
                total[:-count] = total[count:].copy()
                votes[:, -count:] = 0
                total[-count:] = 0
        np.clip(confidence, 0, 1, out=confidence)
        if detailed:
            return mask, confidence, entropy.clip(0, 1)
        return mask, confidence


def load_predictor(path, device=None):
    from .model import build_model

    device = configure_device(device or default_device())
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Checkpoint không tương thích. Dùng best.pt tạo bởi facies.train.")
    classes = validate_classes(checkpoint["classes"])
    arch = checkpoint.get("arch", "unet")
    model = build_model(arch, len(classes), width=int(checkpoint["width"])).to(device)
    model.load_state_dict(checkpoint["state"])
    predictor = Predictor(
        model, classes, int(checkpoint["tile_size"]), device,
        demo=bool(checkpoint.get("demo", False)), active_ids=checkpoint.get("active_ids"),
    )
    predictor.arch = arch
    return predictor


def model_fingerprint(path):
    """Content fingerprint; a path or mtime alone does not identify a model."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_provenance(model_path, predictor, *, tta=False, confidence_threshold=0.0,
                          tile_size=None, overlap=None):
    tile = predictor.tile_size if tile_size is None else tile_size
    overlap = tile // 4 if overlap is None else overlap
    validate_tiles(tile, overlap)
    if not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("Ngưỡng xem lại phải thuộc [0, 1].")
    return {
        "model_file": Path(model_path).name,
        "model_sha256": model_fingerprint(model_path),
        "architecture": getattr(predictor, "arch", "unet"),
        "demo": bool(predictor.demo),
        "trained_class_ids": predictor.active_ids.tolist(),
        "tile_size": tile, "overlap": overlap,
        "horizontal_flip_tta": bool(tta), "confidence_threshold": float(confidence_threshold),
        "confidence_note": "Uncalibrated model probabilities; not geological accuracy.",
        "entropy_note": "Shannon entropy / log(number of trained classes), range [0,1].",
    }


def main():
    from .render import export_results

    parser = argparse.ArgumentParser(description="Khoanh vùng tướng địa chấn trên ảnh lát cắt.")
    parser.add_argument("--model", required=True, type=Path, metavar="BEST.PT")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-region-area", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.27)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--tta", action="store_true", help="Lấy trung bình xác suất khi lật ngang ô ảnh.")
    parser.add_argument("--confidence-threshold", type=float, default=0.0,
                        help="Pixel dưới ngưỡng được đánh dấu 255, cần chuyên gia xem lại.")
    args = parser.parse_args()
    try:
        predictor = load_predictor(args.model, args.device)
        image = load_image(args.image)
        if predictor.demo:
            print("Đang dùng mô hình huấn luyện trên dữ liệu mô phỏng.")
        details = predictor.predict_detailed(
            image, tile_size=args.tile_size, overlap=args.overlap, batch_size=args.batch_size,
            tta=args.tta, confidence_threshold=args.confidence_threshold,
        )
        provenance = prediction_provenance(
            args.model, predictor, tta=args.tta, confidence_threshold=args.confidence_threshold,
            tile_size=args.tile_size, overlap=args.overlap,
        )
        provenance["source_image"] = args.image.name
        results = export_results(
            args.out, image, details["mask"], details["confidence"], predictor.classes,
            min_region_area=args.min_region_area, alpha=args.alpha,
            raw_mask=details["raw_mask"], entropy=details["entropy"],
            review_mask=details["review_mask"], provenance=provenance,
        )
    except (ValueError, OSError, RuntimeError, KeyError, pickle.UnpicklingError) as error:
        parser.exit(2, f"Lỗi: {error}\n")
    for path in results.values():
        print(path.resolve())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()

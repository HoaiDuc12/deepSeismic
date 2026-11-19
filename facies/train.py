import argparse
import json
import sys
import time
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import IGNORE_INDEX, load_classes, load_image, load_mask, load_records, split_records
from .model import build_model
from .experiment import dataset_fingerprint, write_json
from .metrics import confusion_metrics
from .predict import Predictor, configure_device, default_device, load_predictor, normalize, validate_tiles


def _read_pair(record, classes):
    image = load_image(record.image)
    height, width = image.shape[:2]
    mask = load_mask(record, classes, image_size=(width, height))
    return image, mask


def _audit(records, classes):
    counts = np.zeros(len(classes), dtype=np.int64)
    for record in records:
        _, mask = _read_pair(record, classes)
        labels = mask[mask != IGNORE_INDEX]
        if not labels.size:
            raise ValueError(f"Ảnh không có pixel được gán nhãn: {record.image}")
        counts += np.bincount(labels, minlength=len(classes))
    return counts


class TrainingPatches(Dataset):
    def __init__(self, records, classes, tile, patches_per_image, seed, sampling="balanced"):
        self.records = records
        self.classes = classes
        self.tile = tile
        self.patches_per_image = patches_per_image
        self.rng = np.random.default_rng(seed)
        self.sampling = sampling
        self.read = lru_cache(maxsize=6)(self._read)

    def _read(self, index):
        image, mask = _read_pair(self.records[index], self.classes)
        sparse = mask[::4, ::4]
        foreground = np.argwhere((sparse != 0) & (sparse != IGNORE_INDEX)) * 4
        if not len(foreground):
            foreground = np.argwhere((mask != 0) & (mask != IGNORE_INDEX))
        valid = np.argwhere(sparse != IGNORE_INDEX) * 4
        if not len(valid):
            valid = np.argwhere(mask != IGNORE_INDEX)
        class_points = {}
        for class_id in np.unique(mask):
            if class_id in (0, IGNORE_INDEX):
                continue
            points = np.argwhere(sparse == class_id) * 4
            if not len(points):
                points = np.argwhere(mask == class_id)
            class_points[int(class_id)] = points
        height, width = mask.shape
        pad = ((0, max(0, self.tile - height)), (0, max(0, self.tile - width)))
        if pad[0][1] or pad[1][1]:
            image = np.pad(image, (*pad, (0, 0)), mode="edge")
            mask = np.pad(mask, pad, constant_values=IGNORE_INDEX)
        return image, mask, foreground, valid, class_points

    def __len__(self):
        return len(self.records) * self.patches_per_image

    def __getitem__(self, index):
        image, mask, foreground, valid, class_points = self.read(index // self.patches_per_image)
        points = foreground if len(foreground) and self.rng.random() < 0.65 else valid
        if self.sampling == "balanced" and class_points and self.rng.random() < 0.75:
            points = class_points[int(self.rng.choice(list(class_points)))]
        y, x = points[self.rng.integers(len(points))]
        dy, dx = self.rng.integers(self.tile // 4, self.tile - self.tile // 4, size=2)
        y = int(np.clip(y - dy, 0, mask.shape[0] - self.tile))
        x = int(np.clip(x - dx, 0, mask.shape[1] - self.tile))
        image = image[y:y + self.tile, x:x + self.tile]
        mask = mask[y:y + self.tile, x:x + self.tile]
        if self.rng.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        gain = self.rng.uniform(0.85, 1.15)
        offset = self.rng.uniform(-0.05, 0.05)
        image = normalize(np.clip((image - 0.5) * gain + 0.5 + offset, 0, 1))
        return (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            torch.from_numpy(mask.astype(np.int64)),
        )


def segmentation_loss(logits, target, weights, boundary_weight=0.0, dice_weight=0.5):
    logits = logits.float()
    valid = target != IGNORE_INDEX
    if not valid.any():
        return logits.sum() * 0
    pixel_loss = F.cross_entropy(logits, target, weight=weights, ignore_index=IGNORE_INDEX, reduction="none")
    emphasis = torch.ones_like(pixel_loss)
    if boundary_weight > 0:
        edges = torch.zeros_like(valid)
        horizontal = valid[:, :, 1:] & valid[:, :, :-1] & (target[:, :, 1:] != target[:, :, :-1])
        vertical = valid[:, 1:, :] & valid[:, :-1, :] & (target[:, 1:, :] != target[:, :-1, :])
        edges[:, :, 1:] |= horizontal
        edges[:, :, :-1] |= horizontal
        edges[:, 1:, :] |= vertical
        edges[:, :-1, :] |= vertical
        border = F.max_pool2d(edges[:, None].float(), 3, stride=1, padding=1)[:, 0]
        emphasis = emphasis + boundary_weight * border
    effective_weights = weights[target.clamp(0, logits.shape[1] - 1)] * valid * emphasis
    cross_entropy = (pixel_loss * emphasis).sum() / effective_weights.sum().clamp_min(1e-12)
    labels = torch.where(valid, target, 0)
    labels = F.one_hot(labels, logits.shape[1]).permute(0, 3, 1, 2).float() * valid[:, None]
    probabilities = logits.softmax(1) * valid[:, None]
    dims = (0, 2, 3)
    support = labels.sum(dims)
    dice = (2 * (probabilities * labels).sum(dims) + 1) / (probabilities.sum(dims) + support + 1)
    return cross_entropy + dice_weight * (1 - dice[support > 0].mean())


def evaluate(predictor, records, batch_size, overlap=None):
    n = len(predictor.classes)
    confusion = np.zeros((n, n), dtype=np.int64)
    for record in records:
        image, mask = _read_pair(record, predictor.classes)
        prediction, _ = predictor.predict(image, batch_size=batch_size, overlap=overlap)
        valid = mask != IGNORE_INDEX
        encoded = mask[valid].astype(np.int64) * n + prediction[valid]
        confusion += np.bincount(encoded, minlength=n * n).reshape(n, n)
    return {"sections": len(records), **confusion_metrics(confusion, predictor.classes)}


def train(args):
    validate_tiles(args.tile_size, args.tile_size // 4)
    if min(args.epochs, args.batch_size, args.patches_per_image) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Epoch, batch, số ô trên ảnh và learning rate cần lớn hơn 0.")
    if (
        args.patience < 0 or args.lr_patience < 0
        or not all(np.isfinite(value) and value >= 0 for value in (
            args.boundary_weight, args.dice_weight, args.weight_decay, args.grad_clip,
        ))
    ):
        raise ValueError("Patience, trọng số loss, weight decay và gradient clip cần không âm.")
    if not args.resume and any((args.out / name).exists() for name in ("best.pt", "last.pt", "metrics.json")):
        raise ValueError("Thư mục đã có thí nghiệm. Chọn --out khác hoặc --resume đường-dẫn/last.pt.")
    if args.resume and args.resume.resolve().parent != args.out.resolve() and any(
        (args.out / name).exists() for name in ("best.pt", "last.pt", "metrics.json")
    ):
        raise ValueError("Thư mục đích đã có thí nghiệm khác. Resume vào thư mục gốc hoặc một thư mục mới.")
    torch.manual_seed(args.seed)
    device = configure_device(args.device, args.cpu_threads)
    amp_enabled = args.amp == "auto" and device.type == "cuda"
    classes = load_classes(args.classes)
    train_records, val_records, test_records = split_records(
        load_records(args.manifest), args.val_fraction, args.seed,
    )
    partitions = {"train": train_records, "validation": val_records, "test": test_records}
    fingerprint = dataset_fingerprint(partitions, classes)
    print(
        f"{device} | {len(train_records)} ảnh train, {len(val_records)} ảnh validation, "
        f"{len(test_records)} ảnh test. Đang kiểm tra nhãn.",
        flush=True,
    )
    counts = _audit(train_records, classes)
    val_counts = _audit(val_records, classes)
    _audit(test_records, classes)
    if not counts[1:].sum() or not val_counts[1:].sum():
        raise ValueError("Cần pixel tướng (ID từ 1) trong cả tập train và validation.")
    active_ids = np.flatnonzero(counts).tolist()
    if len(active_ids) < 2:
        raise ValueError("Cần ít nhất hai lớp có nhãn trong tập train, có thể gồm tướng và nền.")
    for split_name, support in (("train", counts), ("validation", val_counts)):
        absent = ", ".join(label["name"] for label in classes[1:] if support[label["id"]] == 0)
        if absent:
            print(f"Lớp thiếu ví dụ trong {split_name}: {absent}", flush=True)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    weights /= weights[counts > 0].mean()
    weights = torch.tensor(weights.clip(0.25, 5), dtype=torch.float32, device=device)
    model = build_model(args.arch, len(classes), args.width).to(device)
    dataset = TrainingPatches(
        train_records, classes, args.tile_size, args.patches_per_image, args.seed, args.sampling,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=device.type == "cuda", generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    predictor = Predictor(model, classes, args.tile_size, device, active_ids=active_ids)
    checkpoint_path = args.out / "best.pt"
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    history = []
    best_score = -1.0
    best_metrics = None
    best_epoch = 0
    best_state, bad_epochs, start_epoch = None, 0, 1
    resumed = None
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resumed.get("resume_version") != 1:
            raise ValueError("Cần last.pt chứa trạng thái optimizer để tiếp tục huấn luyện.")
        if resumed.get("dataset_sha256") != fingerprint:
            raise ValueError("Dữ liệu hoặc cách chia nhóm đã thay đổi so với checkpoint.")
        immutable = (
            "arch", "width", "tile_size", "sampling", "boundary_weight", "dice_weight",
            "patches_per_image", "batch_size", "lr", "weight_decay", "seed", "grad_clip", "lr_patience", "amp",
        )
        changed = [key for key in immutable if resumed["config"].get(key) != config.get(key)]
        if changed:
            raise ValueError("Cấu hình khác checkpoint: " + ", ".join(changed))
        if resumed.get("amp_enabled", amp_enabled) != amp_enabled:
            raise ValueError("Chế độ AMP thực tế khác checkpoint. Dùng lại cùng loại thiết bị hoặc --amp off từ đầu.")
        if bool(resumed.get("demo")) != bool(args.demo):
            raise ValueError("Giữ nguyên --demo của lần chạy trước khi resume.")
        model.load_state_dict(resumed["state"])
        optimizer.load_state_dict(resumed["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(resumed["scheduler"])
        if amp_enabled and resumed["scaler"]:
            scaler.load_state_dict(resumed["scaler"])
        torch.set_rng_state(resumed["torch_rng"])
        if device.type == "cuda" and resumed.get("cuda_rng"):
            torch.cuda.set_rng_state_all(resumed["cuda_rng"])
        generator.set_state(resumed["loader_rng"])
        dataset.rng.bit_generator.state = resumed["data_rng"]
        best_score, best_epoch = resumed["best_score"], resumed["best_epoch"]
        best_state, best_metrics = resumed["best_state"], resumed["best_metrics"]
        history, bad_epochs = resumed["history"], resumed["bad_epochs"]
        start_epoch = resumed["epoch"] + 1
        if start_epoch > args.epochs:
            raise ValueError("--epochs là tổng số epoch, cần lớn hơn epoch đã lưu trong last.pt.")
    args.out.mkdir(parents=True, exist_ok=True)

    def inference_checkpoint(state):
        return {
            "format_version": 1, "classes": classes, "arch": args.arch, "width": args.width,
            "tile_size": args.tile_size, "state": state, "demo": args.demo, "active_ids": active_ids,
            "dataset_sha256": fingerprint,
        }

    def save_torch(path, payload):
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    parameter_count = sum(p.numel() for p in model.parameters())
    write_json(args.out / "config.json", {
        **config, "dataset_sha256": fingerprint, "parameter_count": parameter_count,
        "torch_version": str(torch.__version__), "amp_enabled": amp_enabled,
    })
    if best_state is not None:
        save_torch(checkpoint_path, inference_checkpoint(best_state))
    print(f"Kiến trúc: {args.arch} | {parameter_count:,} tham số | AMP: {amp_enabled}", flush=True)

    def report(test_metrics=None):
        return {
            "best_epoch": best_epoch, "completed_epochs": history[-1]["epoch"] if history else 0,
            "stopped_early": bool(args.patience and bad_epochs >= args.patience),
            "architecture": args.arch, "parameter_count": parameter_count, "config": config,
            "selection_metric": "mean_iou_facies_union",
            "dataset_sha256": fingerprint, "demo": args.demo, "seed": args.seed,
            "split_groups": {
                name: sorted({record.group for record in records})
                for name, records in partitions.items()
            },
            "train_support_pixels_by_id": counts.tolist(),
            "validation": best_metrics, "test": test_metrics, "history": history,
        }

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_total = 0.0
        learning_rate = optimizer.param_groups[0]["lr"]
        for image, mask in loader:
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with context:
                logits = model(image.to(device, non_blocking=True))
                loss = segmentation_loss(
                    logits, mask.to(device, non_blocking=True), weights, args.boundary_weight, args.dice_weight,
                )
            if not torch.isfinite(loss):
                raise ValueError("Loss không hữu hạn. Kiểm tra dữ liệu hoặc giảm learning rate.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip if args.grad_clip else float("inf"),
                error_if_nonfinite=not amp_enabled,
            )
            scaler.step(optimizer)
            scaler.update()
            loss_total += loss.item()
        model.eval()
        metrics = evaluate(predictor, val_records, args.batch_size)
        score, mean_loss = metrics["mean_iou_facies_union"], loss_total / len(loader)
        if not np.isfinite(mean_loss):
            raise ValueError("Loss không hữu hạn. Kiểm tra ảnh hoặc giảm learning rate.")
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | loss {mean_loss:.4f} | val mIoU {score:.4f}", flush=True)
        scheduler.step(score)
        history.append({
            "epoch": epoch, "loss": mean_loss, "val_miou_facies": metrics["mean_iou_facies"],
            "val_miou_union": score, "lr": learning_rate, "seconds": time.perf_counter() - epoch_start,
        })
        if score > best_score:
            best_score, best_metrics, best_epoch = score, metrics, epoch
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            save_torch(checkpoint_path, inference_checkpoint(best_state))
        else:
            bad_epochs += 1
        save_torch(args.out / "last.pt", {
            **inference_checkpoint(model.state_dict()),
            "resume_version": 1, "config": config, "epoch": epoch,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "loader_rng": generator.get_state(), "data_rng": dataset.rng.bit_generator.state,
            "history": history, "best_score": best_score, "best_epoch": best_epoch,
            "best_state": best_state, "best_metrics": best_metrics, "bad_epochs": bad_epochs,
            "amp_enabled": amp_enabled,
        })
        write_json(args.out / "metrics.json", report())
        if args.patience and bad_epochs >= args.patience:
            print(f"Dừng sớm: {bad_epochs} epoch không tăng chỉ số validation.", flush=True)
            break
    if test_records:
        best_predictor = load_predictor(checkpoint_path, device)
        test_metrics = evaluate(best_predictor, test_records, args.batch_size)
    else:
        test_metrics = None
    write_json(args.out / "metrics.json", report(test_metrics))
    print(f"Mô hình: {checkpoint_path.resolve()}", flush=True)
    print(f"Báo cáo: {(args.out / 'metrics.json').resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện U-Net phân vùng tướng địa chấn.")
    parser.add_argument("--config", type=Path, help="Cấu hình JSON, tham số dòng lệnh được ưu tiên.")
    parser.add_argument("--manifest", type=Path, metavar="DATASET.CSV")
    parser.add_argument("--classes", type=Path, default=Path("classes.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/facies"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--patches-per-image", type=int, default=12)
    parser.add_argument("--width", type=int, default=24, help="Số kênh cơ sở của U-Net.")
    parser.add_argument("--arch", choices=("unet", "attention_unet"), default="unet")
    parser.add_argument("--sampling", choices=("balanced", "foreground"), default="balanced")
    parser.add_argument("--boundary-weight", type=float, default=1.0, help="Tăng trọng số CE gần ranh giới.")
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="0 để tắt clip gradient.")
    parser.add_argument("--patience", type=int, default=10, help="Số epoch chờ tăng validation; 0 để tắt.")
    parser.add_argument("--lr-patience", type=int, default=3, help="Plateau trước khi giảm learning rate.")
    parser.add_argument("--amp", choices=("auto", "off"), default="auto", help="Mixed precision khi dùng CUDA.")
    parser.add_argument("--resume", type=Path, help="last.pt chứa optimizer và trạng thái lấy mẫu.")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--demo", action="store_true", help="Đánh dấu checkpoint huấn luyện từ dữ liệu mô phỏng.")
    try:
        preliminary, _ = parser.parse_known_args()
        if preliminary.config:
            defaults = json.loads(preliminary.config.read_text(encoding="utf-8-sig"))
            known = {action.dest for action in parser._actions} - {"help", "config"}
            if not isinstance(defaults, dict) or set(defaults) - known:
                raise ValueError("Cấu hình JSON có trường không hỗ trợ.")
            for action in parser._actions:
                if action.dest not in defaults:
                    continue
                value = defaults[action.dest]
                if action.type is Path:
                    valid = isinstance(value, str) or (value is None and action.default is None)
                elif action.type is int:
                    valid = type(value) is int
                elif action.type is float:
                    valid = type(value) in (int, float) and np.isfinite(value)
                elif isinstance(action, argparse._StoreTrueAction):
                    valid = type(value) is bool
                else:
                    valid = isinstance(value, str)
                if not valid or (action.choices and value not in action.choices):
                    raise ValueError(f"Giá trị JSON không hợp lệ: {action.dest}={value!r}")
            parser.set_defaults(**defaults)
        args = parser.parse_args()
        if args.manifest is None:
            raise ValueError("Cần --manifest hoặc manifest trong tệp cấu hình.")
        if args.arch not in ("unet", "attention_unet") or args.sampling not in ("balanced", "foreground"):
            raise ValueError("arch hoặc sampling trong JSON không hợp lệ.")
        if args.amp not in ("auto", "off"):
            raise ValueError("amp cần là auto hoặc off.")
        train(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"Lỗi: {error}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()

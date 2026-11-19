"""Generate paired numerical fixtures for a reproducible pipeline demonstration."""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _wave(phase):
    return np.sin(phase) + 0.34 * np.sin(2 * phase + 0.6)


def simulated_section(seed, height=352, width=576):
    rng = np.random.default_rng(seed)
    y, x = np.indices((height, width), dtype=np.float32)
    xn, yn = x / width, y / height
    image = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    center = 0.49 + rng.uniform(-0.06, 0.06) + 0.035 * np.sin(2 * np.pi * xn + rng.uniform(0, 6))
    top_edge = 11 + 4 * np.sin(xn * 7 + rng.uniform(0, 6))
    bottom_edge = height - 11 + 3 * np.cos(xn * 8 + rng.uniform(0, 6))
    active = (y > top_edge) & (y < bottom_edge) & (x >= 7) & (x < width - 7)
    perm = rng.permutation(np.arange(1, 7))
    for band in range(2):
        first = 0.33 + rng.uniform(-0.047, 0.047) + 0.035 * np.sin(yn * 17 + rng.uniform(0, 6))
        second = 0.67 + rng.uniform(-0.047, 0.047) + 0.025 * np.sin(yn * 19 + rng.uniform(0, 6))
        column = np.where(xn < first, 0, np.where(xn < second, 1, 2))
        for col in range(3):
            label = perm[band * 3 + col]
            area = active & (column == col) & ((yn >= center) == bool(band))
            xx = x - (col + 0.5) * width / 3
            yy = y - (band + 0.5) * height / 2
            phase = rng.uniform(0, 2 * np.pi)
            direction = rng.choice([-1, 1])
            gain = rng.uniform(0.78, 1)
            if label == 1:
                noise = rng.normal(size=image.shape).astype(np.float32)
                smooth = cv2.GaussianBlur(noise, (0, 0), sigmaX=3.1, sigmaY=1.2)
                texture = 1.75 * np.tanh(7.0 * smooth) + 0.18 * np.sin(0.54 * y + noise)
            elif label == 2:
                texture = _wave(rng.uniform(0.51, 0.56) * (yy + direction * 0.012 * xx) + phase)
            elif label == 3:
                waviness = 2.5 * np.sin(xx * 0.095 + yy * 0.035 + phase)
                texture = _wave(rng.uniform(0.95, 1.08) * (yy + waviness) + phase)
                texture *= 0.7 + 0.3 * np.cos(0.09 * xx + 0.12 * yy + phase)
            elif label == 4:
                dome = 36 * np.exp(-((xx - rng.uniform(-10, 10)) / 48) ** 2)
                texture = _wave(rng.uniform(0.30, 0.33) * (yy + dome) + phase)
            elif label == 5:
                fan = np.clip(1 + direction * 0.004 * xx, 0.60, 1.50)
                texture = _wave(rng.uniform(0.42, 0.47) * (yy / fan + direction * 0.42 * xx) + phase)
            else:
                slope = direction * (22 * np.tanh((xx + 0.16 * yy) / 38) + 0.16 * xx)
                texture = _wave(rng.uniform(0.71, 0.78) * (yy + slope) + phase)
            image[area] = (gain * texture)[area]
            mask[area] = label
    image = 0.5 + 0.27 * image + rng.normal(0, 0.032, image.shape)
    return np.round(255 * image.clip(0, 1)).astype(np.uint8), mask


def write_dataset(out):
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    with (out / "dataset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "mask", "group", "split"])
        for i in range(22):
            image, mask = simulated_section(20261118 + i)
            name = f"section_{i:02d}.png"
            Image.fromarray(image).save(out / "images" / name)
            Image.fromarray(mask).save(out / "masks" / name)
            split = "train" if i < 16 else "val" if i < 19 else "test"
            writer.writerow([f"images/{name}", f"masks/{name}", f"sim-{i:02d}", split])


def main():
    parser = argparse.ArgumentParser(description="Tạo 22 cặp ảnh và mask mô phỏng.")
    parser.add_argument("--out", type=Path, default=Path("runs/simulated"))
    args = parser.parse_args()
    write_dataset(args.out)
    print((args.out / "dataset.csv").resolve())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()

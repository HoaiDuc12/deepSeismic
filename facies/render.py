"""Xuất ảnh khoanh vùng và thống kê từ bản đồ tướng địa chấn."""

from __future__ import annotations

import json
import os
import base64
from io import BytesIO
import unicodedata
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont


def _font(size):
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for path in (
        windows / "segoeuib.ttf",
        windows / "arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size), True
        except OSError:
            continue
    return ImageFont.load_default(size=size), False


def _latin_name(name):
    text = name.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _region_anchor(component, left, top):
    # Extra zeros make the distance transform behave correctly at image edges.
    padded = np.pad(component.astype(np.uint8), 1)
    distances = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    _, radius, _, _ = cv2.minMaxLoc(distances)
    if radius < 1:
        # This often occurs along one-pixel layers. Center the label along the layer.
        rows, cols = np.nonzero(component)
        best = np.argmin((rows - rows.mean()) ** 2 + (cols - cols.mean()) ** 2)
        return [left + int(cols[best]), top + int(rows[best])]
    # Prefer a location near the component center if several points tie.
    rows, cols = np.nonzero(distances == radius)
    center_x, center_y = (component.shape[1] - 1) / 2, (component.shape[0] - 1) / 2
    best = np.argmin((rows - center_y) ** 2 + (cols - center_x) ** 2)
    return [left + int(cols[best]), top + int(rows[best])]


def _label_regions(overlay, regions):
    width, height = overlay.size
    size = max(11, min(44, round(max(width / 85, height / 60))))
    font, vietnamese = _font(size)
    canvas = ImageDraw.Draw(overlay, mode="RGBA")
    by_area = sorted(regions, key=lambda region: -region["pixel_count"])
    largest = {}
    for region in by_area:
        largest.setdefault(region["class_id"], region)
    priority = {region["region_id"] for region in largest.values()}
    to_label = list(largest.values()) + [r for r in by_area if r["region_id"] not in priority]
    used = []
    placements = []
    pad_x, pad_y, margin = max(5, size // 2), max(3, size // 4), 2
    for region in to_label:
        name = region["name"] if vietnamese else _latin_name(region["name"])
        if not name:
            name = str(region["class_id"])
        l, t, r, b = canvas.textbbox((0, 0), name, font=font)
        box_w, box_h = r - l + 2 * pad_x + 3, b - t + 2 * pad_y
        if box_w + 2 * margin > width or box_h + 2 * margin > height:
            continue
        ax, ay = region["label_xy"]
        offsets = (
            (0, 0), (0, -box_h), (0, box_h),
            (-box_w, 0), (box_w, 0),
            (-box_w // 2, -box_h), (box_w // 2, -box_h),
            (-box_w // 2, box_h), (box_w // 2, box_h),
            (0, -2 * box_h), (0, 2 * box_h),
        )
        for dx, dy in offsets:
            x = min(max(ax - box_w // 2 + dx, margin), width - box_w - margin)
            y = min(max(ay - box_h // 2 + dy, margin), height - box_h - margin)
            box = (x, y, x + box_w, y + box_h)
            if any(box[0] < bx2 + 3 and box[2] > bx1 - 3 and
                   box[1] < by2 + 3 and box[3] > by1 - 3 for bx1, by1, bx2, by2 in used):
                continue
            used.append(box)
            placements.append((region, name, box, x + pad_x + 3 - l, y + pad_y - t))
            region["label_bbox_xyxy"] = list(box)
            break

    for region, _, (x, y, x2, y2), _, _ in placements:
        ax, ay = region["label_xy"]
        closest = (min(max(ax, x), x2 - 1), min(max(ay, y), y2 - 1))
        if (ax, ay) != closest:
            color = ImageColor.getrgb(region["color"])
            canvas.line(((ax, ay), closest), fill=(20, 25, 34, 210), width=3)
            canvas.line(((ax, ay), closest), fill=(*color, 255), width=1)

    for region, name, (x, y, x2, y2), tx, ty in placements:
        color = ImageColor.getrgb(region["color"])
        canvas.rounded_rectangle((x, y, x2 - 1, y2 - 1), radius=3, fill=(14, 20, 30, 235))
        canvas.rounded_rectangle((x + 2, y + 2, x + 5, y2 - 3), radius=1, fill=(*color, 255))
        canvas.text((tx, ty), name, font=font, fill=(255, 255, 255, 255))


def render_overlay(image, mask, confidence, classes, *, min_region_area=64, alpha=0.27):
    """Return the overlay image and metadata. Pixel boxes use exclusive right/bottom edges.

    Area fractions are relative to the whole section. The region cutoff affects the
    overlay and region list. Class pixel counts and mean scores use the full ID mask.
    """
    image, mask, confidence = map(np.asarray, (image, mask, confidence))
    if (mask.ndim != 2 or not mask.size or image.shape != (*mask.shape, 3)
            or confidence.shape != mask.shape):
        raise ValueError("Ảnh RGB, mask và bản đồ độ tin cậy phải cùng chiều rộng và chiều cao.")
    valid_ids = {int(item["id"]) for item in classes} | {255}
    if mask.dtype.kind not in "iu" or not set(np.unique(mask)).issubset(valid_ids):
        raise ValueError("Mask phải chứa các mã lớp nguyên đã khai báo hoặc 255.")
    if min_region_area < 1 or not 0 <= alpha <= 1:
        raise ValueError("Diện tích hiển thị tối thiểu phải ≥1 và độ phủ màu thuộc [0, 1].")
    if not np.isfinite(confidence).all() or confidence.min() < 0 or confidence.max() > 1:
        raise ValueError("Độ tin cậy phải thuộc [0, 1].")
    if image.dtype == np.uint8:
        pixels = image.copy()
    else:
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Ảnh RGB phải là uint8 hoặc có giá trị trong [0, 1].")
        pixels = np.rint(image * 255).astype(np.uint8)

    height, width = mask.shape
    outline = max(1, min(4, round(max(width, height) / 1000)))
    kernel = np.ones((3, 3), dtype=np.uint8)
    levels = np.arange(256, dtype=np.float32)[:, None]
    channels = np.arange(3)
    regions, class_summary = [], []
    for item in classes:
        class_id = int(item["id"])
        if class_id in (0, 255):
            continue
        name = str(item["name"])
        color = item["color"]
        rgb = ImageColor.getrgb(color) if isinstance(color, str) else tuple(map(int, color))
        color_hex = "#{:02X}{:02X}{:02X}".format(*rgb)
        selection = mask == class_id
        pixel_count = int(np.count_nonzero(selection))
        retained_count = 0
        if pixel_count:
            _, components, stats, _ = cv2.connectedComponentsWithStats(
                selection.astype(np.uint8), connectivity=8
            )
            keep = stats[:, cv2.CC_STAT_AREA] >= min_region_area
            keep[0] = False
            retained = np.flatnonzero(keep)
            retained_count = len(retained)
            if retained_count:
                display = keep[components]
                blend = np.rint((1 - alpha) * levels + alpha * np.array(rgb)).astype(np.uint8)
                pixels[display] = blend[pixels[display], channels]
                eroded = cv2.erode(display.astype(np.uint8), kernel, iterations=outline,
                                   borderType=cv2.BORDER_CONSTANT, borderValue=0)
                pixels[display & (eroded == 0)] = rgb
            for index in retained:
                x, y, w, h, area = map(int, stats[index])
                box = np.s_[y:y + h, x:x + w]
                region = components[box] == index
                regions.append({
                    "region_id": len(regions) + 1,
                    "class_id": class_id,
                    "name": name,
                    "color": color_hex,
                    "pixel_count": area,
                    "bbox_xyxy": [x, y, x + w, y + h],
                    "label_xy": _region_anchor(region, x, y),
                    "label_bbox_xyxy": None,
                    "mean_confidence": float(np.mean(confidence[box][region])),
                })
        class_summary.append({
            "class_id": class_id,
            "name": name,
            "color": color_hex,
            "pixel_count": pixel_count,
            "area_fraction": pixel_count / mask.size,
            "region_count": retained_count,
            "mean_confidence": float(np.mean(confidence[selection])) if pixel_count else None,
        })
    overlay = Image.fromarray(pixels)
    if regions:
        _label_regions(overlay, regions)
    return overlay, {
        "width": width,
        "height": height,
        "min_region_area": min_region_area,
        "region_count": len(regions),
        "class_summary": class_summary,
        "regions": regions,
    }


def _checked_map(values, shape, name, *, boolean=False):
    values = np.asarray(values)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{name} cần cùng kích thước mask và chứa giá trị hữu hạn.")
    if boolean:
        if not np.isin(values, (0, 1)).all():
            raise ValueError(f"{name} cần là bản đồ boolean hoặc 0/1.")
        return values.astype(bool)
    if values.min() < 0 or values.max() > 1:
        raise ValueError(f"{name} cần nằm trong [0, 1].")
    return values


def labelme_prediction(mask, confidence, classes, *, min_region_area=64):
    """Export supported polygons without flattening holes into a solid region.

    LabelMe has no standard subtractive-polygon semantics. Components with holes,
    degenerate contours, or any failed exact rasterization check are omitted and
    explicitly reported. The pixel mask remains the authoritative prediction.
    """
    height, width = mask.shape
    shapes, omissions = [], []
    region_number = 0
    for item in classes:
        if item["id"] in (0, 255):
            continue
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask == item["id"]).astype(np.uint8), connectivity=8,
        )
        for index in range(1, len(stats)):
            x, y, w, h, area = map(int, stats[index])
            region_number += 1
            omitted = {"region_id": region_number, "class_id": item["id"],
                       "name": item["name"], "pixel_count": area, "bbox_xyxy": [x, y, x + w, y + h]}
            if area < min_region_area:
                omissions.append({**omitted, "reason": "below_min_region_area"})
                continue
            component = (labels[y:y + h, x:x + w] == index).astype(np.uint8)
            contours, hierarchy = cv2.findContours(component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None or len(contours) != 1 or hierarchy[0, 0, 2] != -1:
                omissions.append({**omitted, "reason": "holes_or_multiple_contours_not_representable"})
                continue
            points = contours[0][:, 0, :]
            if len(points) < 3 or cv2.contourArea(points) == 0:
                omissions.append({**omitted, "reason": "degenerate_contour"})
                continue
            # PIL is also used by the loader; prove the polygon round-trips exactly.
            raster = Image.new("L", (w, h))
            ImageDraw.Draw(raster).polygon([tuple(map(int, point)) for point in points], fill=1)
            if not np.array_equal(np.asarray(raster), component):
                omissions.append({**omitted, "reason": "polygon_does_not_reproduce_pixel_region"})
                continue
            score = float(confidence[y:y + h, x:x + w][component.astype(bool)].mean())
            shapes.append({
                "label": item["name"], "class_id": item["id"],
                "points": (points + np.array([x, y])).tolist(),
                "group_id": region_number, "shape_type": "polygon",
                "flags": {"machine_prediction": True, "reviewed": False},
                "description": f"Model probability {score:.4f}; uncalibrated; requires expert review.",
                "mean_confidence": score,
            })
    return {
        "version": "5.5.0", "flags": {"machine_prediction": True}, "shapes": shapes,
        "imagePath": "source.png", "imageData": None, "imageHeight": height, "imageWidth": width,
        "_facies": {
            "requires_expert_review": True,
            "confidence_is_calibrated": False,
            "unassigned_pixels": "Remain unannotated (ID 255 on re-import); never infer background.",
            "omitted_regions": omissions,
        },
    }


def _trace_page(source, annotation, classes):
    buffer = BytesIO()
    source.save(buffer, format="PNG")
    payload = {"annotation": annotation, "classes": classes,
               "source": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")}
    safe = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("&", "\\u0026")
    return _TRACE_HTML.replace("__PAYLOAD__", safe)


def export_results(out_dir, image, mask, confidence, classes, *, min_region_area=64, alpha=0.27,
                   raw_mask=None, entropy=None, review_mask=None, provenance=None,
                   export_labelme=True):
    """Save native-size masks, score maps and reviewable machine annotations.

    The supplied thresholded mask is saved verbatim, including ID 255. Neither it
    nor raw_mask is modified by visualization, filtering or polygon extraction.
    Confidence/entropy PNGs are 16-bit values scaled by 65535 (not RGB heatmaps).
    """
    overlay, metadata = render_overlay(
        image, mask, confidence, classes, min_region_area=min_region_area, alpha=alpha
    )
    mask, confidence = np.asarray(mask), np.asarray(confidence)
    raw = mask if raw_mask is None else np.asarray(raw_mask)
    valid_ids = {int(item["id"]) for item in classes} | {255}
    if raw.shape != mask.shape or raw.dtype.kind not in "iu" or not set(np.unique(raw)) <= valid_ids:
        raise ValueError("raw_mask phải cùng kích thước và chứa ID lớp hoặc 255.")
    review = mask == 255 if review_mask is None else _checked_map(
        review_mask, mask.shape, "Bản đồ xem lại", boolean=True,
    )
    if np.any((raw != mask) & ((mask != 255) | ~review)):
        raise ValueError("Mask chỉ được khác raw_mask tại pixel 255 có đánh dấu xem lại.")
    if np.any((mask == 255) & ~review):
        raise ValueError("Pixel 255 cần được đánh dấu trong bản đồ xem lại.")
    entropy = None if entropy is None else _checked_map(entropy, mask.shape, "Entropy")
    metadata.update({
        "mean_confidence": float(confidence.mean()),
        "mean_entropy": None if entropy is None else float(entropy.mean()),
        "review_pixel_count": int(review.sum()), "review_fraction": float(review.mean()),
        "accepted_fraction": float((mask != 255).mean()),
        "confidence_is_calibrated": False,
        "score_png_scale": 65535,
        "provenance": {} if provenance is None else provenance,
    })
    for item in metadata["class_summary"]:
        selected = raw == item["class_id"]
        count = int(selected.sum())
        item.update({
            "raw_pixel_count": count,
            "raw_area_fraction": count / mask.size,
            "review_pixel_count": int((selected & review).sum()),
            "review_fraction": float(review[selected].mean()) if count else None,
            "raw_mean_confidence": float(confidence[selected].mean()) if count else None,
            "raw_mean_entropy": float(entropy[selected].mean()) if count and entropy is not None else None,
        })
    # Per-region scores can be computed from existing bboxes without full label maps.
    if entropy is not None:
        for region in metadata["regions"]:
            x1, y1, x2, y2 = region["bbox_xyxy"]
            box = np.s_[y1:y2, x1:x2]
            # Distinguish adjacent components that happen to share a bounding box.
            local = (mask[box] == region["class_id"]).astype(np.uint8)
            _, labels = cv2.connectedComponents(local, connectivity=8)
            ax, ay = region["label_xy"]
            component = labels == labels[ay - y1, ax - x1]
            region["mean_entropy"] = float(entropy[box][component].mean())

    values = np.asarray(image)
    source = Image.fromarray(values if values.dtype == np.uint8 else np.rint(values * 255).astype(np.uint8))
    annotation = labelme_prediction(mask, confidence, classes, min_region_area=min_region_area) if export_labelme else None
    if annotation is not None:
        annotation["_facies"]["provenance"] = metadata["provenance"]
        metadata["labelme"] = {"exported_polygon_count": len(annotation["shapes"]),
                               "omitted_regions": annotation["_facies"]["omitted_regions"]}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "overlay": out_dir / "overlay.png",
        "mask": out_dir / "mask.png",
        "regions": out_dir / "regions.json",
        "confidence": out_dir / "confidence.png",
        "review": out_dir / "review.png",
        "source": out_dir / "source.png",
    }
    if raw_mask is not None:
        paths["raw_mask"] = out_dir / "raw_mask.png"
    if entropy is not None:
        paths["uncertainty"] = out_dir / "uncertainty.png"
    if annotation is not None:
        paths.update(labelme=out_dir / "labelme.json", trace=out_dir / "trace.html")
    # Encode all files before replacing any existing result. Each replacement is atomic.
    with TemporaryDirectory(prefix=".render-", dir=out_dir) as work_dir:
        staged = {key: Path(work_dir) / path.name for key, path in paths.items()}
        overlay.save(staged["overlay"])
        Image.fromarray(np.asarray(mask, dtype=np.uint8)).save(staged["mask"])
        Image.fromarray(np.rint(confidence * 65535).astype(np.uint16)).save(staged["confidence"])
        Image.fromarray(review.astype(np.uint8) * 255).save(staged["review"])
        source.save(staged["source"])
        if raw_mask is not None:
            Image.fromarray(raw.astype(np.uint8)).save(staged["raw_mask"])
        if entropy is not None:
            Image.fromarray(np.rint(entropy * 65535).astype(np.uint16)).save(staged["uncertainty"])
        if annotation is not None:
            staged["labelme"].write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            staged["trace"].write_text(_trace_page(source, annotation, classes), encoding="utf-8")
        staged["regions"].write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for key, path in paths.items():
            staged[key].replace(path)
    return paths


_TRACE_HTML = r'''<!doctype html>
<html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Duyệt và hiệu chỉnh tướng địa chấn</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#182a3d;font:15px/1.5 system-ui,sans-serif}
header,main{max-width:1500px;margin:auto;padding:18px 22px}header{background:#142d43;color:white;max-width:none}
h1{margin:0;font-size:23px}p{margin:6px 0}.tools{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:15px 0}
button,select{font:inherit;border:1px solid #b5c4d0;border-radius:5px;padding:8px 12px;background:white;color:#17364d}
button{cursor:pointer}button.primary{background:#087e8b;color:white;border-color:#087e8b}button:disabled{opacity:.5;cursor:default}
#wrap{background:#131f2a;overflow:auto;max-height:76vh;border-radius:6px;position:relative}
canvas{display:block;width:100%;height:auto;touch-action:none;cursor:crosshair}
.notice{background:#fff5db;border-left:4px solid #db9f2e;padding:9px 12px;border-radius:3px}
#status{min-height:24px;font-weight:600}small{color:#566c80}details{margin:12px 0}#omissions{max-height:180px;overflow:auto}
</style>
<header><h1>Duyệt và hiệu chỉnh tướng địa chấn</h1><p>Dự đoán máy cần chuyên gia kiểm tra. Điểm xác suất chưa được hiệu chuẩn.</p></header>
<main><div class="notice" id="notice"></div>
<p>Chọn một vùng rồi kéo các đỉnh để sửa. Nhấn “Vẽ vùng mới”, bấm các đỉnh và Enter để hoàn tất; Esc để hủy.
Phần chưa vẽ vẫn chưa gán nhãn (255), không tự đổi thành nền.</p>
<div class="tools"><label>Tướng <select id="class"></select></label>
<button id="new">Vẽ vùng mới</button><button id="finish">Hoàn tất vùng</button><button id="cancel">Hủy nét vẽ</button>
<button id="undo">Hoàn tác</button><button id="delete">Xóa vùng chọn</button><button id="reviewed">Đánh dấu đã duyệt</button></div>
<div class="tools"><label>Vùng <select id="regions"></select></label><label><input type="checkbox" id="visible" checked> Hiện polygon</label>
<button class="primary" id="save">Lưu JSON hiệu chỉnh</button><button id="source">Lưu source.png</button></div>
<div id="status" role="status" aria-live="polite"></div><div id="wrap"><canvas id="canvas"></canvas></div>
<details><summary>Các vùng chưa xuất thành polygon</summary><div id="omissions"></div></details>
<small>JSON theo định dạng LabelMe; lưu cùng source.png. Vùng có lỗ hoặc không thể raster hóa chính xác bị bỏ qua và được liệt kê rõ.
Mask PNG vẫn giữ toàn bộ dự đoán gốc; hiệu chỉnh ở đây không tự cập nhật mask đã xuất.</small>
</main><script type="application/json" id="payload">__PAYLOAD__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('payload').textContent), doc=data.annotation;
const $=id=>document.getElementById(id), canvas=$('canvas'), ctx=canvas.getContext('2d'), pic=new Image();
let chosen=-1,draft=null,drag=-1,saved=[],visible=true;
const color=label=>{let v=data.classes.find(c=>c.name===label)?.color||[255,180,50];return Array.isArray(v)?`rgb(${v.join(',')})`:v};
for(const c of data.classes){let o=document.createElement('option');o.value=c.name;o.textContent=c.name;$('class').append(o)}
const omissions=doc._facies.omitted_regions;
$('notice').textContent=`${doc.shapes.length} polygon máy; ${omissions.length} vùng chưa xuất (lỗ, quá nhỏ hoặc không biểu diễn chính xác). Chuyên gia cần vẽ bổ sung các vùng này.`;
for(const r of omissions){let p=document.createElement('div');p.textContent=`${r.name}: ${r.pixel_count} pixel — ${r.reason}; hộp [${r.bbox_xyxy.join(', ')}]`;$('omissions').append(p)}
function remember(){saved.push(JSON.stringify(doc.shapes));if(saved.length>60)saved.shift()}
function message(text){$('status').textContent=text}
function refresh(){let list=$('regions');list.replaceChildren();let empty=new Option('— Chưa chọn —','-1');list.add(empty);
doc.shapes.forEach((s,i)=>list.add(new Option(`${i+1}. ${s.label}${s.flags?.reviewed?' ✓':''}`,String(i))));
list.value=String(chosen);if(chosen>=0)$('class').value=doc.shapes[chosen].label;
$('finish').disabled=!draft||draft.length<3;$('cancel').disabled=draft===null;
for(const id of ['delete','reviewed'])$(id).disabled=chosen<0;
$('undo').disabled=!saved.length&&!(draft&&draft.length);draw()}
function polygon(points,close){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));if(close)ctx.closePath()}
function draw(){if(!pic.complete||!pic.naturalWidth)return;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(pic,0,0);
let scale=canvas.width/Math.max(canvas.clientWidth,1), lw=1.5*scale;
if(visible)doc.shapes.forEach((s,i)=>{polygon(s.points,true);ctx.fillStyle=color(s.label);ctx.globalAlpha=i===chosen?0.23:0.1;ctx.fill();ctx.globalAlpha=1;
ctx.strokeStyle=color(s.label);ctx.lineWidth=i===chosen?3*scale:lw;ctx.stroke();
if(i===chosen)for(const [x,y] of s.points){ctx.beginPath();ctx.arc(x,y,4*scale,0,Math.PI*2);ctx.fillStyle='white';ctx.fill();ctx.strokeStyle='#142d43';ctx.lineWidth=scale;ctx.stroke()}});
if(draft){polygon(draft,false);ctx.strokeStyle=color($('class').value);ctx.lineWidth=2*scale;ctx.stroke();for(const [x,y] of draft){ctx.fillStyle='white';ctx.fillRect(x-3*scale,y-3*scale,6*scale,6*scale)}}}
function point(e){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(canvas.width-1,(e.clientX-r.left)*canvas.width/r.width)),Math.max(0,Math.min(canvas.height-1,(e.clientY-r.top)*canvas.height/r.height))]}
function mark(shape){shape.flags={...shape.flags,reviewed:true,expert_edited:true};
if(shape.mean_confidence!==undefined){shape.model_mean_confidence_before_edit=shape.mean_confidence;delete shape.mean_confidence}
shape.description='Expert-edited geometry or class; prior model score does not apply to corrected pixels'}
canvas.addEventListener('pointerdown',e=>{const p=point(e);if(draft!==null){draft.push(p);refresh();return}if(!visible)return;
const tolerance=9*canvas.width/Math.max(canvas.clientWidth,1);
if(chosen>=0){const points=doc.shapes[chosen].points;const idx=points.findIndex(q=>Math.hypot(q[0]-p[0],q[1]-p[1])<=tolerance);
if(idx>=0){remember();drag=idx;canvas.setPointerCapture(e.pointerId);return}}
chosen=-1;for(let i=doc.shapes.length-1;i>=0;i--){polygon(doc.shapes[i].points,true);if(ctx.isPointInPath(...p)){chosen=i;break}}refresh()});
canvas.addEventListener('pointermove',e=>{if(drag<0||chosen<0)return;const s=doc.shapes[chosen];s.points[drag]=point(e);mark(s);draw()});
canvas.addEventListener('pointerup',()=>{if(drag>=0){drag=-1;refresh();message('Đã sửa đỉnh. Lưu JSON khi hoàn tất.')}});
canvas.addEventListener('pointercancel',()=>{drag=-1;refresh()});
$('new').onclick=()=>{chosen=-1;draft=[];message('Bấm tối thiểu 3 đỉnh. Enter để hoàn tất.');refresh()};
function finish(){if(!draft||draft.length<3)return;remember();doc.shapes.push({label:$('class').value,points:draft,group_id:null,shape_type:'polygon',flags:{machine_prediction:false,reviewed:true,expert_edited:true},description:'Expert-created polygon'});chosen=doc.shapes.length-1;draft=null;message('Đã thêm vùng.');refresh()}
$('finish').onclick=finish;
function cancel(){draft=null;message('Đã hủy nét đang vẽ.');refresh()}
$('cancel').onclick=cancel;
$('delete').onclick=()=>{if(chosen<0)return;remember();doc.shapes.splice(chosen,1);chosen=-1;refresh();message('Đã xóa vùng; có thể hoàn tác.')};
$('reviewed').onclick=()=>{if(chosen<0)return;remember();doc.shapes[chosen].flags={...doc.shapes[chosen].flags,reviewed:true};refresh();message('Đã đánh dấu vùng này được chuyên gia duyệt.')};
function undo(){if(draft?.length){draft.pop();refresh();return}if(!saved.length)return;doc.shapes=JSON.parse(saved.pop());chosen=-1;refresh();message('Đã hoàn tác.')}
$('undo').onclick=undo;
$('class').onchange=()=>{if(chosen>=0){remember();doc.shapes[chosen].label=$('class').value;delete doc.shapes[chosen].class_id;mark(doc.shapes[chosen]);refresh()}else draw()};
$('regions').onchange=()=>{draft=null;chosen=Number($('regions').value);refresh()};
$('visible').onchange=()=>{visible=$('visible').checked;draw()};
function download(blob,name){let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),3000)}
$('save').onclick=()=>{if(draft!==null){message('Hoàn tất hoặc hủy nét vẽ trước khi lưu.');return}doc._facies.opened_in_trace=true;
download(new Blob([JSON.stringify(doc,null,2)],{type:'application/json;charset=utf-8'}),'labelme_corrected.json');message('Đã xuất JSON. Lưu thêm source.png cùng thư mục để mở trong LabelMe.')};
$('source').onclick=()=>{const bytes=Uint8Array.from(atob(data.source.split(',')[1]),c=>c.charCodeAt(0));download(new Blob([bytes],{type:'image/png'}),'source.png')};
document.addEventListener('keydown',e=>{if(e.target.matches('select,input,textarea'))return;if(e.key==='Enter'){e.preventDefault();finish()}if(e.key==='Escape')cancel();if((e.ctrlKey||e.metaKey)&&e.key==='z'){e.preventDefault();undo()}});
pic.onload=()=>{canvas.width=pic.naturalWidth;canvas.height=pic.naturalHeight;refresh();message('Chọn vùng để xem và hiệu chỉnh.')};pic.src=data.source;
window.addEventListener('resize',draw);
</script></html>'''

"""Giao diện cục bộ: suy luận, rà soát, kiểm tra dữ liệu và thí nghiệm."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime
from io import BytesIO
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "runs" / "facies" / "best.pt"
DEMO_MODEL = ROOT / "demo" / "model.pt"
DEMO_IMAGE = ROOT / "demo" / "sample.png"
TRAIN_COMMAND = "python -m facies.train --manifest data/dataset.csv --classes classes.json --out runs/facies --arch attention_unet"
DEMO_MODE = "Mô hình mô phỏng"
EXPERT_MODE = "Mô hình đã huấn luyện"
LOG = logging.getLogger(__name__)

HEADER = """
<style>
.stMainBlockContainer { max-width:1510px; padding-top:2.6rem; }
.facies-header {
 color:#f4fbfc; background:radial-gradient(ellipse at 90% 100%,#15665d,transparent 54%),#102c36;
 border-radius:16px; border-bottom:4px solid #47b9a2; padding:clamp(1.35rem,4vw,2.6rem); margin-bottom:.45rem;
}
.facies-header h1 {
 color:inherit; font-size:clamp(1.75rem,3.3vw,3rem); letter-spacing:-.035em; line-height:1.2; margin:0 0 .45rem; padding:0;
}
.facies-header p { color:#c4e3e1; line-height:1.5; margin:0; }
[data-testid="stBaseButton-primary"] { background:#08796e; border-color:#08796e; }
[data-testid="stBaseButton-primary"]:hover:not(:disabled) { background:#086559; border-color:#086559; }
[data-testid="stImage"] img { border-radius:6px; }
</style>
<header class="facies-header">
<h1>Phân tướng địa chấn</h1><p>Khoanh vùng, rà soát nhãn và theo dõi thí nghiệm trên ảnh lát cắt.</p>
</header>
"""


@st.cache_resource(max_entries=2, show_spinner=False)
def cached_predictor(path: str, version: tuple[int, int], device: str):
    from facies.predict import default_device, load_predictor
    return load_predictor(path, device=default_device() if device == "Tự động" else "cpu")


def read_checkpoint_path(value: str) -> Path | None:
    value = value.strip().strip('"')
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def file_version(path: Path | None):
    try:
        if path is None or not path.is_file():
            return None
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def show_training_help() -> None:
    st.info("Chưa tìm thấy trọng số mô hình. Cần ảnh sạch cùng nhãn vùng đã được chuyên gia kiểm tra để huấn luyện.")
    st.caption("Nếu tên tướng nằm trực tiếp trên ảnh, xem cách nhập viền màu hoặc vẽ lại vùng trong README.md và trace.html.")
    st.code(TRAIN_COMMAND, language="powershell")
    st.caption("Sau khi huấn luyện, chọn tệp runs/facies/best.pt trong thanh bên.")


def sidebar_settings() -> dict:
    with st.sidebar:
        st.header("Thiết lập")
        mode = EXPERT_MODE
        if DEMO_MODEL.is_file() and DEMO_IMAGE.is_file():
            mode = st.radio("Nguồn mô hình", (EXPERT_MODE, DEMO_MODE), index=0 if DEFAULT_MODEL.is_file() else 1, key="model_source")
        checkpoint = DEMO_MODEL
        if mode == EXPERT_MODE:
            value = st.text_input("Tệp trọng số (.pt)", value="runs/facies/best.pt", key="model_path", help="Đường dẫn trên máy chạy ứng dụng. Chọn trọng số do dự án này tạo.")
            try:
                checkpoint = read_checkpoint_path(value)
            except (OSError, RuntimeError, ValueError):
                checkpoint = None
                st.error("Đường dẫn tệp trọng số không hợp lệ.")
            if file_version(checkpoint):
                st.caption("Đã tìm thấy tệp trọng số.")
        else:
            st.caption("Trọng số minh họa được huấn luyện bằng ảnh mô phỏng.")
            attention_demo = DEMO_MODEL.parent / "attention" / "model.pt"
            if attention_demo.is_file():
                variant = st.selectbox("Trọng số demo", ["U-Net · mẫu gốc", "Attention U-Net · mô phỏng"], key="demo_variant")
                if variant == "Attention U-Net · mô phỏng":
                    checkpoint = attention_demo
        st.subheader("Rà soát dự đoán")
        threshold = st.slider(
            "Ngưỡng yêu cầu rà soát", 0.0, 1.0, 0.55, 0.05, key="review_threshold",
            help="Pixel có xác suất cao nhất thấp hơn ngưỡng được ghi mã 255 trong mask đã lọc. Đặt 0 để giữ toàn bộ dự đoán.",
        )
        st.caption("Xác suất và độ bất định chưa được hiệu chuẩn thành độ chính xác địa chất.")
        min_area = st.number_input("Diện tích vùng hiển thị tối thiểu (pixel)", min_value=1, value=64, step=1, key="min_area", help="Chỉ lọc vùng hiển thị và đa giác xuất; lưu mask pixel riêng.")
        with st.expander("Tùy chọn nâng cao"):
            tta = st.checkbox("TTA: đối chiếu ảnh lật ngang", value=False, key="use_tta", help="Gộp ảnh gốc và ảnh lật ngang; tăng thời gian tính. Cần đánh giá lợi ích trên dữ liệu thực.")
            tile = st.selectbox("Cạnh ô xử lý (pixel)", [None, 128, 192, 256, 384, 512, 768], key="tile_size", format_func=lambda x: "Theo mô hình" if x is None else str(x))
            alpha = st.slider("Độ đậm lớp màu", 0.0, 0.65, 0.27, 0.01, key="overlay_alpha")
            device = st.selectbox("Thiết bị xử lý", ["Tự động", "CPU"], key="device")
    return dict(mode=mode, checkpoint=checkpoint, version=file_version(checkpoint), device=device, tile_size=tile, min_region_area=min_area, alpha=alpha, tta=tta, confidence_threshold=threshold)


def inference_signature(settings: dict) -> tuple:
    return tuple(str(settings[k]) for k in ("checkpoint", "version", "device", "tile_size", "min_region_area", "alpha", "tta", "confidence_threshold"))


def predict_section(image, checkpoint: Path, version, device, tile_size, min_region_area, alpha, *, tta=False, confidence_threshold=0.0, image_name="section.png"):
    from facies.predict import prediction_provenance
    from facies.render import export_results

    start = perf_counter()
    predictor = cached_predictor(str(checkpoint), version, device)
    details = predictor.predict_detailed(image, tile_size=tile_size, batch_size=2, tta=tta, confidence_threshold=confidence_threshold)
    provenance = prediction_provenance(checkpoint, predictor, tta=tta, confidence_threshold=confidence_threshold, tile_size=tile_size)
    provenance["source_image"] = image_name
    with TemporaryDirectory(prefix="facies_results_") as directory:
        paths = export_results(
            directory, image, details["mask"], details["confidence"], predictor.classes,
            min_region_area=min_region_area, alpha=alpha, raw_mask=details["raw_mask"],
            entropy=details["entropy"], review_mask=details["review_mask"], provenance=provenance,
        )
        files = {key: Path(path).read_bytes() for key, path in paths.items()}
    return {
        "files": files, "summary": json.loads(files["regions"]), "elapsed": perf_counter() - start,
        "demo": bool(predictor.demo), "review_fraction": float(details["review_mask"].mean()),
        "mean_confidence": float(details["confidence"].mean()), "mean_entropy": float(details["entropy"].mean()),
        "threshold": confidence_threshold,
    }


def map_preview(data: bytes):
    """Display numeric 16-bit maps without clipping or changing exported data."""
    import numpy as np
    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        values = np.asarray(image)
        if image.mode in {"I;16", "I;16L", "I;16B", "I"}:
            return np.rint(np.clip(values.astype(np.float32) / 65535, 0, 1) * 255).astype(np.uint8)
        return values.copy()


def render_output(image, result, image_name: str, min_region_area: int) -> None:
    summary, files = result["summary"], result["files"]
    classes = [x for x in summary["class_summary"] if x["pixel_count"] > 0]
    if result["demo"]:
        st.warning("Kết quả mô phỏng. Các điểm số từ dữ liệu mô phỏng không thể hiện chất lượng trên lát cắt thực.")
    cols = st.columns(4)
    cols[0].metric("Tướng dự đoán", len(classes), border=True)
    cols[1].metric("Vùng được khoanh", summary["region_count"], border=True)
    cols[2].metric("Pixel cần rà soát", f"{100 * result.get('review_fraction', 0):.1f}%", border=True)
    cols[3].metric("Thời gian xử lý", f"{result['elapsed']:.1f} giây", border=True)
    if result.get("review_fraction", 0) > 0:
        st.caption("Pixel dưới ngưỡng được bỏ trống trong mask đã lọc (mã 255); mask gốc vẫn có toàn bộ dự đoán.")
    compare, uncertainty = st.tabs(["Ảnh và vùng tướng", "Độ tin cậy và rà soát"])
    with compare:
        a, b = st.columns(2)
        a.image(image, caption="Lát cắt đầu vào", width="stretch")
        b.image(files["overlay"], caption="Vùng tướng dự đoán", width="stretch")
        with st.expander("Xem ảnh khoanh vùng toàn chiều rộng"):
            st.image(files["overlay"], width="stretch")
    with uncertainty:
        st.caption("Đây là tín hiệu từ mô hình để ưu tiên kiểm tra; không thay thế đánh giá địa chất của chuyên gia. Bản đồ xám: đen gần 0, trắng gần 1; bản đồ rà soát: trắng là pixel dưới ngưỡng.")
        for panel, key, label in zip(st.columns(3), ("confidence", "uncertainty", "review"), ("Xác suất cao nhất theo pixel", "Độ bất định (entropy chuẩn hóa)", "Pixel cần chuyên gia rà soát")):
            if key in files:
                panel.image(map_preview(files[key]), caption=label, width="stretch")
        if "mean_confidence" in result:
            st.caption(f"Xác suất trung bình: {result['mean_confidence']:.3f} · Entropy trung bình: {result['mean_entropy']:.3f} · Ngưỡng rà soát: {result['threshold']:.2f}")
    st.subheader("Tải kết quả")
    name = Path(image_name).stem
    for column, key, label, suffix, mime in zip(st.columns(3), ("overlay", "mask", "regions"), ("Ảnh khoanh vùng", "Mask đã lọc", "Danh sách vùng"), ("khoanh-vung.png", "mask.png", "vung.json"), ("image/png", "image/png", "application/json")):
        column.download_button(label, files[key], file_name=f"{name}-{suffix}", mime=mime, width="stretch", on_click="ignore", key=f"single_dl_{key}")
    with st.expander("Bản đồ kiểm tra và sửa nhãn"):
        st.caption("Mở HTML để sửa vùng. Khi dùng LabelMe JSON, ghép với source.png cùng bộ xuất. Nhãn dự đoán cần chuyên gia xác nhận trước khi bổ sung vào dữ liệu huấn luyện.")
        omitted = summary.get("labelme", {}).get("omitted_regions", [])
        if omitted:
            st.warning(f"Có {len(omitted)} vùng không chuyển thành đa giác LabelMe vì có lỗ hoặc đường bao không biểu diễn chính xác. Mask pixel giữ nguyên các vùng này; xem lý do trong danh sách vùng JSON.")
        extra = {
            "raw_mask": ("Mask gốc chưa lọc", "raw-mask.png", "image/png"),
            "confidence": ("Bản đồ xác suất", "confidence.png", "image/png"),
            "uncertainty": ("Bản đồ bất định", "uncertainty.png", "image/png"),
            "review": ("Bản đồ vùng cần rà soát", "review.png", "image/png"),
            "labelme": ("Nhãn dự đoán LabelMe", "prediction.json", "application/json"),
            "source": ("Ảnh sạch ghép với LabelMe", "source.png", "image/png"),
            "trace": ("Công cụ sửa vùng ngoại tuyến", "review.html", "text/html"),
        }
        cols = st.columns(2)
        for i, (key, (label, suffix, mime)) in enumerate(extra.items()):
            if key in files:
                cols[i % 2].download_button(label, files[key], file_name="source.png" if key == "source" else f"{name}-{suffix}", mime=mime, on_click="ignore", key=f"single_dl_{key}")
    if classes:
        st.subheader("Thống kê theo tướng")
        st.dataframe([{"Tướng": x["name"], "Mã": x["class_id"], "Tỷ lệ ảnh (%)": 100 * x["area_fraction"], "Vùng được khoanh": x["region_count"]} for x in sorted(classes, key=lambda x: -x["pixel_count"])], hide_index=True, width="stretch", column_config={"Tỷ lệ ảnh (%)": st.column_config.NumberColumn(format="%.2f")})
    st.caption(f"Chỉ hiển thị vùng từ {min_region_area:,} pixel. Diện tích tính theo pixel ảnh, chưa quy đổi sang khoảng cách hoặc chiều sâu.")


def single_workspace(settings: dict):
    st.subheader("Ảnh lát cắt")
    use_sample = settings["mode"] == DEMO_MODE and st.checkbox("Dùng ảnh lát cắt mô phỏng có sẵn", value=True, key="single_sample")
    if use_sample:
        try:
            image_bytes, image_name = DEMO_IMAGE.read_bytes(), DEMO_IMAGE.name
        except OSError as exc:
            st.error(f"Không đọc được ảnh mô phỏng. {exc}")
            return
    else:
        uploaded = st.file_uploader("Chọn ảnh gốc chưa khoanh vùng", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"], key="single_image")
        image_bytes, image_name = (uploaded.getvalue(), uploaded.name) if uploaded else (None, "")
    if image_bytes is None:
        if settings["version"] is None:
            show_training_help()
        return
    from facies.data import load_image
    try:
        image = load_image(BytesIO(image_bytes))
    except (OSError, ValueError) as exc:
        st.error(f"Không đọc được ảnh lát cắt. {exc}")
        return
    h, w = image.shape[:2]
    st.caption(f"{image_name} · {w:,} × {h:,} pixel")
    if settings["version"] is None:
        show_training_help()
        st.image(image, width="stretch")
        return
    signature = hashlib.sha256(image_bytes).digest(), inference_signature(settings)
    if st.button("Chạy phân tướng", type="primary", key="run_single"):
        st.session_state.pop("facies_result", None)
        with st.spinner("Đang phân tướng và lập bản đồ rà soát…", show_time=True):
            try:
                result = predict_section(image, **{k: v for k, v in settings.items() if k != "mode"}, image_name=image_name)
            except Exception as exc:
                LOG.exception("Single inference failed")
                st.error(f"Không hoàn tất phân tướng. {exc}")
            else:
                st.session_state["facies_result"] = signature, result
    previous = st.session_state.get("facies_result")
    if previous and previous[0] == signature:
        render_output(image, previous[1], image_name, settings["min_region_area"])
    else:
        st.image(image, width="stretch")


def batch_workspace(settings: dict):
    st.subheader("Phân tướng nhiều lát cắt")
    st.caption("Đường dẫn trên máy chạy ứng dụng. Mỗi lần chạy tạo một thư mục kết quả mới.")
    source_value = st.text_input("Thư mục ảnh sạch", value="data/images", key="batch_source")
    output_value = st.text_input("Thư mục lưu các lần xử lý", value="runs/batch", key="batch_output")
    recursive = st.checkbox("Đọc cả thư mục con", value=False, key="batch_recursive")
    signature = source_value, output_value, recursive, inference_signature(settings)
    if settings["version"] is None:
        st.info("Chọn tệp trọng số hợp lệ trong thanh bên để xử lý thư mục.")
    if st.button("Chạy cả thư mục", type="primary", disabled=settings["version"] is None, key="run_batch"):
        st.session_state.pop("batch_result", None)
        bar = st.progress(0.0, text="Đang kiểm tra thư mục…")
        try:
            from facies.batch import process_folder
            from facies.predict import default_device
            source, output = read_checkpoint_path(source_value), read_checkpoint_path(output_value)
            if source is None or not source.is_dir():
                raise ValueError("Thư mục ảnh sạch không tồn tại.")
            if output is None:
                raise ValueError("Cần nhập thư mục lưu kết quả.")
            target = output / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6])
            result = process_folder(
                settings["checkpoint"], source, target, recursive=recursive,
                device=default_device() if settings["device"] == "Tự động" else "cpu", tile_size=settings["tile_size"],
                tta=settings["tta"], confidence_threshold=settings["confidence_threshold"],
                min_region_area=settings["min_region_area"], alpha=settings["alpha"], batch_size=2,
                progress=lambda done, total, name: bar.progress(done / max(total, 1), text=f"{done}/{total} · {name}"),
            )
        except Exception as exc:
            LOG.exception("Batch inference failed")
            st.error(f"Không hoàn tất xử lý thư mục. {exc}")
        else:
            st.session_state["batch_result"] = signature, result
        finally:
            bar.empty()
    previous = st.session_state.get("batch_result")
    if not previous or previous[0] != signature:
        return
    result = previous[1]
    rows, errors = result["results"], result["errors"]
    a, b = st.columns(2)
    a.metric("Ảnh xử lý được", len(rows), border=True)
    b.metric("Ảnh cần kiểm tra lỗi", len(errors), border=True)
    st.caption(f"Bảng kết quả: {result['index_csv']}")
    st.caption("Ưu tiên rà soát = 60% tỷ lệ pixel dưới ngưỡng + 40% entropy trung bình. Điểm này không phải tỷ lệ dự đoán sai.")
    if result.get("summary", {}).get("provenance", {}).get("demo"):
        st.warning("Thư mục đang dùng trọng số mô phỏng. Chưa có đánh giá chất lượng trên dữ liệu thực.")
    if rows:
        st.dataframe([{"Ưu tiên": x["rank"], "Ảnh": x["source_relative"], "Cần rà soát (%)": 100 * x["review_fraction"], "Entropy": x["mean_entropy"], "Điểm ưu tiên": x["priority_score"], "Vùng": x["region_count"]} for x in rows], hide_index=True, width="stretch")
        st.download_button("Tải bảng ưu tiên CSV", Path(result["index_csv"]).read_bytes(), file_name="review-priority.csv", mime="text/csv", on_click="ignore", key="batch_index_download")
        selected = st.selectbox("Mở ảnh để rà soát", range(len(rows)), format_func=lambda i: f"{rows[i]['rank']}. {rows[i]['source_relative']}", key="batch_selected")
        base = Path(result["index_csv"]).parent
        paths = {key: str(base / value) for key, value in rows[selected]["paths"].items()}
        if "overlay" in paths:
            st.image(paths["overlay"], width="stretch")
        if "trace" in paths:
            st.download_button("Tải công cụ sửa vùng cho ảnh này", Path(paths["trace"]).read_bytes(), file_name=Path(rows[selected]["source_relative"]).stem + "-review.html", mime="text/html", on_click="ignore", key="batch_trace_download")
    if errors:
        with st.expander("Các ảnh chưa xử lý"):
            st.dataframe(errors, hide_index=True, width="stretch")


def audit_workspace():
    st.subheader("Kiểm tra dữ liệu trước huấn luyện")
    st.caption("Kiểm tra cặp ảnh–nhãn, độ phủ, tướng hiếm và cách chia tuyến. Không tự sửa hoặc xóa dữ liệu.")
    a, b = st.columns(2)
    manifest_value = a.text_input("Tệp manifest CSV", value="data/dataset.csv", key="audit_manifest")
    classes_value = b.text_input("Tệp định nghĩa tướng", value="classes.json", key="audit_classes")
    if st.button("Kiểm tra bộ dữ liệu", key="run_audit"):
        st.session_state.pop("audit_result", None)
        try:
            from facies.audit import audit_dataset
            manifest, classes = read_checkpoint_path(manifest_value), read_checkpoint_path(classes_value)
            if manifest is None or classes is None:
                raise ValueError("Cần nhập cả tệp manifest và bộ nhãn.")
            with st.spinner("Đang kiểm tra từng ảnh và nhãn…", show_time=True):
                result = audit_dataset(manifest, classes)
        except Exception as exc:
            LOG.exception("Dataset audit failed")
            st.error(f"Không kiểm tra được dữ liệu. {exc}")
        else:
            st.session_state["audit_result"] = (manifest_value, classes_value), result
    previous = st.session_state.get("audit_result")
    if not previous or previous[0] != (manifest_value, classes_value):
        st.info("Chưa kiểm tra bộ dữ liệu trong phiên này. Chọn manifest khi đã chuẩn bị xong nhãn chuyên gia.")
        return
    render_audit_result(previous[1])


def render_audit_result(report: dict):
    summary = report.get("summary", {})
    errors, warnings = report.get("errors", []), report.get("warnings", [])
    if report.get("ready_for_training"):
        st.success("Bộ dữ liệu qua kiểm tra kỹ thuật. Chuyên gia vẫn cần xác nhận nội dung nhãn.")
    else:
        st.warning("Bộ dữ liệu còn vấn đề cần xem trước khi huấn luyện.")
    cols = st.columns(4)
    cols[0].metric("Lát cắt kiểm tra", summary.get("sections", 0), border=True)
    cols[1].metric("Lỗi cần sửa", len(errors), border=True)
    cols[2].metric("Cảnh báo", len(warnings), border=True)
    fraction = finite_number(summary.get("ignore_fraction"))
    cols[3].metric("Pixel chưa có nhãn", f"{100 * fraction:.1f}%" if fraction is not None else "—", border=True)
    for rows, label in ((errors, "Lỗi cần sửa"), (warnings, "Cảnh báo cần xem")):
        if rows:
            st.markdown(f"**{label}**")
            st.dataframe(rows, width="stretch", hide_index=True)
    splits = report.get("splits", {})
    if splits:
        st.markdown("**Độ phủ theo tập dữ liệu**")
        st.dataframe([{"Tập": key, "Lát cắt": value["sections"], "Lát cắt hợp lệ": value["valid_sections"], "Số nhóm": len(value["groups"]), "Pixel có nhãn": value["labeled_pixels"], "Pixel bỏ qua": value["ignored_pixels"]} for key, value in splits.items()], hide_index=True, width="stretch")
        selected = st.selectbox("Xem phân bố tướng của tập", list(splits), key="audit_split")
        classes = splits[selected]["classes"]
        if classes:
            st.dataframe([{"Tướng": x["name"], "Pixel": x["pixel_count"], "Lát cắt có tướng": x["section_count"], "Tỷ lệ pixel có nhãn": x["labeled_fraction"]} for x in classes], hide_index=True, width="stretch")
    st.download_button("Tải báo cáo kiểm tra JSON", json.dumps(report, ensure_ascii=False, indent=2), file_name="dataset-audit.json", mime="application/json", on_click="ignore", key="audit_download")
    with st.expander("Báo cáo kiểm tra đầy đủ"):
        st.json(report, expanded=False)


def finite_number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def load_metric_report(path: Path) -> dict:
    if path.stat().st_size > 25_000_000:
        raise ValueError("Báo cáo lớn hơn 25 MB; hãy chọn metrics.json của một lần huấn luyện.")
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict):
        raise ValueError("Không thấy kết quả validation/test trong báo cáo.")
    if not any(isinstance(report.get(key), dict) for key in ("validation", "test")):
        classes, matrix = report.get("classes"), report.get("confusion_matrix")
        valid_flat = (
            report.get("split") in ("val", "test")
            and isinstance(classes, list) and bool(classes)
            and all(isinstance(row, dict) for row in classes)
            and isinstance(matrix, list) and len(matrix) == len(classes)
            and all(isinstance(row, list) and len(row) == len(classes) for row in matrix)
            and all(type(value) is int and value >= 0 for row in matrix for value in row)
        )
        if not valid_flat:
            raise ValueError("Không thấy kết quả validation/test hợp lệ trong báo cáo.")
        # Evaluator reports are flat; retain every original field and source metric.
        # This adds only a display-compatible split, never a fabricated training history.
        split = "validation" if report["split"] == "val" else "test"
        report = {**report, split: dict(report)}
    for split in ("validation", "test"):
        values = report.get(split)
        if isinstance(values, dict) and "classes" in values:
            if not isinstance(values["classes"], list) or any(not isinstance(row, dict) for row in values["classes"]):
                raise ValueError(f"Danh sách điểm theo tướng trong {split} không hợp lệ.")
    return report


def experiment_rows(reports: dict[str, dict], split: str):
    rows = []
    for name, report in reports.items():
        values = report.get(split)
        if not isinstance(values, dict):
            values = {}
        config = report.get("config")
        if not isinstance(config, dict):
            config = {}
        rows.append({
            "Thí nghiệm": name, "Nguồn": "Mô phỏng" if report.get("demo") else ("Thực" if report.get("demo") is False else "Chưa khai báo"),
            "Kiến trúc": report.get("architecture", report.get("arch", config.get("arch", "unet"))),
            "SHA-256 dữ liệu (12 ký tự)": str(report.get("dataset_sha256", ""))[:12] or None,
            "Epoch tốt nhất": report.get("best_epoch"), "Lát cắt đánh giá": values.get("sections"),
            "mIoU tướng": finite_number(values.get("mean_iou_facies")), "Dice tướng": finite_number(values.get("mean_dice_facies")),
            "F1 biên": finite_number(values.get("mean_boundary_f1_facies")), "Độ đúng pixel": finite_number(values.get("pixel_accuracy")),
        })
    return rows


def render_experiment_detail(report: dict, split: str):
    if report.get("demo"):
        st.warning("Thí nghiệm trên ảnh mô phỏng. Điểm số dưới đây không chứng minh hiệu quả trên địa chấn thực.")
    history = report.get("history", [])
    if isinstance(history, list):
        points = [x for x in history if isinstance(x, dict) and finite_number(x.get("epoch")) is not None]
        for panel, key, label in zip(st.columns(2), ("loss", "val_miou_facies"), ("Hàm mất mát huấn luyện", "mIoU tướng · validation")):
            series = [{"Epoch": x["epoch"], label: finite_number(x.get(key))} for x in points if finite_number(x.get(key)) is not None]
            panel.markdown(f"**{label}**")
            if series:
                panel.line_chart(pd.DataFrame(series), x="Epoch", y=label, color="#08796e")
            else:
                panel.caption("Báo cáo chưa lưu chuỗi số liệu này.")
    values = report.get(split)
    if not isinstance(values, dict):
        st.info("Thí nghiệm này chưa có kết quả cho tập được chọn.")
        return
    classes = values.get("classes", [])
    if isinstance(classes, list) and classes:
        st.markdown("**Điểm theo từng tướng**")
        boundary_rows = values.get("boundary_classes")
        boundaries = {row["id"]: row for row in boundary_rows if isinstance(row, dict) and "id" in row} if isinstance(boundary_rows, list) else {}
        class_rows = []
        for row in classes:
            if not isinstance(row, dict):
                continue
            item = {"Tướng": row.get("name", row.get("id")), "Pixel chuyên gia": row.get("support_pixels"), "IoU": finite_number(row.get("iou")), "Dice": finite_number(row.get("dice")), "Precision": finite_number(row.get("precision")), "Recall": finite_number(row.get("recall"))}
            if boundaries:
                item["F1 biên"] = finite_number(boundaries.get(row.get("id"), {}).get("boundary_f1"))
            class_rows.append(item)
        st.dataframe(class_rows, hide_index=True, width="stretch")
        matrix = values.get("confusion_matrix")
        if isinstance(matrix, list) and len(matrix) == len(classes) and all(isinstance(x, list) and len(x) == len(classes) for x in matrix):
            st.markdown("**Ma trận nhầm lẫn · số pixel**")
            st.caption("Hàng là tướng chuyên gia; cột là tướng dự đoán. Pixel chưa có nhãn chuyên gia được loại khỏi phép tính.")
            names = [str(x.get("name", x.get("id"))) for x in classes]
            confusion = pd.DataFrame(matrix, index=names, columns=names)
            confusion.index.name = "Chuyên gia ↓ / Dự đoán →"
            st.dataframe(confusion, width="stretch")
    for key in ("aggregation", "boundary_definition"):
        if values.get(key):
            st.caption(values[key])
    with st.expander("Cấu hình và số liệu gốc"):
        st.json(report, expanded=False)


def experiment_workspace():
    st.subheader("Đối chiếu các lần huấn luyện")
    st.caption("Đọc số liệu từ metrics.json đã lưu. Chỉ so sánh trực tiếp khi dùng cùng dữ liệu, cách chia tuyến và định nghĩa điểm.")
    folder_value = st.text_input("Thư mục chứa thí nghiệm", value="runs", key="experiments_folder")
    candidates = []
    try:
        folder = read_checkpoint_path(folder_value)
        if folder and folder.is_dir():
            candidates = sorted(islice(folder.rglob("metrics.json"), 200))
        demo_folder = ROOT / "demo"
        if demo_folder.is_dir():
            candidates.extend(sorted(islice(demo_folder.rglob("metrics.json"), 50)))
        candidates = list(dict.fromkeys(str(x.resolve()) for x in candidates))
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Không đọc được thư mục báo cáo. {exc}")
    def short_name(value):
        try:
            return str(Path(value).relative_to(ROOT))
        except ValueError:
            return str(value)
    demo_path = str((ROOT / "demo" / "metrics.json").resolve())
    controlled = [str((ROOT / "demo" / name / "metrics.json").resolve()) for name in ("baseline", "attention")]
    defaults = controlled if all(path in candidates for path in controlled) else [demo_path] if demo_path in candidates else candidates[-1:]
    selected = st.multiselect("Báo cáo cần so sánh", candidates, default=defaults, format_func=short_name, key="selected_reports")
    if not selected:
        st.info("Chọn ít nhất một metrics.json. Sau huấn luyện, báo cáo nằm trong thư mục kết quả.")
        return
    reports = {}
    for value in selected:
        try:
            reports[short_name(value)] = load_metric_report(Path(value))
        except (OSError, ValueError, TypeError) as exc:
            st.error(f"Không đọc được {short_name(value)}. {exc}")
    if not reports:
        return
    fingerprints = {report["dataset_sha256"] for report in reports.values() if isinstance(report.get("dataset_sha256"), str)}
    if len(fingerprints) > 1:
        st.warning("Các báo cáo có dấu vân tay dữ liệu khác nhau; không suy ra mô hình tốt hơn chỉ từ chênh lệch điểm.")
    elif len(reports) > 1 and any(not report.get("dataset_sha256") for report in reports.values()):
        st.caption("Có báo cáo cũ thiếu dấu vân tay dữ liệu. Hãy kiểm tra lại tập đánh giá và danh sách nhóm trong số liệu gốc.")
    split = st.radio("Tập đánh giá", ("validation", "test"), horizontal=True, key="report_split")
    st.dataframe(experiment_rows(reports, split), hide_index=True, width="stretch", column_config={x: st.column_config.NumberColumn(format="%.4f") for x in ("mIoU tướng", "Dice tướng", "F1 biên", "Độ đúng pixel")})
    st.caption("Ô trống là số liệu chưa có hoặc không xác định; không được coi là 0. Báo cáo cũ vẫn xem được khi thiếu F1 biên.")
    chosen = st.selectbox("Xem chi tiết thí nghiệm", list(reports), key="report_detail")
    render_experiment_detail(reports[chosen], split)


def main():
    st.set_page_config(page_title="Phân tướng địa chấn", layout="wide", initial_sidebar_state="expanded")
    st.html(HEADER)
    settings = sidebar_settings()
    single, batch, audit, experiments = st.tabs(["Phân tướng một ảnh", "Xử lý thư mục", "Kiểm tra dữ liệu", "Thí nghiệm"])
    with single:
        single_workspace(settings)
    with batch:
        batch_workspace(settings)
    with audit:
        audit_workspace()
    with experiments:
        experiment_workspace()


if __name__ == "__main__":
    main()

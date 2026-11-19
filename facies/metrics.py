"""Metrics on expert-labeled pixels and boundaries away from unknown areas."""

import cv2
import numpy as np

from .data import IGNORE_INDEX


def _checked_pair(truth, prediction, num_classes, ignore_index):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.ndim != 2 or truth.shape != prediction.shape:
        raise ValueError("Mask chuyên gia và mask dự đoán phải là hai mảng HW cùng kích thước.")
    if not np.issubdtype(truth.dtype, np.integer) or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError("Mask tính điểm cần chứa ID nguyên.")
    valid = truth != ignore_index
    if np.any(((truth < 0) | (truth >= num_classes)) & valid):
        raise ValueError("Mask chuyên gia có ID ngoài bảng lớp.")
    if np.any((prediction < 0) | (prediction >= num_classes)):
        raise ValueError("Mask dự đoán có ID ngoài bảng lớp.")
    return truth, prediction, valid


def confusion_matrix(truth, prediction, num_classes, *, ignore_index=IGNORE_INDEX):
    """Rows are true IDs and columns predicted IDs; ignore applies to the expert mask."""
    truth, prediction, valid = _checked_pair(truth, prediction, num_classes, ignore_index)
    encoded = truth[valid].astype(np.int64) * num_classes + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def confusion_metrics(confusion, classes):
    """Summarize additive confusion counts while preserving earlier trainer report fields."""
    matrix = np.asarray(confusion)
    if matrix.shape != (len(classes), len(classes)) or not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError("Ma trận nhầm lẫn cần có số nguyên và kích thước bằng số lớp.")
    if np.any(matrix < 0):
        raise ValueError("Số pixel trong ma trận nhầm lẫn không được âm.")
    true_positive = matrix.diagonal()
    truth, prediction = matrix.sum(1), matrix.sum(0)
    rows = []
    for index, label in enumerate(classes):
        tp, support, predicted = (int(value[index]) for value in (true_positive, truth, prediction))
        union, total = support + predicted - tp, support + predicted
        rows.append({
            "id": label["id"], "name": label["name"], "support_pixels": support,
            "predicted_pixels": predicted, "true_positive_pixels": tp,
            "iou": tp / union if union else None,
            "dice": 2 * tp / total if total else None,
            "precision": tp / predicted if predicted else None,
            "recall": tp / support if support else None,
            "f1": 2 * tp / total if total else None,
        })
    supported = [row for row in rows[1:] if row["support_pixels"] > 0]
    present_union = [row for row in rows[1:] if row["iou"] is not None]
    return {
        "labeled_pixels": int(matrix.sum()),
        "pixel_accuracy": float(true_positive.sum() / matrix.sum()) if matrix.sum() else None,
        "mean_iou_facies": _mean(row["iou"] for row in supported),
        "mean_dice_facies": _mean(row["dice"] for row in supported),
        "mean_iou_facies_union": _mean(row["iou"] for row in present_union),
        "mean_f1_facies": _mean(row["f1"] for row in present_union),
        "classes": rows,
        "confusion_matrix": matrix.tolist(),
        "aggregation": (
            "mean_iou_facies/mean_dice_facies: trung bình các tướng có pixel chuyên gia; "
            "mean_iou_facies_union/mean_f1_facies: gồm cả tướng chỉ được dự đoán; loại nền ID 0. "
            "Giá trị không xác định là null; hàng confusion là nhãn thật, cột là dự đoán."
        ),
    }


def boundary_counts(truth, prediction, num_classes, *, tolerance=2, ignore_index=IGNORE_INDEX):
    """Return Cx4 additive [matched_pred, pred_edges, matched_true, true_edges].

    Class edges are inner 8-neighbor perimeters. Euclidean distance <= tolerance
    matches edges. Eroding the valid region by tolerance+1 excludes uncertain
    outlines at unknown labels and image borders from both precision and recall.
    """
    if type(tolerance) is not int or not 0 <= tolerance <= 128:
        raise ValueError("Dung sai biên cần là số nguyên trong khoảng 0–128 pixel.")
    truth, prediction, valid = _checked_pair(truth, prediction, num_classes, ignore_index)
    margin = tolerance + 1
    safe = cv2.erode(
        valid.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    counts = np.zeros((num_classes, 4), dtype=np.int64)
    kernel = np.ones((3, 3), np.uint8)
    for class_id in range(num_classes):
        edges = []
        for source in (prediction, truth):
            region = (source == class_id).astype(np.uint8)
            interior = cv2.erode(region, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            edges.append(region.astype(bool) & ~interior.astype(bool) & safe)
        predicted_edge, true_edge = edges
        counts[class_id, 1] = np.count_nonzero(predicted_edge)
        counts[class_id, 3] = np.count_nonzero(true_edge)
        if predicted_edge.any() and true_edge.any():
            to_truth = cv2.distanceTransform((~true_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            to_prediction = cv2.distanceTransform((~predicted_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            counts[class_id, 0] = np.count_nonzero(predicted_edge & (to_truth <= tolerance + 1e-6))
            counts[class_id, 2] = np.count_nonzero(true_edge & (to_prediction <= tolerance + 1e-6))
    return counts


def boundary_metrics(counts, classes, tolerance=2):
    counts = np.asarray(counts)
    if counts.shape != (len(classes), 4) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("Số đếm biên cần có kích thước số lớp × 4 và giá trị nguyên không âm.")
    if np.any(counts[:, 0] > counts[:, 1]) or np.any(counts[:, 2] > counts[:, 3]):
        raise ValueError("Số pixel khớp biên không được vượt tổng số pixel biên.")
    rows = []
    for label, row in zip(classes, counts):
        matched_pred, predicted, matched_true, truth = map(int, row)
        precision = matched_pred / predicted if predicted else None
        recall = matched_true / truth if truth else None
        f1 = None
        if predicted + truth:
            p, r = precision or 0.0, recall or 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({
            "id": label["id"], "name": label["name"],
            "boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1,
            "predicted_boundary_pixels": predicted, "true_boundary_pixels": truth,
            "matched_predicted_boundary_pixels": matched_pred, "matched_true_boundary_pixels": matched_true,
        })
    return {
        "boundary_tolerance_pixels": tolerance,
        "mean_boundary_f1_facies": _mean(row["boundary_f1"] for row in rows[1:]),
        "boundary_classes": rows,
        "boundary_definition": (
            "Biên phía trong theo 8 lân cận; khớp theo khoảng cách Euclid. "
            "Loại vùng cách nhãn 255 hoặc lề ảnh tối đa tolerance+1 pixel. "
            "Các lớp không có biên thật và dự đoán trong vùng hợp lệ có điểm null."
        ),
    }

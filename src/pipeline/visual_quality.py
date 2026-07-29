"""Shared visual-quality heuristic for live and cached video selection."""

from __future__ import annotations

from typing import Any


def crop_quality_components(
    crop: Any,
    bbox_xyxy: list[int],
    frame_shape: tuple[int, ...],
    yolo_confidence: float,
) -> dict[str, Any]:
    """Return the exact components used to rank a sign crop visually."""
    import cv2

    if crop is None or crop.size == 0:
        return {
            "crop_area": 0,
            "laplacian_variance": 0.0,
            "sharpness_factor": 0.0,
            "confidence_factor": 0.0,
            "edge_touches": 0,
            "edge_factor": 0.0,
            "final_score": 0.0,
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    area = int(crop.shape[0] * crop.shape[1])
    sharpness_factor = 1.0 + min(sharpness, 500.0) / 500.0
    confidence_factor = 0.5 + max(0.0, min(float(yolo_confidence), 1.0))

    x1, y1, x2, y2 = bbox_xyxy
    height, width = frame_shape[:2]
    margin = 4
    edge_touches = sum(
        [
            x1 <= margin,
            y1 <= margin,
            x2 >= width - margin,
            y2 >= height - margin,
        ]
    )
    edge_factor = 0.65 if edge_touches == 1 else 0.40 if edge_touches >= 2 else 1.0
    final_score = area * sharpness_factor * confidence_factor * edge_factor
    return {
        "crop_area": area,
        "laplacian_variance": sharpness,
        "sharpness_factor": sharpness_factor,
        "confidence_factor": confidence_factor,
        "edge_touches": edge_touches,
        "edge_factor": edge_factor,
        "final_score": final_score,
    }


def crop_quality(
    crop: Any,
    bbox_xyxy: list[int] | None = None,
    frame_shape: tuple[int, ...] | None = None,
    yolo_conf: float = 1.0,
) -> float:
    """Return only the final score, matching the historical live helper."""
    if bbox_xyxy is None or frame_shape is None:
        if crop is None or crop.size == 0:
            return 0.0
        height, width = crop.shape[:2]
        bbox_xyxy = [5, 5, max(5, width - 5), max(5, height - 5)]
        frame_shape = crop.shape
    return float(
        crop_quality_components(crop, bbox_xyxy, frame_shape, yolo_conf)[
            "final_score"
        ]
    )

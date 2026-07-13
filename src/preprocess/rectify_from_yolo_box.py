from pathlib import Path

import cv2
import numpy as np

from mobile_sam import sam_model_registry, SamPredictor


CHECKPOINT_PATH = "weights/mobile_sam/mobile_sam.pt"


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def warp_to_rect(img, src_pts):
    tl, tr, br, bl = src_pts

    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(br - tr), np.linalg.norm(bl - tl)))

    if width <= 0 or height <= 0:
        return None

    dst_pts = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, matrix, (width, height))


def get_sam_corners(mask_image):
    contours, _ = cv2.findContours(
        mask_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None, None

    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        points = approx.reshape(4, 2).astype("float32")
        corners = order_points(points)
        return corners, largest_contour, approx

    # Fallback for rounded-corner signs:
    # Use the minimum-area rectangle around the SAM mask.
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = box.astype("float32")

    corners = order_points(box)

    return corners, largest_contour, approx


def rectify_from_detection(image_path, detection, output_dir, predictor):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        print(f"Failed to load image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    yolo_box = np.array(detection["box"])
    x1, y1, x2, y2 = yolo_box.tolist()

    base_name = (
        f"{detection['index']}_"
        f"{detection['label'].replace(' ', '_')}_"
        f"{detection['confidence']:.2f}"
    )

    mask_path = output_dir / f"sam_mask_{base_name}.png"
    debug_path = output_dir / f"debug_{base_name}.jpg"
    output_path = output_dir / f"rectified_{base_name}.jpg"

    predictor.set_image(image_rgb)

    masks, scores, _ = predictor.predict(
        box=yolo_box,
        multimask_output=True
    )

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]

    mask_image = (best_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(mask_path), mask_image)

    corners, largest_contour, approx = get_sam_corners(mask_image)

    debug = image_bgr.copy()
    cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 6)

    if largest_contour is not None:
        cv2.drawContours(debug, [largest_contour], -1, (0, 0, 255), 8)

    if approx is not None:
        cv2.drawContours(debug, [approx], -1, (0, 255, 0), 8)

    cv2.imwrite(str(debug_path), debug)

    if corners is None:
        print(f"[{base_name}] SAM contour did not simplify to 4 corners.")
        return None

    warped = warp_to_rect(image_bgr, corners)

    if warped is None:
        print(f"[{base_name}] Warp failed.")
        return None

    cv2.imwrite(str(output_path), warped)

    print(f"[{base_name}] SAM score: {scores[best_idx]:.3f}")
    print(f"[{base_name}] Saved rectified image: {output_path}")

    return output_path


def crop_regions_from_detections(image_path, detections, output_dir):
    model = sam_model_registry["vit_t"](checkpoint=CHECKPOINT_PATH)
    predictor = SamPredictor(model)

    rectified_paths = []

    for detection in detections:
        rectified_path = rectify_from_detection(
            image_path=image_path,
            detection=detection,
            output_dir=output_dir,
            predictor=predictor
        )

        if rectified_path is not None:
            rectified_paths.append(rectified_path)

    return rectified_paths
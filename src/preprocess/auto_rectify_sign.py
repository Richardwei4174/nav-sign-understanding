from pathlib import Path
import shutil

import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def auto_rectify_crop(crop_path, output_dir):
    crop_path = Path(crop_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"rectified_{crop_path.name}"
    debug_path = output_dir / f"debug_{crop_path.name}"

    img = cv2.imread(str(crop_path))

    if img is None:
        print(f"Failed to load crop: {crop_path}")
        return crop_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_approx = None
    best_area = 0

    img_area = img.shape[0] * img.shape[1]
    min_area = img_area * 0.10

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        if len(approx) == 4 and area > best_area:
            best_approx = approx
            best_area = area

    if best_approx is None:
        shutil.copy(crop_path, output_path)
        print(f"No 4-corner candidate found. Copied original: {output_path}")
        return output_path

    points = best_approx.reshape(4, 2)
    src_pts = order_points(points)

    top_left, top_right, bottom_right, bottom_left = src_pts

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    output_width = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(bottom_right - top_right)
    height_left = np.linalg.norm(bottom_left - top_left)
    output_height = int(max(height_right, height_left))

    if output_width <= 0 or output_height <= 0:
        shutil.copy(crop_path, output_path)
        print(f"Invalid rectified size. Copied original: {output_path}")
        return output_path

    dst_pts = np.float32([
        [0, 0],
        [output_width, 0],
        [output_width, output_height],
        [0, output_height],
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(
        img,
        matrix,
        (output_width, output_height)
    )

    debug = img.copy()
    cv2.drawContours(debug, [best_approx], -1, (0, 0, 255), 4)

    cv2.imwrite(str(debug_path), debug)
    cv2.imwrite(str(output_path), warped)

    print(f"Rectified crop saved: {output_path}")

    return output_path


def auto_rectify_crops(crop_paths, output_dir):
    rectified_paths = []

    for crop_path in crop_paths:
        rectified_path = auto_rectify_crop(crop_path, output_dir)
        rectified_paths.append(rectified_path)

    return rectified_paths
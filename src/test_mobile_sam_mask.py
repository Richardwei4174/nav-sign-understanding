import cv2
import numpy as np

from mobile_sam import sam_model_registry, SamPredictor


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


image_path = "data/test_images/IMG_3444.JPG"
box = np.array([1708, 81, 2481, 2401])

checkpoint_path = "weights/mobile_sam/mobile_sam.pt"

model = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
predictor = SamPredictor(model)

image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise RuntimeError(f"Failed to load image: {image_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

predictor.set_image(image_rgb)

masks, scores, logits = predictor.predict(
    box=box,
    multimask_output=True
)

best_idx = int(np.argmax(scores))
best_mask = masks[best_idx]

mask_image = (best_mask.astype(np.uint8) * 255)
cv2.imwrite("outputs/sam_mask_test.png", mask_image)

contours, _ = cv2.findContours(
    mask_image,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    raise RuntimeError("No contour found from SAM mask.")

largest_contour = max(contours, key=cv2.contourArea)

epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

debug = image_bgr.copy()
cv2.rectangle(debug, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 6)
cv2.drawContours(debug, [largest_contour], -1, (0, 0, 255), 8)
cv2.drawContours(debug, [approx], -1, (0, 255, 0), 8)

cv2.imwrite("outputs/sam_contour_debug.png", debug)

if len(approx) != 4:
    raise RuntimeError(f"Expected 4 points, got {len(approx)}")

points = approx.reshape(4, 2).astype("float32")
src_pts = order_points(points)

top_left, top_right, bottom_right, bottom_left = src_pts

width_top = np.linalg.norm(top_right - top_left)
width_bottom = np.linalg.norm(bottom_right - bottom_left)
output_width = int(max(width_top, width_bottom))

height_right = np.linalg.norm(bottom_right - top_right)
height_left = np.linalg.norm(bottom_left - top_left)
output_height = int(max(height_right, height_left))

dst_pts = np.float32([
    [0, 0],
    [output_width, 0],
    [output_width, output_height],
    [0, output_height],
])

matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

warped = cv2.warpPerspective(
    image_bgr,
    matrix,
    (output_width, output_height)
)

cv2.imwrite("outputs/sam_rectified_test.png", warped)

print("Saved rectified image to outputs/sam_rectified_test.png")

print("Scores:", scores)
print("Approx points:", len(approx))
print("Saved:")
print("outputs/sam_mask_test.png")
print("outputs/sam_contour_debug.png")
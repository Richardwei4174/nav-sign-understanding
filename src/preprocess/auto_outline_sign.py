import cv2
import os

image_path = "data/Nav_sign_data/IMG_3347.JPG"
output_path = "outputs/debug_contours_IMG_3531.jpg"

img = cv2.imread(image_path)

if img is None:
    print(f"Failed to load image: {image_path}")
    exit()

# 1. Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Blur to reduce tiny noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Find edges
edges = cv2.Canny(blurred, 50, 150)

# 4. Find contours
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

debug = img.copy()

for contour in contours:
    area = cv2.contourArea(contour)

    # Ignore tiny outlines like text strokes
    if area < 5000:
        continue

    # Approximate contour with polygon
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

    # Draw candidate contours
    cv2.drawContours(debug, [approx], -1, (0, 0, 255), 8)

    print(f"Area: {area}, Points: {len(approx)}")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, debug)

print(f"Saved debug image to: {output_path}")
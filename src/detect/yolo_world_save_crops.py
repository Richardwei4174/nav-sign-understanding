from pathlib import Path
import cv2
from ultralytics import YOLOWorld


MODEL = None

def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def remove_duplicate_boxes(boxes, iou_threshold=0.6):
    unique_boxes = []

    for box in boxes:
        keep = True

        current_coords = box.xyxy[0].cpu().numpy()

        for kept_box in unique_boxes:
            kept_coords = kept_box.xyxy[0].cpu().numpy()

            if compute_iou(current_coords, kept_coords) > iou_threshold:
                keep = False
                break

        if keep:
            unique_boxes.append(box)

    return unique_boxes

def get_model():
    global MODEL

    if MODEL is None:
        MODEL = YOLOWorld("yolov8s-world.pt")

        MODEL.set_classes([
            "navigation sign",
            "directional sign",
            "wayfinding sign",
            "sign with arrow",
            "directional arrow sign",
            "hallway directional sign",
            "exit sign",
        ])

    return MODEL


def save_yolo_box_debug(image_path, boxes, model, output_path):
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Failed to load image for debug: {image_path}")
        return

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf_score = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(
            img,
            f"{i}: {label} {conf_score:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imwrite(str(output_path), img)


def save_yolo_crops(image_path, output_dir, conf=0.05):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Failed to load {image_path}")
        return []

    model = get_model()

    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=False,
        verbose=False,
    )

    boxes = results[0].boxes
    boxes = remove_duplicate_boxes(boxes, iou_threshold=0.6)

    crop_paths = []
    detections = []

    debug_box_path = output_dir.parent / "yolo_boxes.jpg"
    save_yolo_box_debug(image_path, boxes, model, debug_box_path)
    print(f"Saved YOLO box debug: {debug_box_path}")

    print(f"{image_path.name}: {len(boxes)} YOLO detections")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf_score = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        detections.append({
            "index": i,
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "label": label,
            "confidence": conf_score
        })        

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        safe_label = label.replace(" ", "_")
        crop_name = f"crop_{i}_{safe_label}_{conf_score:.2f}.jpg"
        crop_path = output_dir / crop_name

        cv2.imwrite(str(crop_path), crop)
        crop_paths.append(crop_path)

        print(f"  saved {crop_path}")

    return crop_paths, detections
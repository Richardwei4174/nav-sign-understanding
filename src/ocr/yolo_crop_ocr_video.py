from pathlib import Path
import json
import cv2
from ultralytics import YOLOWorld
from paddleocr import PaddleOCR


# Video
# ↓
# YOLO
# ↓
# OCR
# ↓
# ocr_results.json

# so watch the video and use Yolo to find signs and give it to OCR to read the text

ROOT = Path(__file__).resolve().parents[2]

video_path = ROOT / "data" / "test_videos" / "IMG_3772.MOV"

output_root = ROOT / "outputs" / "yolo_ocr_video"
crops_dir = output_root / "crops"
annotated_dir = output_root / "annotated_frames"
json_path = output_root / "ocr_results.json"
frames_dir = output_root / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)

crops_dir.mkdir(parents=True, exist_ok=True)
annotated_dir.mkdir(parents=True, exist_ok=True)

USE_BACKGROUND_CLASS = True

model = YOLOWorld("yolov8s-world.pt")

classes = [
    "navigation sign",
    "directional sign",
    "wayfinding sign",
    "sign with arrow",
    "directional arrow sign",
    "hallway directional sign",
    "exit sign",
]

if USE_BACKGROUND_CLASS:
    classes.append("")

model.set_classes(classes)

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=True,
    show_log=False,
)

results_json = []

results = model.predict(
    source=str(video_path),
    conf=0.05,
    iou=0.3,
    agnostic_nms=True,
    vid_stride=6,
    stream=True,
    save=False,
)

for frame_idx, result in enumerate(results):
    frame = result.orig_img.copy()
    h, w = frame.shape[:2]
    clean_frame_path = frames_dir / f"frame_{frame_idx:04d}.jpg"
    cv2.imwrite(str(clean_frame_path), frame)    

    frame_record = {
        "frame_index": frame_idx,
        "detections": []
    }

    for box_idx, box in enumerate(result.boxes):

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]

        crop_name = f"frame_{frame_idx:04d}_box_{box_idx:02d}.jpg"
        crop_path = crops_dir / crop_name

        cv2.imwrite(str(crop_path), crop)

        ocr_result = ocr.ocr(str(crop_path), cls=True)

        ocr_lines = []

        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                ocr_lines.append({
                    "text": line[1][0],
                    "confidence": round(float(line[1][1]), 3)
                })

        frame_record["detections"].append({
            "frame_path": str(clean_frame_path.relative_to(ROOT)),
            "crop_path": str(crop_path.relative_to(ROOT)),
            "bbox_xyxy": [x1, y1, x2, y2],
            "ocr": ocr_lines
        })

        # Draw bounding box for debugging
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

    annotated_path = annotated_dir / f"frame_{frame_idx:04d}.jpg"
    cv2.imwrite(str(annotated_path), frame)

    results_json.append(frame_record)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_json, f, indent=2)

print("YOLO + PaddleOCR complete.")
print(f"Crops: {crops_dir}")
print(f"Annotated frames: {annotated_dir}")
print(f"OCR results: {json_path}")
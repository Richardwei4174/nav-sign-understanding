from pathlib import Path
from ultralytics import YOLOWorld

ROOT = Path(__file__).resolve().parents[2]

video_path = ROOT / "data" / "test_videos" / "IMG_3772.MOV"
output_project = ROOT / "outputs" / "yolo_video_results"

USE_BACKGROUND_CLASS = True  # True = add ""

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

if USE_BACKGROUND_CLASS:
    experiment_name = "background_class_video_stride6_nms"
else:
    experiment_name = "baseline_video_stride6_nms"

experiment_name = "background_class_video_stride6_nms_iou03"

results = model.predict(
    source=str(video_path),
    save=True,
    conf=0.05,
    iou=0.3,          # remove overlapping duplicate boxes
    agnostic_nms=True, # suppress overlaps even if labels differ
    vid_stride=6,
    stream=True,
    project=str(output_project),
    name=experiment_name,
    exist_ok=True,
)

for _ in results:
    pass

print("YOLO-World video test complete.")
print(f"Saved output to: {output_project / experiment_name}")
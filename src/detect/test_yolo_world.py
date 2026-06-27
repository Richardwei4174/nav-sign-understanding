from pathlib import Path
from ultralytics import YOLOWorld

ROOT = Path(__file__).resolve().parents[2]

image_path = str(ROOT / "data" / "Nav_sign_data" / "IMG_3357.JPG")
output_project = str(ROOT / "runs" / "detect")

model = YOLOWorld("yolov8s-world.pt")

model.set_classes([
    "navigation sign",
    "directional sign",
    "wayfinding sign",
    "room number sign",
    "hallway sign",
    "sign with arrow",
])

results = model.predict(
    source=image_path,
    save=True,
    conf=0.05,
    project=output_project,
    name="yolo_world",
    exist_ok=True
)

print("YOLO-World test complete.")
print("Check runs/detect/predict or predict2.")
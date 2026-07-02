from pathlib import Path
from ultralytics import YOLOWorld

ROOT = Path(__file__).resolve().parents[2]

image_path = str(ROOT / "data" / "test_images" / "IMG_0001.jpg")
output_project = str(ROOT / "outputs" / "yolo_world_test")

model = YOLOWorld("yolov8s-world.pt")

model.set_classes([
    "navigation sign",
    "directional sign",
    "wayfinding sign",
    "sign with arrow",
    "directional arrow sign",
    "hallway directional sign",
    "exit sign",
    "" # this is so important to keep as it solves an important problem of finding nav signs
])

results = model.predict(
    source=image_path,
    save=True,
    conf=0.05,
    imgsz=1280,
    project=output_project,
    name="prompt",
    exist_ok=True
)
boxes = results[0].boxes
print(f"Detected {len(boxes)} objects")

print("YOLO-World test complete.")
print("Check outputs/yolo_world_test/baseline")

from pathlib import Path
from ultralytics import YOLOWorld

ROOT = Path(__file__).resolve().parents[2]

image_folder = ROOT / "data" / "Nav_sign_data"
output_project = ROOT / "outputs" / "yolo_world_results"

model = YOLOWorld("yolov8s-world.pt")

model.set_classes([
    "navigation sign",
    "directional sign",
    "wayfinding sign",
    "directional arrow sign",
    "hallway directional sign",
    "exit sign",
    "",
])

results = model.predict(
    source=str(image_folder),
    save=True,
    conf=0.10,
    imgsz=1280,
    project=str(output_project),
    name="no sign_with_arrow",
    exist_ok=True
)

print("\nYOLO-World batch test complete.")
print(f"Output saved to: {output_project / 'background_class'}")
print("\nDetection counts:")

for result in results:
    image_name = Path(result.path).name
    num_boxes = len(result.boxes)
    print(f"{image_name}: {num_boxes} detections")
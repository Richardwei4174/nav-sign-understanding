from ultralytics import YOLO

image_path = "data/Nav_sign_data/IMG_3347.JPG"

model = YOLO("yolov8n.pt")  # small pretrained YOLO model

results = model.predict(
    source=image_path,
    save=True,
    conf=0.25
)

print("YOLO test complete.")
print("Check the runs/detect/predict folder.")
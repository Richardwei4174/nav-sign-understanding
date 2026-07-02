from pathlib import Path
from paddleocr import PaddleOCR

ROOT = Path(__file__).resolve().parents[2]

image_path = ROOT / "data" / "test_images" / "IMG_0001.jpg"

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=True,
)

result = ocr.ocr(str(image_path), cls=True)

print("\nOCR RESULTS")
print("=" * 40)

for page in result:
    if page is None:
        continue

    for line in page:
        box = line[0]
        text = line[1][0]
        conf = line[1][1]

        print(f"Text: {text}")
        print(f"Confidence: {conf:.3f}")
        print(f"Box: {box}")
        print("-" * 40)
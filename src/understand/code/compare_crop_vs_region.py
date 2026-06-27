import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(SRC_ROOT))

from understand.code.rpi_continuous_testing import GeminiDirectionQA


def main():
    qa = GeminiDirectionQA(
        root=".",
        api_key_path="keys/gemini_api_key.yaml",
        model_version="gemini-3.5-flash",
        prompt_file="src/understand/prompts/qa_prompt.txt"
    )

    tight_crop = "outputs/pipeline/IMG_3347/crops/crop_0_navigation_sign_0.59.jpg"

    larger_region = "outputs/pipeline/IMG_3347/rectified/region_0_navigation_sign_0.59.jpg"

    questions = [
        "Where is 1201 - 1301?"
    ]

    test_images = {
        "tight_crop": tight_crop,
        "larger_region": larger_region
    }

    results = {}

    for name, image_path in test_images.items():
        print(f"\nTesting {name}: {image_path}")

        if not Path(image_path).exists():
            print(f"Missing image: {image_path}")
            results[name] = {
                "image_path": image_path,
                "error": "missing image"
            }
            continue

        raw_response = qa.ask_questions_for_image(
            image_path=image_path,
            questions=questions
        )

        parsed = qa.parse_response(raw_response)

        results[name] = {
            "image_path": image_path,
            "raw_response": raw_response,
            "parsed_response": parsed
        }

        print(raw_response)

    output_path = "outputs/compare_crop_vs_region.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved comparison to {output_path}")


if __name__ == "__main__":
    main()
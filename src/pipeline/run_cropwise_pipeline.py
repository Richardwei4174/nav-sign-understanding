import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path


# python -m src.pipeline.run_complete_pipeline \
#   --input-dir data/Nav_sign_data \
#   --output-root outputs/pipeline \
#   --qa-file src/understand/qa_test_set/rpi_test_set.json \
#   --prompt-file src/understand/prompts/qa_prompt.txt

# this version do one api call per rectified img. It combined the answers together for the final answer for each img. It only shows gemini the original img if there's any unknown or locational left for gemini to double check in case yolo did not detect the sign we are looking for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SRC_ROOT))

from detect.yolo_world_save_crops import save_yolo_crops
from preprocess.rectify_from_yolo_box import crop_regions_from_detections

from src.understand.code.rpi_continuous_testing import (
    GeminiDirectionQA,
    normalize_prediction,
    is_correct,
)


def make_image_output_dirs(image_path, output_root):
    image_path = Path(image_path)
    output_root = Path(output_root)

    image_name = image_path.stem
    image_output_dir = output_root / image_name

    crops_dir = image_output_dir / "crops"
    rectified_dir = image_output_dir / "rectified"

    image_output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    rectified_dir.mkdir(parents=True, exist_ok=True)

    original_copy_path = image_output_dir / image_path.name
    if not original_copy_path.exists():
        shutil.copy(image_path, original_copy_path)

    return {
        "image_output_dir": image_output_dir,
        "crops_dir": crops_dir,
        "rectified_dir": rectified_dir,
        "results_path": image_output_dir / "gemini_results.json",
    }


def get_image_paths(input_dir):
    input_dir = Path(input_dir)

    image_paths = []
    image_paths.extend(input_dir.glob("*.jpg"))
    image_paths.extend(input_dir.glob("*.JPG"))
    image_paths.extend(input_dir.glob("*.png"))
    image_paths.extend(input_dir.glob("*.PNG"))
    image_paths.extend(input_dir.glob("*.jpeg"))
    image_paths.extend(input_dir.glob("*.JPEG"))

    return sorted(image_paths)


def load_qa_lookup(qa_file):
    with open(qa_file, "r", encoding="utf-8") as f:
        qa_test_set = json.load(f)

    qa_lookup = {}

    for item in qa_test_set:
        image_name = item["imagePath"]
        qa_lookup[image_name] = item

    return qa_lookup


def ask_gemini_with_retries(
    qa,
    original_image_path,
    rectified_image_paths,
    questions,
):
    raw_response = None
    parsed_response = None
    attempt = 1

    while True:
        try:
            raw_response, parsed_response = qa.ask_questions_with_fallback(
                original_image_path=str(original_image_path),
                rectified_image_paths=[str(p) for p in rectified_image_paths],
                questions=questions,
            )

            if parsed_response is not None:
                print(f"Gemini succeeded on attempt {attempt}")
                return raw_response, parsed_response, attempt

            sleep_time = random.randint(15, 60)
            print(
                f"Attempt {attempt} failed to parse. "
                f"Retrying in {sleep_time} seconds..."
            )
            time.sleep(sleep_time)
            attempt += 1

        except Exception as e:
            sleep_time = random.randint(15, 60)
            print(f"Gemini error on attempt {attempt}: {e}")
            print(f"Retrying Gemini in {sleep_time} seconds...")
            time.sleep(sleep_time)
            attempt += 1


def ask_cropwise_then_merge(
    qa,
    original_image_path,
    rectified_image_paths,
    questions,
):
    all_crop_results = []
    final_response = {}

    for rectified_path in rectified_image_paths:
        print("\n--------------------------------")
        print(f"Testing single rectified crop: {rectified_path}")
        print("--------------------------------")

        attempt = 1

        while True:
            try:
                raw_response = qa.ask_questions_for_images(
                    image_paths=[str(rectified_path)],
                    questions=questions,
                )

                parsed_response = qa.parse_response(raw_response)

                if parsed_response is not None:
                    print(f"Gemini succeeded on crop attempt {attempt}")
                    attempts = attempt
                    break

                sleep_time = random.randint(15, 60)
                print(
                    f"Crop attempt {attempt} failed to parse. "
                    f"Retrying in {sleep_time} seconds..."
                )
                time.sleep(sleep_time)
                attempt += 1

            except Exception as e:
                sleep_time = random.randint(15, 60)
                print(f"Gemini crop error on attempt {attempt}: {e}")
                print(f"Retrying crop Gemini in {sleep_time} seconds...")
                time.sleep(sleep_time)
                attempt += 1

        all_crop_results.append({
            "rectified_image": str(rectified_path),
            "gemini_attempts": attempts,
            "raw_response": raw_response,
            "parsed_response": parsed_response,
        })

        for question in questions:
            answer = parsed_response.get(question, "unknown")

            if question not in final_response and answer not in ["unknown", "ERROR", None]:
                final_response[question] = answer

    fallback_questions = []

    for question in questions:
        answer = final_response.get(question, "unknown")

        if answer in ["unknown", "locational"]:
            fallback_questions.append(question)

    if fallback_questions:
        print("\n================================")
        print("FALLBACK TO ORIGINAL IMAGE")
        print("================================")

        raw_response, parsed_response, fallback_attempts = ask_gemini_with_retries(
            qa=qa,
            original_image_path=original_image_path,
            rectified_image_paths=[],
            questions=fallback_questions,
        )

        all_crop_results.append({
            "rectified_image": None,
            "source": "original_image_fallback",
            "gemini_attempts": fallback_attempts,
            "raw_response": raw_response,
            "parsed_response": parsed_response,
            "fallback_questions": fallback_questions,
        })

        if parsed_response is not None:
            for question in fallback_questions:
                final_response[question] = parsed_response.get(
                    question,
                    final_response.get(question, "unknown"),
                )
        else:
            for question in fallback_questions:
                final_response[question] = final_response.get(question, "unknown")

    return json.dumps(final_response), final_response, all_crop_results



def run_single_image(image_path, output_root, qa, qa_lookup):
    dirs = make_image_output_dirs(image_path, output_root)
    image_path = Path(image_path)
    image_name = image_path.name

    print("\n==============================")
    print(f"Processing image: {image_path}")
    print("==============================")
    print(f"Image folder: {dirs['image_output_dir']}")
    print(f"Crops folder: {dirs['crops_dir']}")
    print(f"Rectified folder: {dirs['rectified_dir']}")

    # Step 1: YOLO-World sign detection + crop saving
    crop_paths, detections = save_yolo_crops(
        image_path=image_path,
        output_dir=dirs["crops_dir"],
    )

    # Step 2: MobileSAM/OpenCV rectification from detections
    rectified_paths = crop_regions_from_detections(
        image_path=image_path,
        detections=detections,
        output_dir=dirs["rectified_dir"],
    )

    print(f"Found {len(rectified_paths)} rectified images for {image_name}")

    # Step 3: Find this image's questions
    qa_item = qa_lookup.get(image_name)
    gemini_results = []
    raw_response = None
    parsed_response = None
    gemini_attempts = 0
    crop_results = []

    if qa_item is None:
        print(f"No QA questions found for {image_name}")
    else:
        question_items = qa_item["questions"]
        questions = [q["question"] for q in question_items]

        # Step 4: Ask Gemini one rectified crop at a time, merge non-unknown answers,
        # then fallback to the original image for questions still unknown or locational.
        raw_response, parsed_response, crop_results = ask_cropwise_then_merge(
            qa=qa,
            original_image_path=image_path,
            rectified_image_paths=rectified_paths,
            questions=questions,
        )

        gemini_attempts = sum(
            crop["gemini_attempts"] for crop in crop_results
        )

        if parsed_response is None:
            parsed_response = {}

        image_correct = 0
        image_total = 0

        for q in question_items:
            question = q["question"]
            expected = q["answer"]

            predicted = normalize_prediction(
                parsed_response.get(question, "ERROR")
            )

            correct = is_correct(predicted, expected)

            image_total += 1
            if correct:
                image_correct += 1

            result_entry = {
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
            }

            if predicted == "ERROR":
                result_entry["error_response"] = raw_response

            gemini_results.append(result_entry)

            print(f"  {question} | expected={expected} | predicted={predicted}")

        print(
            f"Image accuracy: {image_correct}/{image_total} "
            f"({image_correct / image_total if image_total > 0 else 0:.4f})"
        )

    results = {
        "input_image": str(image_path),
        "detections": detections,
        "crop_paths": [str(p) for p in crop_paths],
        "rectified_paths": [str(p) for p in rectified_paths],
        "gemini_attempts": gemini_attempts,
        "cropwise_results": crop_results,
        "raw_gemini_response": raw_response,
        "parsed_gemini_response": parsed_response,
        "gemini_results": gemini_results,
    }

    with open(dirs["results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved: {dirs['results_path']}")

    return {
        "imagePath": image_name,
        "results": gemini_results,
        "image_correct": sum(1 for r in gemini_results if r["correct"]),
        "image_total": len(gemini_results),
        "image_accuracy": (
            sum(1 for r in gemini_results if r["correct"]) / len(gemini_results)
            if gemini_results else 0
        ),
    }    


def run_pipeline(
    input_dir,
    output_root,
    qa_file,
    root,
    api_key_path,
    model_version,
    prompt_file,
    retry_count,
    output,
):
    image_paths = get_image_paths(input_dir)

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    qa_lookup = load_qa_lookup(qa_file)

    qa = GeminiDirectionQA(
        root=root,
        api_key_path=api_key_path,
        model_version=model_version,
        prompt_file=prompt_file,
    )

    results = []

    total_questions = 0
    total_correct = 0

    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"Loaded QA questions for {len(qa_lookup)} images")

    total_images = len(image_paths)
    for i, image_path in enumerate(image_paths, start=1):
        print("\n========================================")
        print(f"Progress: {i}/{total_images}")
        print("========================================")

        image_result = run_single_image(
            image_path=image_path,
            output_root=output_root,
            qa=qa,
            qa_lookup=qa_lookup,
        )

        results.append(image_result)

        total_questions += image_result["image_total"]
        total_correct += image_result["image_correct"]

        combined_output = {
            "summary": {
                "total_questions": total_questions,
                "total_correct": total_correct,
                "accuracy": (
                    total_correct / total_questions
                    if total_questions > 0 else 0
                ),
            },
            "results": results,
        }

        with open(output, "w", encoding="utf-8") as f:
            json.dump(combined_output, f, indent=4, ensure_ascii=False)        

        # Small delay between images to avoid hammering the API.
        time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="data/test_images",
        help="Folder containing input images",
    )

    parser.add_argument(
        "--output-root",
        default="outputs/pipeline",
        help="Root folder for pipeline outputs",
    )

    parser.add_argument(
        "--qa-file",
        default="src/understand/qa_test_set/rpi_test_set.json",
        help="QA JSON file containing imagePath and questions",
    )

    parser.add_argument(
        "--root",
        default=".",
        help="Project root used by GeminiDirectionQA",
    )

    parser.add_argument(
        "--api-key-path",
        default="keys/gemini_api_key.yaml",
        help="Path to Gemini API key YAML relative to root",
    )

    parser.add_argument(
        "--model-version",
        default="gemini-3.5-flash",
        help="Gemini model version",
    )

    parser.add_argument(
        "--prompt-file",
        default="src/understand/prompts/qa_prompt.txt",
        help="Prompt file relative to root",
    )

    parser.add_argument(
        "--retry-count",
        type=int,
        default=3,
        help="Number of Gemini retry attempts per image",
    )
    parser.add_argument(
        "--output",
        default="outputs/complete_pipeline_results.json",
        help="Combined QA results JSON",
    )

    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        output_root=args.output_root,
        qa_file=args.qa_file,
        root=args.root,
        api_key_path=args.api_key_path,
        model_version=args.model_version,
        prompt_file=args.prompt_file,
        retry_count=args.retry_count,
        output=args.output,
    )
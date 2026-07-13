import argparse
import base64
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
import cv2




# python -m src.pipeline.run_multiview_pipeline \
#   --image data/Nav_sign_data/IMG_0001.jpg \
#   --output-root outputs/pipeline_multiview_single \
#   --qa-file src/understand/qa_test_set/rpi_test_set.json \
#   --prompt-file src/understand/prompts/qa_prompt.txt \
#   --output outputs/multiview_single_result.json


# python -m src.pipeline.run_multiview_pipeline \
#   --input-dir data/test_images \
#   --output-root outputs/checking_test_images \
#   --qa-file src/understand/qa_test_set/rpi_test_set.json \
#   --prompt-file src/understand/prompts/qa_prompt.txt \
#   --output outputs/checking_test_results.json

# Argument	Purpose
# --image	Process one image.
# --input-dir	Process an entire folder (don't use with --image).
# --output-root	Where each image's pipeline outputs (crops/, rectified/, annotated_original.jpg, etc.) are stored.
# --qa-file	Loads the QA questions for evaluation.
# --prompt-file	Loads the Gemini prompt.
# --output	Writes the combined summary JSON (overall accuracy across all processed images).


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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_string(image_path):
    return f"data:image/jpg;base64,{encode_image(image_path)}"

def create_annotated_original_image(
    image_path,
    detections,
    output_path,
):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["box"]

        label = f"Image {i + 2}"

        cv2.rectangle(
            image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            8,
        )

        cv2.putText(
            image,
            label,
            (int(x1), max(int(y1) - 15, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 255, 0),
            5,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), image)

    return output_path

def ask_multiview_with_retries(
    qa,
    original_image_path,
    rectified_image_paths,
    detections,
    questions,
):
    attempt = 1

    crop_descriptions = []

    for i, rectified_path in enumerate(rectified_image_paths):
        if i < len(detections):
            box = detections[i]["box"]
            label = detections[i].get("label", "unknown")
            confidence = detections[i].get("confidence", "unknown")
        else:
            box = "unknown"
            label = "unknown"
            confidence = "unknown"

        crop_descriptions.append(
            f"Image {i + 2}: This is the rectified version of the region labeled "
            f"'Image {i + 2}' in Image 1."
        )

    multiview_note = (
        "You are given multiple views of the SAME scene.\n\n"

        "Image 1 is the ORIGINAL image with bounding boxes drawn around each "
        "detected navigational sign. Each box is labeled Crop 1, Crop 2, etc.\n\n"

        "The remaining images are rectified close-up views of those labeled boxes:\n"
        + "\n".join(crop_descriptions)
        + "\n\n"

        "Use Image 1 to understand the overall scene layout, spatial relationships, "
        "and which sign each crop belongs to.\n"

        "Use the rectified crop images to read small text and determine arrow "
        "directions more accurately.\n"

        "The rectified images are NOT independent signs. Each one is simply a "
        "higher-quality view of the corresponding labeled region in Image 1.\n"

        "Reason jointly across ALL images before answering the questions."
    )

    while True:
        try:
            question_text = "\n".join([f"- {q}" for q in questions])

            full_prompt = (
                f"{qa.qa_prompt}\n\n"
                f"{multiview_note}\n\n"
                f"Questions:\n"
                f"{question_text}"
            )

            content = [
                {
                    "type": "text",
                    "text": full_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": get_image_string(original_image_path),
                        "detail": "high",
                    },
                },
            ]

            for rectified_path in rectified_image_paths:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": get_image_string(rectified_path),
                        "detail": "high",
                    },
                })

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant capable of understanding "
                        "navigational signs."
                    ),
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]

            completion = qa.client.chat.completions.create(
                model=qa.model_version,
                messages=messages,
                n=1,
                temperature=0,
            )

            raw_response = completion.choices[0].message.content
            parsed_response = qa.parse_response(raw_response)

            if parsed_response is not None:
                print(f"Gemini multiview succeeded on attempt {attempt}")
                return raw_response, parsed_response, attempt

            sleep_time = random.randint(15, 60)
            print(
                f"Multiview attempt {attempt} failed to parse. "
                f"Retrying in {sleep_time} seconds..."
            )
            time.sleep(sleep_time)
            attempt += 1

        except Exception as e:
            sleep_time = random.randint(15, 60)
            print(f"Gemini multiview error on attempt {attempt}: {e}")
            print(f"Retrying multiview Gemini in {sleep_time} seconds...")
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


def run_multiview_from_detections(
    image_path,
    output_root,
    qa,
    question_items,
    detections,
    crop_paths=None,
    image_output_name=None,
):
    if image_output_name is None:
        dirs = make_image_output_dirs(image_path, output_root)
    else:
        image_output_dir = Path(output_root) / image_output_name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        dirs = {
            "image_output_dir": image_output_dir,
            "crops_dir": image_output_dir / "crops",
            "rectified_dir": image_output_dir / "rectified",
            "results_path": image_output_dir / "gemini_results.json",
        }

        dirs["crops_dir"].mkdir(exist_ok=True)
        dirs["rectified_dir"].mkdir(exist_ok=True)

    image_path = Path(image_path)
    image_name = image_path.name

    copied_crop_paths = []

    if crop_paths:
        for crop_path in crop_paths:
            crop_path = Path(crop_path)
            copied_crop_path = dirs["crops_dir"] / crop_path.name

            if crop_path.exists():
                shutil.copy(crop_path, copied_crop_path)
                copied_crop_paths.append(copied_crop_path)
            else:
                print(f"Warning: crop path does not exist: {crop_path}")

    rectified_paths = crop_regions_from_detections(
        image_path=image_path,
        detections=detections,
        output_dir=dirs["rectified_dir"],
    )

    annotated_original_path = (
        dirs["image_output_dir"] / "annotated_original.jpg"
    )

    create_annotated_original_image(
        image_path=image_path,
        detections=detections,
        output_path=annotated_original_path,
    )

    print(f"Found {len(rectified_paths)} rectified images for {image_name}")

    questions = [q["question"] for q in question_items]

    raw_response, parsed_response, gemini_attempts = ask_multiview_with_retries(
        qa=qa,
        original_image_path=annotated_original_path,
        rectified_image_paths=rectified_paths,
        detections=detections,
        questions=questions,
    )

    if parsed_response is None:
        parsed_response = {}

    gemini_results = []
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
        "crop_paths": [str(p) for p in copied_crop_paths],
        "rectified_paths": [str(p) for p in rectified_paths],
        "gemini_attempts": gemini_attempts,
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
        "image_correct": image_correct,
        "image_total": image_total,
        "image_accuracy": (
            image_correct / image_total if image_total > 0 else 0
        ),
    }

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

    annotated_original_path = (
        dirs["image_output_dir"] / "annotated_original.jpg"
    )

    create_annotated_original_image(
        image_path=image_path,
        detections=detections,
        output_path=annotated_original_path,
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

        # Step 4: 
        # then fallback to the original image for questions still unknown or locational.
        raw_response, parsed_response, gemini_attempts = ask_multiview_with_retries(
            qa=qa,
            original_image_path=annotated_original_path,
            rectified_image_paths=rectified_paths,
            detections=detections,
            questions=questions,
        )

        crop_results = []

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
        "multiview_results": crop_results,
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
    image,
    output_root,
    qa_file,
    root,
    api_key_path,
    model_version,
    prompt_file,
    retry_count,
    output,
):
    if image is not None:
        image_paths = [Path(image)]
    else:
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
        "--image",
        default=None,
        help="Run the pipeline on a single image instead of an entire folder.",
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
        image=args.image,
        output_root=args.output_root,
        qa_file=args.qa_file,
        root=args.root,
        api_key_path=args.api_key_path,
        model_version=args.model_version,
        prompt_file=args.prompt_file,
        retry_count=args.retry_count,
        output=args.output,
    )
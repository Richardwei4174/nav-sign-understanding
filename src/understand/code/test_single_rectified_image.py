import argparse
import base64
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

# python -m src.understand.code.test_single_rectified_image   --image data/Nav_sign_data/IMG_3465.JPG   --output-dir outputs/single_crop_test/IMG_3465

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SRC_ROOT))

from openai import OpenAI
from src.utils import file_utils

from detect.yolo_world_save_crops import save_yolo_crops
from preprocess.rectify_from_yolo_box import crop_regions_from_detections


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_string(image_path):
    return f"data:image/jpg;base64,{encode_image(image_path)}"


def ask_gemini(client, model_version, prompt_file, image_path, questions):
    qa_prompt = file_utils.read_prompt(prompt_file)

    question_text = "\n".join([f"- {q}" for q in questions])

    full_prompt = f"{qa_prompt}\n\nQuestions:\n{question_text}"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant capable of understanding navigational signs.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": get_image_string(image_path),
                        "detail": "high",
                    },
                },
            ],
        },
    ]

    completion = client.chat.completions.create(
        model=model_version,
        messages=messages,
        n=1,
        temperature=0,
    )

    return completion.choices[0].message.content


def parse_response(raw_response):
    try:
        cleaned = raw_response.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception:
        return None


def ask_with_retries(client, model_version, prompt_file, image_path, questions):
    attempt = 1

    while True:
        try:
            raw_response = ask_gemini(
                client=client,
                model_version=model_version,
                prompt_file=prompt_file,
                image_path=image_path,
                questions=questions,
            )

            parsed_response = parse_response(raw_response)

            if parsed_response is not None:
                print(f"Gemini succeeded on attempt {attempt}")
                return raw_response, parsed_response

            sleep_time = random.randint(15, 60)
            print(f"Attempt {attempt} failed to parse. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
            attempt += 1

        except Exception as e:
            sleep_time = random.randint(15, 60)
            print(f"Gemini error on attempt {attempt}: {e}")
            print(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
            attempt += 1


def make_output_dirs(image_path, output_dir):
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    crops_dir = output_dir / "crops"
    rectified_dir = output_dir / "rectified"

    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    rectified_dir.mkdir(parents=True, exist_ok=True)

    original_copy = output_dir / image_path.name
    if not original_copy.exists():
        shutil.copy(image_path, original_copy)

    return crops_dir, rectified_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--qa-file",
        default="src/understand/qa_test_set/rpi_test_set.json",
    )

    parser.add_argument("--question", action="append")

    parser.add_argument("--root", default=".")
    parser.add_argument("--api-key-path", default="keys/gemini_api_key.yaml")
    parser.add_argument("--model-version", default="gemini-3.5-flash")
    parser.add_argument(
        "--prompt-file",
        default="src/understand/prompts/qa_prompt.txt",
    )

    args = parser.parse_args()

    questions = args.question

    if questions is None:
        with open(args.qa_file, "r", encoding="utf-8") as f:
            qa_test_set = json.load(f)

        image_name = Path(args.image).name

        for item in qa_test_set:
            if item["imagePath"] == image_name:
                questions = [
                    q["question"]
                    for q in item["questions"]
                ]
                break

        if questions is None:
            raise ValueError(f"No QA questions found for {image_name}")

    print(f"Loaded {len(questions)} questions.")


    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    crops_dir, rectified_dir = make_output_dirs(image_path, output_dir)

    print("\n==============================")
    print(f"Running one-image crop test: {image_path}")
    print("==============================")
    print(f"Output folder: {output_dir}")
    print(f"Crops folder: {crops_dir}")
    print(f"Rectified folder: {rectified_dir}")

    crop_paths, detections = save_yolo_crops(
        image_path=image_path,
        output_dir=crops_dir,
    )

    rectified_paths = crop_regions_from_detections(
        image_path=image_path,
        detections=detections,
        output_dir=rectified_dir,
    )

    print(f"\nDetections: {len(detections)}")
    print(f"Crops: {len(crop_paths)}")
    print(f"Rectified images: {len(rectified_paths)}")

    api_key = file_utils.load_yaml(
        os.path.join(args.root, args.api_key_path)
    )["api_key"]

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    prompt_file = os.path.join(args.root, args.prompt_file)

    all_results = []

    for rectified_path in rectified_paths:
        print("\n================================")
        print(f"Testing rectified image: {rectified_path}")
        print("================================")

        raw_response, parsed_response = ask_with_retries(
            client=client,
            model_version=args.model_version,
            prompt_file=prompt_file,
            image_path=rectified_path,
            questions=questions,
        )

        all_results.append({
            "rectified_image": str(rectified_path),
            "raw_response": raw_response,
            "parsed_response": parsed_response,
        })

        print("\nRaw response:")
        print(raw_response)

        print("\nParsed response:")
        print(json.dumps(parsed_response, indent=2, ensure_ascii=False))

    final_response = {}

    for result in all_results:
        parsed = result["parsed_response"]

        for question in questions:
            answer = parsed.get(question, "unknown")

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

        raw_response, parsed_response = ask_with_retries(
            client=client,
            model_version=args.model_version,
            prompt_file=prompt_file,
            image_path=image_path,
            questions=fallback_questions,
        )

        print("\nFallback response:")
        print(json.dumps(parsed_response, indent=2, ensure_ascii=False))

        for question in fallback_questions:
            final_response[question] = parsed_response.get(
                question,
                final_response.get(question, "unknown"),
            )        


    summary_path = output_dir / "single_crop_test_results.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_image": str(image_path),
                "detections": detections,
                "crop_paths": [str(p) for p in crop_paths],
                "rectified_paths": [str(p) for p in rectified_paths],
                "questions": questions,
                "results": all_results,
                "final_response": final_response,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n================================")
    print("SUMMARY")
    print("================================")

    for result in all_results:
        print(f"\nImage: {result['rectified_image']}")
        parsed = result["parsed_response"]

        for question in questions:
            print(f"  {question} -> {parsed.get(question, 'MISSING')}")

    print("\n================================")
    print("FINAL MERGED RESPONSE")
    print("================================")

    for question in questions:
        print(f"{question} -> {final_response[question]}")

    print(f"\nSaved test results to: {summary_path}")


if __name__ == "__main__":
    main()
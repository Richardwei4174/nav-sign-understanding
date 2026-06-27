import argparse
import base64
import json
import os
import time
import random
from openai import OpenAI
from src.utils import file_utils


# python -m src.understand.code.rpi_continuous_testing \
#   --image-folder data/Nav_sign_data \
#   --qa-file src/understand/qa_test_set/rpi_test_set.json \
#   --output outputs/qa_results.json \
#   --prompt-file src/understand/prompts/qa_prompt.txt



class GeminiDirectionQA:
    def __init__(self, root, api_key_path, model_version, prompt_file):
        self.root = root
        self.api_key_path = api_key_path
        self.model_version = model_version
        self.prompt_file = prompt_file
        self.setup_model()
        self.qa_prompt = file_utils.read_prompt(
            os.path.join(self.root, self.prompt_file)
        )

    def setup_model(self):
        api_key = file_utils.load_yaml(
            os.path.join(self.root, self.api_key_path)
        )["api_key"]

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_image_string(self, image_path):
        return f"data:image/jpg;base64,{self.encode_image(image_path)}"

    # original image + prompt + questions (fall back)
    def ask_questions_for_image(self, image_path, questions):
        question_text = "\n".join([f"- {q}" for q in questions])

        full_prompt = (
            f"{self.qa_prompt}\n\n"
            f"Questions:\n"
            f"{question_text}"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant capable of understanding "
                    "navigational signs."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": full_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self.get_image_string(image_path),
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

        completion = self.client.chat.completions.create(
            model=self.model_version,
            messages=messages,
            n=1,
            temperature=0
        )

        return completion.choices[0].message.content
    
    # rectified image included
    def ask_questions_for_images(self, image_paths, questions):
        question_text = "\n".join([f"- {q}" for q in questions])

        full_prompt = (
            f"{self.qa_prompt}\n\n"
            f"Questions:\n"
            f"{question_text}"
        )

        content = [
            {
                "type": "text",
                "text": full_prompt
            }
        ]

        for image_path in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": self.get_image_string(image_path),
                    "detail": "high"
                }
            })

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant capable of understanding "
                    "navigational signs."
                )
            },
            {
                "role": "user",
                "content": content
            }
        ]

        completion = self.client.chat.completions.create(
            model=self.model_version,
            messages=messages,
            n=1,
            temperature=0
        )

        return completion.choices[0].message.content    

    def parse_response(self, raw_response):
        try:
            cleaned = raw_response.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except Exception:
            return None
    def ask_questions_with_fallback(self, original_image_path, rectified_image_paths, questions):
        if rectified_image_paths:
            raw_rectified = self.ask_questions_for_images(
                image_paths=rectified_image_paths,
                questions=questions
            )
            parsed_rectified = self.parse_response(raw_rectified)
        else:
            raw_rectified = None
            parsed_rectified = None

        unknown_questions = get_unknown_questions(parsed_rectified, questions)

        if parsed_rectified is not None and not unknown_questions:
            return raw_rectified, parsed_rectified

        raw_original = self.ask_questions_for_image(
            image_path=original_image_path,
            questions=unknown_questions
        )
        parsed_original = self.parse_response(raw_original)

        final_response = parsed_rectified if parsed_rectified is not None else {}

        if parsed_original is not None:
            for question in unknown_questions:
                final_response[question] = parsed_original.get(question, "unknown")

        return json.dumps(final_response), final_response
        

#yes the two function is indented correctly
def normalize_prediction(predicted):
    if isinstance(predicted, list):
        # remove duplicates while preserving order
        cleaned = []
        for p in predicted:
            if p not in cleaned:
                cleaned.append(p)

        if len(cleaned) == 1:
            return cleaned[0]

        return cleaned

    return predicted



def is_correct(predicted, expected):
    if isinstance(expected, list) and isinstance(predicted, list):
        return set(predicted) == set(expected)

    if isinstance(expected, list):
        return predicted in expected

    if isinstance(predicted, list):
        return expected in predicted

    return predicted == expected

def get_rectified_images(image_name):
    image_stem = os.path.splitext(image_name)[0]

    rectified_dir = os.path.join(
        "outputs",
        "pipeline",
        image_stem,
        "rectified"
    )

    if not os.path.exists(rectified_dir):
        return []

    rectified_images = []

    for filename in sorted(os.listdir(rectified_dir)):
        if filename.startswith("rectified_") and filename.endswith(".jpg"):
            rectified_images.append(os.path.join(rectified_dir, filename))

    return rectified_images

def get_unknown_questions(parsed_response, questions):
    unknown_questions = []

    if parsed_response is None:
        return questions

    for question in questions:
        answer = parsed_response.get(question, "unknown")

        if answer in ["unknown", "ERROR", None]:
            unknown_questions.append(question)

    return unknown_questions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default=".")
    parser.add_argument("--image-folder", default="gt/fullpipeline_dataset")
    parser.add_argument("--qa-file", default="qa_test_set.json")
    parser.add_argument("--output", default="qa_results.json")
    parser.add_argument("--api-key-path", default="keys/gemini_api_key.yaml")
    parser.add_argument("--model-version", default="gemini-3.5-flash")
    parser.add_argument("--prompt-file", default="prompts/qa_prompt.txt")

    args = parser.parse_args()

    qa = GeminiDirectionQA(
        root=args.root,
        api_key_path=args.api_key_path,
        model_version=args.model_version,
        prompt_file=args.prompt_file
    )

    with open(args.qa_file, "r", encoding="utf-8") as f:
        qa_test_set = json.load(f)

    # TEST MODE: only first 3 images
    qa_test_set = qa_test_set[:3]

    results = []
    total_questions = 0
    total_correct = 0

    for item_idx, item in enumerate(qa_test_set):
        image_name = item["imagePath"]
        image_path = os.path.join(args.image_folder, image_name)

        print(f"\nProcessing image {item_idx + 1}/{len(qa_test_set)}: {image_name}")

        image_result = {
            "imagePath": image_name,
            "results": []
        }

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            results.append(image_result)
            continue

        question_items = item["questions"]
        questions = [q["question"] for q in question_items]

        raw_response = None
        parsed_response = None

        attempt = 1

        while True:
            try:

                rectified_image_paths = get_rectified_images(image_name)

                print(f"Found {len(rectified_image_paths)} rectified images for {image_name}")

                raw_response, parsed_response = qa.ask_questions_with_fallback(
                    original_image_path=image_path,
                    rectified_image_paths=rectified_image_paths,
                    questions=questions
                )

                if parsed_response is not None:
                    break

                sleep_time = random.randint(15, 60)
                print(f"Attempt {attempt} failed to parse. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                attempt += 1

            except Exception as e:
                print(f"Error on {image_name}: {e}")
                sleep_time = random.randint(15, 60)
                print(f"Attempt {attempt} failed. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                attempt += 1

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
            total_questions += 1

            if correct:
                image_correct += 1
                total_correct += 1

            result_entry = {
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "correct": correct
            }

            if predicted == "ERROR":
                result_entry["error_response"] = raw_response

            image_result["results"].append(result_entry)

            print(f"  {question} | expected={expected} | predicted={predicted}")

        image_result["image_correct"] = image_correct
        image_result["image_total"] = image_total
        image_result["image_accuracy"] = (
            image_correct / image_total if image_total > 0 else 0
        )

        results.append(image_result)

        partial_output = {
            "summary": {
                "total_questions_so_far": total_questions,
                "total_correct_so_far": total_correct,
                "accuracy_so_far": (
                    total_correct / total_questions
                    if total_questions > 0 else 0
                )
            },
            "results": results
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(partial_output, f, indent=4, ensure_ascii=False)

        time.sleep(2)

    accuracy = total_correct / total_questions if total_questions > 0 else 0

    final_output = {
        "summary": {
            "total_questions": total_questions,
            "total_correct": total_correct,
            "accuracy": accuracy
        },
        "results": results
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print("\nDone!")
    print(f"Total questions: {total_questions}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved results to {args.output}")
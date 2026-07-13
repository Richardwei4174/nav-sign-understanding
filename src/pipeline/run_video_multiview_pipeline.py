import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SRC_ROOT))

from src.pipeline.run_multiview_pipeline import run_multiview_from_detections
from src.understand.code.rpi_continuous_testing import GeminiDirectionQA


def load_selection(selection_json):
    with open(selection_json, "r", encoding="utf-8") as f:
        selection = json.load(f)

    return selection["selected"]


def build_detection_from_selection(selected):
    return [
        {
            "index": 0,
            "box": selected["bbox_xyxy"],
            "label": "navigation sign",
            "confidence": 1.0,
        }
    ]


def load_video_qa(qa_file, video_path):
    with open(qa_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    video_name = Path(video_path).name

    for item in qa_data:
        if item["videoPath"] == video_name:
            return item["questions"]

    raise ValueError(f"No QA found for video: {video_name}")


def run_video_multiview_pipeline(
    selection_json,
    video_path,
    output_root,
    qa_file,
    root,
    api_key_path,
    model_version,
    prompt_file,
):
    selected = load_selection(selection_json)

    selected_frame_path = PROJECT_ROOT / selected["frame_path"]
    selected_crop_path = PROJECT_ROOT / selected["crop_path"]
    detections = build_detection_from_selection(selected)

    question_items = load_video_qa(
        qa_file=qa_file,
        video_path=video_path,
    )

    qa = GeminiDirectionQA(
        root=root,
        api_key_path=api_key_path,
        model_version=model_version,
        prompt_file=prompt_file,
    )

    print("\n==============================")
    print("VIDEO MULTIVIEW PIPELINE")
    print("==============================")
    print(f"Video: {video_path}")
    print(f"Selected frame: {selected_frame_path}")
    print(f"Selected crop: {selected_crop_path}")
    print(f"Selected box: {selected['bbox_xyxy']}")
    print(f"Questions: {[q['question'] for q in question_items]}")

    image_output_name = Path(video_path).stem

    result = run_multiview_from_detections(
        image_path=selected_frame_path,
        output_root=output_root,
        qa=qa,
        question_items=question_items,
        detections=detections,
        crop_paths=[selected_crop_path],
        image_output_name=image_output_name,
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--selection-json",
        default="outputs/frame_selection/selection.json",
        help="Selection JSON produced by select_best_frame.py",
    )

    parser.add_argument(
        "--video",
        default="data/test_videos/IMG_3772.MOV",
        help="Original input video path",
    )

    parser.add_argument(
        "--output-root",
        default="outputs/video_multiview_pipeline",
        help="Output folder for video multiview results",
    )

    parser.add_argument(
        "--qa-file",
        default="src/understand/qa_test_set/rpi_video_test_set.json",
        help="Video QA JSON file",
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

    args = parser.parse_args()

    run_video_multiview_pipeline(
        selection_json=PROJECT_ROOT / args.selection_json,
        video_path=PROJECT_ROOT / args.video,
        output_root=PROJECT_ROOT / args.output_root,
        qa_file=PROJECT_ROOT / args.qa_file,
        root=args.root,
        api_key_path=args.api_key_path,
        model_version=args.model_version,
        prompt_file=args.prompt_file,
    )
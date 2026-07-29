"""Analyze completed Fixed Window results across recorded-video frame strides.

This command reads existing aggregate results and observation-cache metadata.
It never imports or invokes model, OCR, tracking, selection, or Gemini code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


STRIDES = (3, 6, 12)
EXPECTED_TARGETS = 78
POLICY = "fixed_window"
PER_STRIDE_FIELDS = (
    "selected_source_frame_index",
    "selected_source_timestamp_seconds",
    "gemini_prediction",
    "correct",
    "candidate_retrieved",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def cache_totals(
    cache_root: Path, stride: int, expected_video_stems: set[str]
) -> dict[str, Any]:
    values = {
        "processed_frames": [],
        "video_frames": [],
        "runtime_seconds": [],
    }
    found_stems = set()
    for path in sorted((cache_root / f"stride_{stride}").glob("*/observations.json")):
        if path.parent.name not in expected_video_stems:
            continue
        document = load_json(path)
        found_stems.add(path.parent.name)
        metadata = document.get("video_metadata", {})
        statistics = document.get("statistics", {})
        fields = {
            "processed_frames": metadata.get("processed_frames"),
            "video_frames": metadata.get("reported_source_frames"),
            "runtime_seconds": statistics.get("cache_generation_seconds"),
        }
        for name, raw_value in fields.items():
            value = finite_number(raw_value)
            if value is not None:
                values[name].append(value)

    expected_count = len(expected_video_stems)

    def complete_total(name: str) -> float | None:
        return sum(values[name]) if len(values[name]) == expected_count else None

    processed = complete_total("processed_frames")
    video_frames = complete_total("video_frames")
    runtime = complete_total("runtime_seconds")
    return {
        "cache_videos_expected": expected_count,
        "cache_videos_found": len(found_stems),
        "processed_frames_values_included": len(values["processed_frames"]),
        "total_processed_frames": int(processed) if processed is not None else None,
        "video_frames_values_included": len(values["video_frames"]),
        "total_video_frames": int(video_frames) if video_frames is not None else None,
        "percentage_frames_processed": (
            100.0 * processed / video_frames
            if processed is not None and video_frames
            else None
        ),
        "runtime_values_included": len(values["runtime_seconds"]),
        "total_runtime_seconds": runtime,
        "average_runtime_seconds_per_processed_frame": (
            runtime / processed if runtime is not None and processed else None
        ),
    }


def stride_summary(
    stride: int,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    cache_root: Path,
) -> dict[str, Any]:
    evaluated = [result for result in results if result.get("correct") is not None]
    correct = sum(result.get("correct") is True for result in evaluated)
    retrieved = sum(
        result.get("candidate_retrieved") is True for result in results
    )
    video_stems = {Path(result["video_filename"]).stem for result in results}
    return {
        "stride": stride,
        "total_targets": len(results),
        "evaluated_targets": len(evaluated),
        "correct": correct,
        "incorrect": len(evaluated) - correct,
        "micro_accuracy": correct / len(evaluated) if evaluated else None,
        "macro_accuracy_mean_per_video_accuracy": summary.get(
            "macro_accuracy_mean_per_video_accuracy"
        ),
        "candidate_retrieval_count": retrieved,
        "retrieval_rate": retrieved / len(results) if results else None,
        "gemini_call_count": sum(
            int(result.get("logical_gemini_calls", 0)) for result in results
        ),
        **cache_totals(cache_root, stride, video_stems),
    }


def all_same_non_null(values: list[Any]) -> bool:
    return bool(values) and all(value is not None for value in values) and all(
        value == values[0] for value in values[1:]
    )


def values_disagree(values: list[Any]) -> bool:
    present = [value for value in values if value is not None]
    return len(present) >= 2 and any(value != present[0] for value in present[1:])


def join_questions(
    results_by_stride: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    indexes: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
    for stride, results in results_by_stride.items():
        index = {}
        for result in results:
            key = (result["video_filename"], result["question_id"])
            if key in index:
                raise ValueError(f"Duplicate stride {stride} result key: {key}")
            index[key] = result
        indexes[stride] = index

    reference = set(indexes[STRIDES[0]])
    for stride in STRIDES[1:]:
        keys = set(indexes[stride])
        if keys != reference:
            raise ValueError(
                f"Join keys differ for stride {stride}; "
                f"missing={sorted(reference - keys)}, extra={sorted(keys - reference)}"
            )
    if len(reference) != EXPECTED_TARGETS:
        raise ValueError(
            f"Expected exactly {EXPECTED_TARGETS} joined rows; found {len(reference)}"
        )

    rows = []
    for key in sorted(reference, key=lambda item: (item[0].lower(), item[1])):
        records = {stride: indexes[stride][key] for stride in STRIDES}
        identity = records[STRIDES[0]]
        for stride in STRIDES[1:]:
            for field in ("question", "destination", "expected_direction"):
                if records[stride].get(field) != identity.get(field):
                    raise ValueError(f"{field} differs across strides for {key}")
        row: dict[str, Any] = {
            "video_filename": key[0],
            "question_id": key[1],
            "question": identity.get("question"),
            "destination": identity.get("destination"),
            "expected_direction": identity.get("expected_direction"),
        }
        for stride in STRIDES:
            for field in PER_STRIDE_FIELDS:
                row[f"stride_{stride}_{field}"] = records[stride].get(field)

        predictions = [records[stride].get("gemini_prediction") for stride in STRIDES]
        correctness = [records[stride].get("correct") for stride in STRIDES]
        row.update(
            {
                "all_same_prediction": all_same_non_null(predictions),
                "all_same_correctness": all_same_non_null(correctness),
                "correctness_disagrees": values_disagree(correctness),
                "stride_3_only_correct": correctness == [True, False, False],
                "stride_6_only_correct": correctness == [False, True, False],
                "stride_12_only_correct": correctness == [False, False, True],
                "stride_3_correct_stride_12_wrong": (
                    correctness[0] is True and correctness[2] is False
                ),
                "stride_12_correct_stride_3_wrong": (
                    correctness[2] is True and correctness[0] is False
                ),
                "any_retrieval_failure": any(
                    records[stride].get("candidate_retrieved") is not True
                    for stride in STRIDES
                ),
            }
        )
        rows.append(row)
    return rows


def console_summary(
    summaries: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> None:
    for summary in summaries:
        print(
            f"Stride {summary['stride']}: accuracy={summary['micro_accuracy']:.4f}, "
            f"retrieval_rate={summary['retrieval_rate']:.4f}"
        )
    prediction_differences = sum(
        values_disagree(
            [row[f"stride_{stride}_gemini_prediction"] for stride in STRIDES]
        )
        for row in rows
    )
    changed = [row for row in rows if row["correctness_disagrees"]]
    print(f"Questions with predictions differing across strides: {prediction_differences}")
    print(f"Questions changing correctness across strides: {len(changed)}")
    for row in changed:
        states = ", ".join(
            f"stride_{stride}={row[f'stride_{stride}_correct']}" for stride in STRIDES
        )
        print(
            f"  {row['video_filename']} {row['question_id']}: "
            f"{row['question']} ({states})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default="outputs/video_temporal_ablation/full/runs",
    )
    parser.add_argument(
        "--cache-root",
        default="outputs/video_temporal_ablation/full/cache",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/video_temporal_ablation/full/analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    cache_root = Path(args.cache_root)
    output_dir = Path(args.output_dir)
    results_by_stride: dict[int, list[dict[str, Any]]] = {}
    aggregate_summaries: dict[int, dict[str, Any]] = {}

    # Validate every input and the join before creating or replacing outputs.
    for stride in STRIDES:
        policy_dir = runs_root / f"stride_{stride}" / POLICY
        aggregate = load_json(policy_dir / "all_results.json")
        results = aggregate.get("results")
        if not isinstance(results, list) or len(results) != EXPECTED_TARGETS:
            count = len(results) if isinstance(results, list) else "invalid"
            raise ValueError(
                f"{policy_dir / 'all_results.json'} must contain "
                f"{EXPECTED_TARGETS} targets; found {count}"
            )
        results_by_stride[stride] = results
        aggregate_summaries[stride] = load_json(policy_dir / "summary.json")

    question_rows = join_questions(results_by_stride)
    summaries = [
        stride_summary(
            stride,
            results_by_stride[stride],
            aggregate_summaries[stride],
            cache_root,
        )
        for stride in STRIDES
    ]
    summary_document = {
        "policy": POLICY,
        "runs_root": str(runs_root),
        "runtime_definition": "observation cache generation runtime",
        "cache_metric_availability": (
            "A cache-derived total is null unless all expected video caches "
            "contain that numeric field."
        ),
        "strides": summaries,
    }
    question_document = {
        "policy": POLICY,
        "runs_root": str(runs_root),
        "join_key": ["video_filename", "question_id"],
        "row_count": len(question_rows),
        "results": question_rows,
    }

    write_json(output_dir / "stride_comparison.json", summary_document)
    write_csv(output_dir / "stride_comparison.csv", summaries)
    write_json(
        output_dir / "per_question_stride_comparison.json", question_document
    )
    write_csv(output_dir / "per_question_stride_comparison.csv", question_rows)
    print(f"Validated {EXPECTED_TARGETS} targets for each stride and join.")
    console_summary(summaries, question_rows)
    print(f"Wrote stride analysis outputs to: {output_dir}")


if __name__ == "__main__":
    main()

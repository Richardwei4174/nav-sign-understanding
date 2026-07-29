"""Analyze completed recorded-video temporal-policy aggregate results.

This command reads existing ``all_results.json`` and ``summary.json`` files.
It does not import or invoke detection, OCR, tracking, selection, or Gemini code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


POLICIES = (
    "first_match",
    "fixed_window",
    "last_seen",
    "track_lifetime_best_view",
)
EXPECTED_TARGETS = 78
TIMING_FIELDS = {
    "selection_delay": "selection_delay_seconds",
    "selection_timestamp": "selected_source_timestamp_seconds",
}
POLICY_RESULT_FIELDS = (
    "selected_source_frame_index",
    "selected_source_timestamp_seconds",
    "selection_delay_seconds",
    "selected_crop_path",
    "selected_frame_path",
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


def numeric_values(results: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for result in results:
        value = result.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                values.append(number)
    return values


def percentile_linear(values: list[float], percentile: float) -> float | None:
    """Return a NumPy-style linearly interpolated percentile."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def policy_comparison(
    policy: str,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    total = len(results)
    evaluated = [item for item in results if item.get("correct") is not None]
    correct = sum(item.get("correct") is True for item in evaluated)
    retrieval_count = sum(item.get("candidate_retrieved") is True for item in results)
    delays = numeric_values(results, TIMING_FIELDS["selection_delay"])
    timestamps = numeric_values(results, TIMING_FIELDS["selection_timestamp"])
    return {
        "policy": policy,
        "total_targets": total,
        "evaluated_targets": len(evaluated),
        "correct": correct,
        "incorrect": len(evaluated) - correct,
        "micro_accuracy": correct / len(evaluated) if evaluated else None,
        "macro_accuracy_mean_per_video_accuracy": summary.get(
            "macro_accuracy_mean_per_video_accuracy"
        ),
        "candidate_retrieval_count": retrieval_count,
        "candidate_retrieval_rate": retrieval_count / total if total else None,
        "gemini_call_count": sum(
            int(item.get("logical_gemini_calls", 0)) for item in results
        ),
        "mean_selection_delay_seconds": statistics.fmean(delays) if delays else None,
        "median_selection_delay_seconds": statistics.median(delays) if delays else None,
        "p90_selection_delay_seconds": percentile_linear(delays, 0.90),
        "maximum_selection_delay_seconds": max(delays) if delays else None,
        "selection_delay_values_included": len(delays),
        "mean_selection_timestamp_seconds": (
            statistics.fmean(timestamps) if timestamps else None
        ),
        "median_selection_timestamp_seconds": (
            statistics.median(timestamps) if timestamps else None
        ),
        "selection_timestamp_values_included": len(timestamps),
    }


def all_same_non_null(values: list[Any]) -> bool:
    return bool(values) and all(value is not None for value in values) and all(
        value == values[0] for value in values[1:]
    )


def values_disagree(values: list[Any]) -> bool:
    present = [value for value in values if value is not None]
    return len(present) >= 2 and any(value != present[0] for value in present[1:])


def joined_question_rows(
    results_by_policy: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    indexed: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for policy, results in results_by_policy.items():
        policy_index = {}
        for result in results:
            key = (result["video_filename"], result["question_id"])
            if key in policy_index:
                raise ValueError(f"Duplicate {policy} result key: {key}")
            policy_index[key] = result
        indexed[policy] = policy_index

    key_sets = {policy: set(index) for policy, index in indexed.items()}
    reference = key_sets[POLICIES[0]]
    for policy in POLICIES[1:]:
        if key_sets[policy] != reference:
            missing = sorted(reference - key_sets[policy])
            extra = sorted(key_sets[policy] - reference)
            raise ValueError(
                f"Join keys differ for {policy}; missing={missing}, extra={extra}"
            )
    if len(reference) != EXPECTED_TARGETS:
        raise ValueError(
            f"Expected exactly {EXPECTED_TARGETS} joined rows; found {len(reference)}"
        )

    rows = []
    for key in sorted(reference, key=lambda item: (item[0].lower(), item[1])):
        records = {policy: indexed[policy][key] for policy in POLICIES}
        identity = records[POLICIES[0]]
        for policy in POLICIES[1:]:
            for field in ("question", "destination", "expected_direction"):
                if records[policy].get(field) != identity.get(field):
                    raise ValueError(f"{field} differs across policies for {key}")

        row: dict[str, Any] = {
            "video_filename": key[0],
            "question_id": key[1],
            "question": identity.get("question"),
            "destination": identity.get("destination"),
            "expected_direction": identity.get("expected_direction"),
        }
        for policy in POLICIES:
            for field in POLICY_RESULT_FIELDS:
                row[f"{policy}_{field}"] = records[policy].get(field)

        frames = [records[p].get("selected_source_frame_index") for p in POLICIES]
        predictions = [records[p].get("gemini_prediction") for p in POLICIES]
        correctness = [records[p].get("correct") for p in POLICIES]
        original_correctness = correctness[:3]
        fixed_correct = records["fixed_window"].get("correct")
        last_correct = records["last_seen"].get("correct")
        track_correct = records["track_lifetime_best_view"].get("correct")
        row.update(
            {
                "all_same_selected_frame": all_same_non_null(frames),
                "fixed_window_last_seen_same_frame": all_same_non_null(
                    [frames[1], frames[2]]
                ),
                "all_same_prediction": all_same_non_null(predictions),
                "fixed_window_last_seen_same_prediction": all_same_non_null(
                    [predictions[1], predictions[2]]
                ),
                "correctness_disagrees": values_disagree(correctness),
                "predictions_disagree": values_disagree(predictions),
                "first_match_only_correct": (
                    original_correctness == [True, False, False]
                ),
                "fixed_window_only_correct": (
                    original_correctness == [False, True, False]
                ),
                "last_seen_only_correct": (
                    original_correctness == [False, False, True]
                ),
                "track_lifetime_best_view_only_correct": (
                    correctness == [False, False, False, True]
                ),
                "fixed_window_correct_last_seen_wrong": (
                    fixed_correct is True and last_correct is False
                ),
                "last_seen_correct_fixed_window_wrong": (
                    last_correct is True and fixed_correct is False
                ),
                "track_lifetime_correct_last_seen_wrong": (
                    track_correct is True and last_correct is False
                ),
                "last_seen_correct_track_lifetime_wrong": (
                    last_correct is True and track_correct is False
                ),
                "all_wrong": correctness == [False, False, False, False],
                "any_retrieval_failure": any(
                    records[p].get("candidate_retrieved") is not True
                    for p in POLICIES
                ),
            }
        )
        rows.append(row)
    return rows


def pairwise_counts(
    rows: list[dict[str, Any]], first: str, second: str
) -> dict[str, int]:
    first_frame = f"{first}_selected_source_frame_index"
    second_frame = f"{second}_selected_source_frame_index"
    first_prediction = f"{first}_gemini_prediction"
    second_prediction = f"{second}_gemini_prediction"
    first_correct = f"{first}_correct"
    second_correct = f"{second}_correct"
    return {
        "same_selected_frame": sum(
            all_same_non_null([row[first_frame], row[second_frame]]) for row in rows
        ),
        "prediction_differences": sum(
            values_disagree([row[first_prediction], row[second_prediction]])
            for row in rows
        ),
        "correctness_differences": sum(
            values_disagree([row[first_correct], row[second_correct]]) for row in rows
        ),
        f"{first}_correct_{second}_wrong": sum(
            row[first_correct] is True and row[second_correct] is False
            for row in rows
        ),
        f"{second}_correct_{first}_wrong": sum(
            row[second_correct] is True and row[first_correct] is False
            for row in rows
        ),
    }


def case_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_filename": row["video_filename"],
        "question_id": row["question_id"],
        "question": row["question"],
        "expected_direction": row["expected_direction"],
        "last_seen_prediction": row["last_seen_gemini_prediction"],
        "track_lifetime_best_view_prediction": row[
            "track_lifetime_best_view_gemini_prediction"
        ],
        "last_seen_selected_source_frame_index": row[
            "last_seen_selected_source_frame_index"
        ],
        "track_lifetime_best_view_selected_source_frame_index": row[
            "track_lifetime_best_view_selected_source_frame_index"
        ],
    }


def counted_noun(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def build_four_policy_summary(
    rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    track = "track_lifetime_best_view"
    pairwise = {
        other: pairwise_counts(rows, track, other)
        for other in ("first_match", "fixed_window", "last_seen")
    }
    track_fixed = pairwise["fixed_window"]
    track_last = pairwise["last_seen"]
    corrected_last = [
        case_record(row)
        for row in rows
        if row["track_lifetime_correct_last_seen_wrong"]
    ]
    harmed_last = [
        case_record(row)
        for row in rows
        if row["last_seen_correct_track_lifetime_wrong"]
    ]
    accuracy_table = [
        {
            "policy": item["policy"],
            "correct": item["correct"],
            "total": item["evaluated_targets"],
            "accuracy": item["micro_accuracy"],
            "retrieved": item["candidate_retrieval_count"],
        }
        for item in comparison_rows
    ]
    labels = {
        "first_match": "First Match",
        "fixed_window": "Fixed Window",
        "last_seen": "Last Seen",
        track: "Track-Lifetime Best View",
    }
    accuracy_sentence = "; ".join(
        f"{labels[item['policy']]} {item['correct']}/{item['total']} "
        f"({100 * item['accuracy']:.2f}%)"
        for item in accuracy_table
    )
    return {
        "scope": "four-policy recorded-video temporal comparison",
        "question_count": len(rows),
        "accuracy_table": accuracy_table,
        "track_lifetime_best_view_pairwise": pairwise,
        "track_lifetime_corrected_last_seen_errors": corrected_last,
        "last_seen_correct_track_lifetime_incorrect": harmed_last,
        "paper_ready_summary": {
            "accuracy": accuracy_sentence + ".",
            "frame_selection": (
                "Track-Lifetime Best View selected the same source frame as "
                f"First Match for {pairwise['first_match']['same_selected_frame']} "
                f"of {len(rows)} questions, Fixed Window for "
                f"{track_fixed['same_selected_frame']} of {len(rows)}, and Last Seen "
                f"for {track_last['same_selected_frame']} of {len(rows)}."
            ),
            "prediction_and_correctness": (
                "Relative to Fixed Window, Track-Lifetime Best View differed on "
                f"{counted_noun(track_fixed['prediction_differences'], 'prediction')} "
                f"and {counted_noun(track_fixed['correctness_differences'], 'correctness outcome')}; "
                "relative to Last Seen, it differed on "
                f"{counted_noun(track_last['prediction_differences'], 'prediction')} "
                f"and {counted_noun(track_last['correctness_differences'], 'correctness outcome')}."
            ),
            "last_seen_directional_changes": (
                "Track-Lifetime Best View corrected "
                f"{counted_noun(len(corrected_last), 'Last Seen error')} and introduced "
                f"{counted_noun(len(harmed_last), 'error')} on questions Last Seen answered correctly."
            ),
        },
    }


def console_summary(
    rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    four_policy_summary: dict[str, Any],
) -> None:
    prediction_differences = sum(
        not row["fixed_window_last_seen_same_prediction"]
        and row["fixed_window_gemini_prediction"] is not None
        and row["last_seen_gemini_prediction"] is not None
        for row in rows
    )
    correctness_differences = sum(
        row["fixed_window_correct_last_seen_wrong"]
        or row["last_seen_correct_fixed_window_wrong"]
        for row in rows
    )
    print("\nAccuracy across all four policies:")
    print(f"{'Policy':<32} {'Correct':>9} {'Accuracy':>10} {'Retrieved':>10}")
    for item in comparison_rows:
        print(
            f"{item['policy']:<32} "
            f"{item['correct']:>3}/{item['evaluated_targets']:<5} "
            f"{100 * item['micro_accuracy']:>9.2f}% "
            f"{item['candidate_retrieval_count']:>4}/{item['total_targets']:<5}"
        )
    track_pairs = four_policy_summary["track_lifetime_best_view_pairwise"]
    print("\nTrack-Lifetime Best View selected the same frame as:")
    for policy in ("first_match", "fixed_window", "last_seen"):
        print(f"  {policy}: {track_pairs[policy]['same_selected_frame']}")
    for policy in ("fixed_window", "last_seen"):
        counts = track_pairs[policy]
        print(
            f"Track-Lifetime Best View vs {policy}: "
            f"prediction differences={counts['prediction_differences']}, "
            f"correctness differences={counts['correctness_differences']}"
        )
    print(
        "Same selected frame across all four policies: "
        f"{sum(row['all_same_selected_frame'] for row in rows)}"
    )
    print(
        "Fixed Window and Last Seen selected the same frame: "
        f"{sum(row['fixed_window_last_seen_same_frame'] for row in rows)}"
    )
    print(
        "Fixed Window and Last Seen predictions differed: "
        f"{prediction_differences}"
    )
    print(
        "Fixed Window and Last Seen correctness differed: "
        f"{correctness_differences}"
    )
    print("Fixed Window right, Last Seen wrong:")
    for row in rows:
        if row["fixed_window_correct_last_seen_wrong"]:
            print(f"  {row['video_filename']} {row['question_id']}: {row['question']}")
    print("Last Seen right, Fixed Window wrong:")
    for row in rows:
        if row["last_seen_correct_fixed_window_wrong"]:
            print(f"  {row['video_filename']} {row['question_id']}: {row['question']}")
    print("Track-Lifetime Best View corrected a Last Seen error:")
    for item in four_policy_summary["track_lifetime_corrected_last_seen_errors"]:
        print(
            f"  {item['video_filename']} {item['question_id']}: {item['question']} "
            f"(expected={item['expected_direction']}, "
            f"last_seen={item['last_seen_prediction']}, "
            f"track_lifetime={item['track_lifetime_best_view_prediction']})"
        )
    print("Last Seen correct, Track-Lifetime Best View incorrect:")
    harmed_cases = four_policy_summary["last_seen_correct_track_lifetime_incorrect"]
    if not harmed_cases:
        print("  (none)")
    for item in harmed_cases:
        print(
            f"  {item['video_filename']} {item['question_id']}: {item['question']} "
            f"(expected={item['expected_direction']}, "
            f"last_seen={item['last_seen_prediction']}, "
            f"track_lifetime={item['track_lifetime_best_view_prediction']})"
        )
    print("\nPaper-ready four-policy summary:")
    for sentence in four_policy_summary["paper_ready_summary"].values():
        print(f"  {sentence}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="outputs/video_temporal_ablation/full/runs/stride_6",
        help="Directory containing all four temporal-policy result folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/video_temporal_ablation/full/analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    results_by_policy: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}

    # Complete all validation before creating the output directory or files.
    for policy in POLICIES:
        all_results_path = run_dir / policy / "all_results.json"
        summary_path = run_dir / policy / "summary.json"
        aggregate = load_json(all_results_path)
        results = aggregate.get("results")
        if not isinstance(results, list) or len(results) != EXPECTED_TARGETS:
            count = len(results) if isinstance(results, list) else "invalid"
            raise ValueError(
                f"{all_results_path} must contain {EXPECTED_TARGETS} targets; "
                f"found {count}"
            )
        results_by_policy[policy] = results
        summaries[policy] = load_json(summary_path)

    question_rows = joined_question_rows(results_by_policy)
    comparison_rows = [
        policy_comparison(policy, results_by_policy[policy], summaries[policy])
        for policy in POLICIES
    ]
    comparison_document = {
        "run_directory": str(run_dir),
        "timing_null_handling": (
            "Null, non-numeric, and non-finite timing values are excluded "
            "independently for each statistic."
        ),
        "p90_method": "linear interpolation at (n - 1) * 0.90",
        "policies": comparison_rows,
    }
    four_policy_summary = build_four_policy_summary(
        question_rows, comparison_rows
    )
    question_document = {
        "run_directory": str(run_dir),
        "join_key": ["video_filename", "question_id"],
        "row_count": len(question_rows),
        "results": question_rows,
    }

    write_json(output_dir / "temporal_policy_comparison.json", comparison_document)
    write_csv(output_dir / "temporal_policy_comparison.csv", comparison_rows)
    write_json(
        output_dir / "four_policy_comparison_summary.json",
        four_policy_summary,
    )
    write_json(
        output_dir / "per_question_policy_comparison.json", question_document
    )
    write_csv(output_dir / "per_question_policy_comparison.csv", question_rows)
    print(f"Validated {EXPECTED_TARGETS} targets for each policy and join.")
    console_summary(question_rows, comparison_rows, four_policy_summary)
    print(f"Wrote analysis outputs to: {output_dir}")


if __name__ == "__main__":
    main()

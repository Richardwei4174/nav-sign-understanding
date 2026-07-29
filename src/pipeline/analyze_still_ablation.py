"""Join and compare completed still-image ablation results per question."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


BASELINE = "original_only"
CONDITIONS = (
    "original_only",
    "raw_crop_multiview",
    "rectified_crops_only",
    "full_multiview",
    "raw_crops_only",
    "original_rectified_no_annotation",
    "original_raw_no_annotation",
)
EXPECTED_QUESTIONS = 1100


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    temporary.replace(path)


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {key: csv_value(value) for key, value in row.items()} for row in rows
        )
    temporary.replace(path)


def index_results(
    condition: str, document: dict[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    index = {}
    for image in document.get("results", []):
        for question in image.get("questions", []):
            key = (image["imagePath"], question["question"])
            if key in index:
                raise ValueError(f"Duplicate key in {condition}: {key}")
            index[key] = question
    if len(index) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"{condition} must contain {EXPECTED_QUESTIONS} unique questions; "
            f"found {len(index)}"
        )
    return index


def expected_class(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def mcnemar_exact_p_value(baseline_wrong_other_correct: int, baseline_correct_other_wrong: int) -> float:
    """Two-sided exact McNemar p-value using Binomial(n, 0.5)."""
    discordant = baseline_wrong_other_correct + baseline_correct_other_wrong
    if discordant == 0:
        return 1.0
    tail = min(baseline_wrong_other_correct, baseline_correct_other_wrong)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def paired_summary(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    baseline_correct = f"{BASELINE}_correct"
    other_correct = f"{condition}_correct"
    baseline_prediction = f"{BASELINE}_predicted"
    other_prediction = f"{condition}_predicted"
    wrong_to_correct = sum(
        row[baseline_correct] is False and row[other_correct] is True for row in rows
    )
    correct_to_wrong = sum(
        row[baseline_correct] is True and row[other_correct] is False for row in rows
    )
    both_correct = sum(
        row[baseline_correct] is True and row[other_correct] is True for row in rows
    )
    both_wrong = sum(
        row[baseline_correct] is False and row[other_correct] is False for row in rows
    )
    changed_same_correctness = sum(
        row[baseline_prediction] != row[other_prediction]
        and row[baseline_correct] == row[other_correct]
        for row in rows
    )
    return {
        "condition": condition,
        "original_wrong_condition_correct": wrong_to_correct,
        "original_correct_condition_wrong": correct_to_wrong,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "prediction_changed_correctness_same": changed_same_correctness,
        "discordant_total": wrong_to_correct + correct_to_wrong,
        "mcnemar_exact_two_sided_p_value": mcnemar_exact_p_value(
            wrong_to_correct, correct_to_wrong
        ),
    }


def per_direction(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classes = sorted({row["expected_class"] for row in rows})
    output = []
    for direction in classes:
        selected = [row for row in rows if row["expected_class"] == direction]
        record: dict[str, Any] = {
            "ground_truth_class": direction,
            "question_count": len(selected),
        }
        for condition in CONDITIONS:
            correct = sum(row[f"{condition}_correct"] is True for row in selected)
            record[f"{condition}_correct"] = correct
            record[f"{condition}_accuracy"] = correct / len(selected)
        output.append(record)
    return output


def join_results(
    indexes: dict[str, dict[tuple[str, str], dict[str, Any]]]
) -> list[dict[str, Any]]:
    reference = set(indexes[BASELINE])
    for condition in CONDITIONS[1:]:
        keys = set(indexes[condition])
        if keys != reference:
            raise ValueError(
                f"Question keys differ for {condition}; "
                f"missing={sorted(reference - keys)}, extra={sorted(keys - reference)}"
            )
    if len(reference) != EXPECTED_QUESTIONS:
        raise ValueError(f"Joined table must contain {EXPECTED_QUESTIONS} rows")

    rows = []
    for image_name, question_text in sorted(
        reference, key=lambda key: (key[0].lower(), key[1])
    ):
        records = {
            condition: indexes[condition][(image_name, question_text)]
            for condition in CONDITIONS
        }
        expected = records[BASELINE].get("expected")
        for condition in CONDITIONS[1:]:
            if records[condition].get("expected") != expected:
                raise ValueError(
                    f"Expected answer differs for {image_name}, {question_text}, {condition}"
                )
        row: dict[str, Any] = {
            "image_filename": image_name,
            "question": question_text,
            "expected": expected,
            "expected_class": expected_class(expected),
        }
        for condition in CONDITIONS:
            row[f"{condition}_predicted"] = records[condition].get("predicted")
            row[f"{condition}_correct"] = records[condition].get("correct")
        full_correct = row["full_multiview_correct"]
        original_correct = row["original_only_correct"]
        row.update(
            {
                "original_wrong_full_correct": (
                    original_correct is False and full_correct is True
                ),
                "original_correct_full_wrong": (
                    original_correct is True and full_correct is False
                ),
                "original_full_both_correct": (
                    original_correct is True and full_correct is True
                ),
                "original_full_both_wrong": (
                    original_correct is False and full_correct is False
                ),
                "original_full_prediction_changed": (
                    row["original_only_predicted"] != row["full_multiview_predicted"]
                ),
                "original_full_prediction_changed_correctness_same": (
                    row["original_only_predicted"] != row["full_multiview_predicted"]
                    and original_correct == full_correct
                ),
            }
        )
        rows.append(row)
    return rows


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Validated and joined {summary['joined_question_count']} questions.")
    print("Paired comparisons against Original only:")
    for item in summary["paired_comparisons"]:
        print(
            f"  {item['condition']}: wrong->correct="
            f"{item['original_wrong_condition_correct']}, correct->wrong="
            f"{item['original_correct_condition_wrong']}, "
            f"exact_p={item['mcnemar_exact_two_sided_p_value']:.12g}"
        )
    print(f"Full-multiview failures: {len(summary['full_multiview_failures'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", default="outputs/still_ablation/full/results"
    )
    parser.add_argument(
        "--output-dir", default="outputs/still_ablation/full/analysis"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    source_files = {
        condition: results_dir / f"{condition}.json" for condition in CONDITIONS
    }
    documents = {condition: load_json(path) for condition, path in source_files.items()}
    indexes = {
        condition: index_results(condition, documents[condition])
        for condition in CONDITIONS
    }
    rows = join_results(indexes)

    paired = [paired_summary(rows, condition) for condition in CONDITIONS[1:]]
    direction_results = per_direction(rows)
    failures = [
        {
            "image_filename": row["image_filename"],
            "question": row["question"],
            "expected": row["expected"],
            "full_multiview_predicted": row["full_multiview_predicted"],
            "original_only_predicted": row["original_only_predicted"],
            "original_only_correct": row["original_only_correct"],
        }
        for row in rows
        if row["full_multiview_correct"] is False
    ]
    fixed = [row for row in rows if row["original_wrong_full_correct"]]
    introduced = [row for row in rows if row["original_correct_full_wrong"]]
    summary = {
        "schema_version": 1,
        "baseline": BASELINE,
        "source_files": {key: str(value) for key, value in source_files.items()},
        "matching_key": ["imagePath", "question"],
        "scoring_logic": (
            "Uses each source result's persisted correct boolean. The experiment "
            "scores scalar answers by equality, expected lists against predicted "
            "lists by set equality, and scalar/list pairs by membership."
        ),
        "mcnemar_method": (
            "Two-sided exact binomial test on discordant pairs under p=0.5; "
            "p=min(1, 2*P[X<=min(b,c)]) for X~Binomial(b+c,0.5)."
        ),
        "joined_question_count": len(rows),
        "paired_comparisons": paired,
        "per_direction": direction_results,
        "full_multiview_failures": failures,
        "representative_manual_inspection_candidates": {
            "full_multiview_fixed_original_first_5": [
                {
                    "image_filename": row["image_filename"],
                    "question": row["question"],
                    "expected": row["expected"],
                    "original_only_predicted": row["original_only_predicted"],
                    "full_multiview_predicted": row["full_multiview_predicted"],
                }
                for row in fixed[:5]
            ],
            "full_multiview_introduced_error_first_5": [
                {
                    "image_filename": row["image_filename"],
                    "question": row["question"],
                    "expected": row["expected"],
                    "original_only_predicted": row["original_only_predicted"],
                    "full_multiview_predicted": row["full_multiview_predicted"],
                }
                for row in introduced[:5]
            ],
        },
    }

    # All input and join validation completes before any output is replaced.
    write_json(output_dir / "still_condition_comparison_summary.json", summary)
    write_json(
        output_dir / "per_question_still_condition_comparison.json",
        {
            "schema_version": 1,
            "source_files": summary["source_files"],
            "matching_key": summary["matching_key"],
            "row_count": len(rows),
            "results": rows,
        },
    )
    write_csv(output_dir / "per_question_still_condition_comparison.csv", rows)
    print_summary(summary)
    print(f"Wrote still-image analysis to: {output_dir}")


if __name__ == "__main__":
    main()

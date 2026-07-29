"""Summarize manually reviewed video-temporal-ablation retrieval results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TRUE_VALUES = {"true", "1", "yes", "y"}
FALSE_VALUES = {"false", "0", "no", "n"}


def parse_bool(value: Any, field: str, *, allow_blank: bool = False) -> bool | None:
    normalized = str(value).strip().lower()
    if allow_blank and not normalized:
        return None
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean for {field}: {value!r}")


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Review CSV contains no result rows.")

    counts = {
        "total": len(rows),
        "present": 0,
        "absent": 0,
        "retrieved": 0,
        "correct_physical_sign": 0,
        "present_correctly_retrieved": 0,
        "absent_retrieved": 0,
        "absent_not_retrieved": 0,
        "grounded_present_with_direction": 0,
        "grounded_present_direction_correct": 0,
        "automatic_correct": 0,
        "grounded_end_to_end_success": 0,
    }

    for row_number, row in enumerate(rows, start=2):
        present = parse_bool(row.get("target_present", ""), "target_present")
        retrieved = parse_bool(
            row.get("candidate_retrieved", ""), "candidate_retrieved"
        )
        automatic_correct = parse_bool(
            row.get("correct", ""), "correct", allow_blank=True
        )
        if automatic_correct:
            counts["automatic_correct"] += 1

        if present:
            counts["present"] += 1
        else:
            counts["absent"] += 1
            if retrieved:
                counts["absent_retrieved"] += 1
            else:
                counts["absent_not_retrieved"] += 1
                counts["grounded_end_to_end_success"] += 1

        if not retrieved:
            continue
        counts["retrieved"] += 1
        physical = parse_bool(
            row.get("selected_correct_physical_sign", ""),
            f"selected_correct_physical_sign at row {row_number}",
        )
        # Visibility is required for retrieved rows even though it is diagnostic.
        parse_bool(
            row.get("destination_visible_in_selected_crop", ""),
            f"destination_visible_in_selected_crop at row {row_number}",
        )
        if physical:
            counts["correct_physical_sign"] += 1
            if present:
                counts["present_correctly_retrieved"] += 1
                if automatic_correct is not None:
                    counts["grounded_present_with_direction"] += 1
                    if automatic_correct:
                        counts["grounded_present_direction_correct"] += 1
                        counts["grounded_end_to_end_success"] += 1

    metrics = {
        "retrieval_precision": safe_rate(
            counts["correct_physical_sign"], counts["retrieved"]
        ),
        "retrieval_recall": safe_rate(
            counts["present_correctly_retrieved"], counts["present"]
        ),
        "false_positive_rate": safe_rate(
            counts["absent_retrieved"], counts["absent"]
        ),
        "true_negative_rate": safe_rate(
            counts["absent_not_retrieved"], counts["absent"]
        ),
        "direction_accuracy_on_correctly_grounded_present_targets": safe_rate(
            counts["grounded_present_direction_correct"],
            counts["grounded_present_with_direction"],
        ),
        "automatic_end_to_end_accuracy": safe_rate(
            counts["automatic_correct"], counts["total"]
        ),
        "grounded_end_to_end_success_rate": safe_rate(
            counts["grounded_end_to_end_success"], counts["total"]
        ),
    }
    return {"counts": counts, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    review_path = Path(args.review_csv)
    output_path = Path(args.output_json)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output_path}")
    with review_path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    output = {
        "review_csv": str(review_path),
        **summarize(rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False)
    temporary.replace(output_path)
    print(f"Wrote review summary: {output_path}")


if __name__ == "__main__":
    main()

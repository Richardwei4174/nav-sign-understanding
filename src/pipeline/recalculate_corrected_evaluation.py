"""Rescore frozen TRB predictions using adjudicated reference semantics.

This command never invokes inference and never modifies frozen annotations or
result JSONs. It writes versioned derived summaries under a separate directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "experiments/evaluation/corrected_reference_manifest_v2.json"
STILL_RESULTS = ROOT / "outputs/still_ablation/full/results"
VIDEO_RUNS = ROOT / "outputs/video_temporal_ablation/full/runs"
DEFAULT_OUTPUT = ROOT / "outputs/corrected_evaluation/v2"

STILL_CONDITIONS = (
    "original_only",
    "raw_crops_only",
    "rectified_crops_only",
    "original_raw_no_annotation",
    "original_rectified_no_annotation",
    "raw_crop_multiview",
    "full_multiview",
)
VIDEO_CONDITIONS = {
    "stride_6_first_match": VIDEO_RUNS / "stride_6/first_match/all_results.json",
    "stride_6_fixed_window": VIDEO_RUNS / "stride_6/fixed_window/all_results.json",
    "stride_6_last_seen": VIDEO_RUNS / "stride_6/last_seen/all_results.json",
    "stride_6_track_lifetime_best_view": VIDEO_RUNS / "stride_6/track_lifetime_best_view/all_results.json",
    "stride_3_fixed_window": VIDEO_RUNS / "stride_3/fixed_window/all_results.json",
    "stride_12_fixed_window": VIDEO_RUNS / "stride_12/fixed_window/all_results.json",
}


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labels(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    return {str(value)}


def score(prediction: Any, reference: Any, reference_type: str) -> bool:
    predicted = labels(prediction)
    expected = labels(reference)
    if reference_type == "scalar":
        if len(expected) != 1:
            raise ValueError(f"Scalar reference must contain one label: {reference!r}")
        return predicted == expected
    if reference_type == "alternatives":
        return bool(predicted) and predicted <= expected
    if reference_type == "joint_required":
        return predicted == expected
    raise ValueError(f"Unknown reference type: {reference_type}")


def exact_mcnemar(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = min(b, c)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant
    return min(1.0, 2.0 * probability)


def reference_index(manifest: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in manifest["overrides"]:
        index[(item["dataset"], item["identifier"], item["question"])] = item
    for item in manifest["confirmed_scalar_adjudications"]:
        key = (item["dataset"], item["identifier"], item["question"])
        index[key] = {**item, "reference_type": "scalar"}
    return index


def corrected_reference(
    index: dict[tuple[str, str, str], dict[str, Any]],
    dataset: str,
    identifier: str,
    question: str,
    frozen_reference: Any,
) -> tuple[Any, str, bool]:
    item = index.get((dataset, identifier, question))
    if item is None:
        return frozen_reference, "scalar", False
    return item["reference"], item["reference_type"], True


def paired(rows: list[dict[str, Any]], first: str, second: str) -> dict[str, Any]:
    both_correct = first_only = second_only = both_wrong = 0
    prediction_differences = 0
    for row in rows:
        a = row["conditions"][first]
        b = row["conditions"][second]
        prediction_differences += a["predicted"] != b["predicted"]
        if a["correct"] and b["correct"]:
            both_correct += 1
        elif a["correct"]:
            first_only += 1
        elif b["correct"]:
            second_only += 1
        else:
            both_wrong += 1
    return {
        "first": first,
        "second": second,
        "both_correct": both_correct,
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "both_wrong": both_wrong,
        "correctness_differences": first_only + second_only,
        "prediction_differences": prediction_differences,
        "mcnemar_exact_two_sided_p_value": exact_mcnemar(first_only, second_only),
    }


def still_analysis(index: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    documents = {name: load(STILL_RESULTS / f"{name}.json") for name in STILL_CONDITIONS}
    sources = {
        name: {(image["imagePath"], q["question"]): q for image in document["results"] for q in image["questions"]}
        for name, document in documents.items()
    }
    keys = set(sources[STILL_CONDITIONS[0]])
    if len(keys) != 1100 or any(set(source) != keys for source in sources.values()):
        raise ValueError("Still-image result conditions do not share the expected 1,100 questions")
    rows = []
    affected = []
    for identifier, question in sorted(keys):
        frozen = sources[STILL_CONDITIONS[0]][(identifier, question)]["expected"]
        reference, reference_type, adjudicated = corrected_reference(
            index, "still", identifier, question, frozen
        )
        row = {
            "identifier": identifier,
            "question": question,
            "frozen_reference": frozen,
            "corrected_reference": reference,
            "reference_type": reference_type,
            "conditions": {},
        }
        changed = reference != frozen or adjudicated
        for condition in STILL_CONDITIONS:
            record = sources[condition][(identifier, question)]
            corrected = score(record["predicted"], reference, reference_type)
            row["conditions"][condition] = {
                "predicted": record["predicted"],
                "original_correct": bool(record["correct"]),
                "correct": corrected,
            }
            changed = changed or corrected != bool(record["correct"])
        rows.append(row)
        if changed:
            affected.append(row)
    totals = {}
    for condition in STILL_CONDITIONS:
        document = documents[condition]
        correct = sum(row["conditions"][condition]["correct"] for row in rows)
        totals[condition] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows),
            "failed_images": document["summary"]["preprocessing_failures"],
        }
    return {
        "question_count": len(rows),
        "totals": totals,
        "original_vs_full_multiview": paired(rows, "original_only", "full_multiview"),
        "affected_questions": affected,
        "rows": rows,
    }


def video_analysis(index: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    documents = {name: load(path) for name, path in VIDEO_CONDITIONS.items()}
    sources = {
        name: {(q["video_filename"], q["question_id"]): q for q in document["results"]}
        for name, document in documents.items()
    }
    keys = set(sources["stride_6_first_match"])
    if len(keys) != 78 or any(set(source) != keys for source in sources.values()):
        raise ValueError("Video conditions do not share the expected 78 questions")
    rows = []
    affected = []
    for identifier, question_id in sorted(keys):
        first_record = sources["stride_6_first_match"][(identifier, question_id)]
        question = first_record["question"]
        frozen = first_record["expected_direction"]
        reference, reference_type, adjudicated = corrected_reference(
            index, "video", identifier, question, frozen
        )
        row = {
            "identifier": identifier,
            "question_id": question_id,
            "question": question,
            "frozen_reference": frozen,
            "corrected_reference": reference,
            "reference_type": reference_type,
            "conditions": {},
        }
        changed = reference != frozen or adjudicated
        for condition in VIDEO_CONDITIONS:
            record = sources[condition][(identifier, question_id)]
            prediction = record.get("gemini_prediction", "unknown")
            corrected = score(prediction, reference, reference_type)
            row["conditions"][condition] = {
                "predicted": prediction,
                "original_correct": bool(record["correct"]),
                "correct": corrected,
                "retrieved": bool(record["candidate_retrieved"]),
            }
            changed = changed or corrected != bool(record["correct"])
        rows.append(row)
        if changed:
            affected.append(row)
    totals = {}
    for condition in VIDEO_CONDITIONS:
        correct = sum(row["conditions"][condition]["correct"] for row in rows)
        retrieved = sum(row["conditions"][condition]["retrieved"] for row in rows)
        totals[condition] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows),
            "retrieved": retrieved,
        }
    policies = [
        "stride_6_first_match",
        "stride_6_fixed_window",
        "stride_6_last_seen",
        "stride_6_track_lifetime_best_view",
    ]
    policy_pairs = [paired(rows, policies[i], policies[j]) for i in range(len(policies)) for j in range(i + 1, len(policies))]
    strides = ["stride_3_fixed_window", "stride_6_fixed_window", "stride_12_fixed_window"]
    stride_pairs = [paired(rows, strides[i], strides[j]) for i in range(len(strides)) for j in range(i + 1, len(strides))]
    return {
        "question_count": len(rows),
        "totals": totals,
        "policy_pairs": policy_pairs,
        "stride_pairs": stride_pairs,
        "affected_questions": affected,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = load(args.manifest)
    for source in manifest["frozen_sources"].values():
        path = ROOT / source["path"]
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Frozen source hash mismatch: {path}")
    index = reference_index(manifest)
    still = still_analysis(index)
    video = video_analysis(index)
    provenance = {
        "schema_version": 2,
        "derivation": "Stored predictions rescored without inference",
        "manifest": str(args.manifest.relative_to(ROOT)),
        "manifest_sha256": sha256(args.manifest),
        "original_result_artifacts_modified": False,
    }
    dump(args.output_dir / "still_corrected_summary.json", {**provenance, **{k: v for k, v in still.items() if k != "rows"}})
    dump(args.output_dir / "still_corrected_per_question.json", {**provenance, "rows": still["rows"]})
    dump(args.output_dir / "video_corrected_summary.json", {**provenance, **{k: v for k, v in video.items() if k != "rows"}})
    dump(args.output_dir / "video_corrected_per_question.json", {**provenance, "rows": video["rows"]})
    print(json.dumps({"still": still["totals"], "video": video["totals"]}, indent=2))


if __name__ == "__main__":
    main()

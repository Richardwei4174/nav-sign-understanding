from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from src.pipeline.run_video_temporal_ablation import (
    FINAL_EXPERIMENT_MATRIX,
    POLICIES,
    PROJECT_ROOT,
    cache_preparation_action,
    cache_configuration,
    canonical_json_sha256,
    dry_run,
    load_qa_entries,
    normalize_loaded_cache,
    policy_result_path,
    prepare_cache,
    question_records,
    result_record,
    run_policy,
    score_reference,
    validate_cache,
    write_json_atomic,
)


class VideoResumeSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        (PROJECT_ROOT / "tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "tmp", prefix="video_resume_test_"
        )
        self.root = Path(self.temporary.name)
        self.video = self.root / "NEW_VIDEO.MOV"
        self.video.write_bytes(b"synthetic-video")
        self.items = [
            {"question": "Where is Alpha?", "answer": "unknown"},
            {"question": "Where is Beta?", "answer": "unknown"},
        ]
        self.entries = [
            {
                "videoPath": self.video.name,
                "questions": self.items,
                "_manifest_source": str(self.root / "extension.json"),
            }
        ]
        self.output_root = self.root / "outputs"
        self.cache_path = (
            self.output_root
            / "cache"
            / "stride_6"
            / self.video.stem
            / "observations.json"
        )
        questions = question_records(self.items)
        stored = [
            {key: value for key, value in item.items() if key != "question_item"}
            for item in questions
        ]
        write_json_atomic(
            self.cache_path,
            {
                "configuration": cache_configuration(6, self.video, self.items),
                "questions": stored,
                "frames": [],
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_extension_merge_rejects_duplicate_video_keys(self) -> None:
        base = self.root / "base.json"
        extension = self.root / "extension.json"
        base.write_text(
            json.dumps([{"videoPath": "A.MOV", "questions": self.items}]),
            encoding="utf-8",
        )
        extension.write_text(
            json.dumps([{"videoPath": "B.MOV", "questions": self.items}]),
            encoding="utf-8",
        )
        merged = load_qa_entries(base, (extension,))
        self.assertEqual([item["videoPath"] for item in merged], ["A.MOV", "B.MOV"])

        extension.write_text(
            json.dumps([{"videoPath": "A.MOV", "questions": self.items}]),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Duplicate videoPath"):
            load_qa_entries(base, (extension,))

    def test_explicit_reference_scoring(self) -> None:
        self.assertTrue(score_reference("left", "left", "scalar"))
        self.assertTrue(
            score_reference("right", ["right", "left"], "alternatives")
        )
        self.assertTrue(
            score_reference("left", ["right", "left"], "alternatives")
        )
        self.assertFalse(
            score_reference("straight", ["right", "left"], "alternatives")
        )
        # A scalar reference never accepts a list containing an extra label.
        self.assertFalse(score_reference(["left", "right"], "left", "scalar"))

    def test_final_matrix_and_result_paths_are_unique(self) -> None:
        self.assertEqual(len(POLICIES), 4)
        self.assertEqual(len(FINAL_EXPERIMENT_MATRIX), 6)
        self.assertEqual(len(set(FINAL_EXPERIMENT_MATRIX)), 6)
        self.assertEqual(
            {
                stride
                for stride, policy in FINAL_EXPERIMENT_MATRIX
                if policy == "track_lifetime_best_view"
            },
            {3, 6, 12},
        )
        self.assertEqual(
            FINAL_EXPERIMENT_MATRIX.count((6, "track_lifetime_best_view")),
            1,
        )
        fixed_3 = policy_result_path(
            self.output_root, 3, "fixed_window", self.video
        )
        track_3 = policy_result_path(
            self.output_root, 3, "track_lifetime_best_view", self.video
        )
        track_12 = policy_result_path(
            self.output_root, 12, "track_lifetime_best_view", self.video
        )
        self.assertNotEqual(fixed_3, track_3)
        self.assertNotEqual(track_3, track_12)
        self.assertIn("fixed_window", fixed_3.parts)
        self.assertIn("track_lifetime_best_view", track_3.parts)

    def test_track_lifetime_runs_from_each_stride_specific_cache(self) -> None:
        stored = [
            {key: value for key, value in item.items() if key != "question_item"}
            for item in question_records(self.items)
        ]
        for stride in (3, 6, 12):
            cache_path = (
                self.output_root
                / "cache"
                / f"stride_{stride}"
                / self.video.stem
                / "observations.json"
            )
            write_json_atomic(
                cache_path,
                {
                    "configuration": cache_configuration(
                        stride, self.video, self.items
                    ),
                    "questions": stored,
                    "frames": [],
                },
            )
            result = run_policy(
                self.video,
                self.entries,
                self.output_root,
                stride,
                "track_lifetime_best_view",
                selection_only=True,
                limit_targets=None,
                root=".",
                api_key_path="unused",
                model_version="unused",
                prompt_file="unused",
                resume=True,
                overwrite=False,
            )
            self.assertEqual(
                result,
                policy_result_path(
                    self.output_root,
                    stride,
                    "track_lifetime_best_view",
                    self.video,
                ),
            )

    def test_untyped_list_is_rejected_and_fingerprint_tracks_semantics(self) -> None:
        untyped = [{"question": "Where is Exit?", "answer": ["left", "right"]}]
        with self.assertRaisesRegex(ValueError, "reference_type"):
            question_records(untyped)
        typed = [
            {
                "question": "Where is Exit?",
                "answer": ["left", "right"],
                "reference_type": "alternatives",
            }
        ]
        self.assertNotEqual(
            canonical_json_sha256(untyped),
            canonical_json_sha256(typed),
        )

    def test_legacy_scalar_cache_is_reusable(self) -> None:
        cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        for question in cache["questions"]:
            question.pop("reference_type")
        cache["configuration"]["qa_sha256"] = "legacy-whole-manifest-hash"
        cache["configuration"].pop("qa_video_sha256")
        current = question_records(self.items)
        expected = [
            {key: value for key, value in item.items() if key != "question_item"}
            for item in current
        ]
        validate_cache(
            cache,
            cache_configuration(6, self.video, self.items),
            expected,
            self.cache_path,
        )
        normalized = normalize_loaded_cache(cache)
        self.assertEqual(
            [item["reference_type"] for item in normalized["questions"]],
            ["scalar", "scalar"],
        )
        self.assertNotIn("reference_type", cache["questions"][0])

    def test_policy_path_normalizes_legacy_and_preserves_alternatives(
        self,
    ) -> None:
        prompt = self.root / "prompt.txt"
        prompt.write_text("synthetic prompt", encoding="utf-8")
        prompt_relative = prompt.relative_to(PROJECT_ROOT).as_posix()

        def write_policy_cache(
            video: Path,
            items: list[dict[str, object]],
            *,
            legacy: bool,
        ) -> Path:
            stored = [
                {
                    key: value
                    for key, value in item.items()
                    if key != "question_item"
                }
                for item in question_records(items)
            ]
            if legacy:
                for item in stored:
                    item.pop("reference_type")
            frame_path = self.root / f"{video.stem}_frame.jpg"
            crop_path = self.root / f"{video.stem}_crop.jpg"
            frame_path.write_bytes(b"frame")
            crop_path.write_bytes(b"crop")
            matches = {
                item["question_id"]: {
                    "destination": item["destination"],
                    "match_score": 90.0 if index == 0 else 0.0,
                    "qualifies": index == 0,
                }
                for index, item in enumerate(stored)
            }
            cache_path = (
                self.output_root
                / "cache"
                / "stride_3"
                / video.stem
                / "observations.json"
            )
            write_json_atomic(
                cache_path,
                {
                    "configuration": cache_configuration(3, video, items),
                    "questions": stored,
                    "frames": [
                        {
                            "source_frame_index": 0,
                            "source_timestamp_seconds": 0.0,
                            "processed_frame_index": 0,
                            "full_frame_path": frame_path.relative_to(
                                PROJECT_ROOT
                            ).as_posix(),
                            "detections": [
                                {
                                    "bbox_xyxy": [0, 0, 2, 2],
                                    "stable_track_id": "bytetrack_1",
                                    "raw_crop_path": crop_path.relative_to(
                                        PROJECT_ROOT
                                    ).as_posix(),
                                    "ocr_text": "Exit",
                                    "average_ocr_confidence": 0.9,
                                    "visual_quality": {"final_score": 10.0},
                                    "target_matches": matches,
                                }
                            ],
                        }
                    ],
                },
            )
            return cache_path

        legacy_items: list[dict[str, object]] = [
            {"question": "Where is Exit?", "answer": "right"},
            {"question": "Where is Missing?", "answer": "unknown"},
        ]
        legacy_entries = [
            {
                "videoPath": self.video.name,
                "questions": legacy_items,
                "_manifest_source": str(self.root / "base.json"),
            }
        ]
        legacy_cache = write_policy_cache(
            self.video, legacy_items, legacy=True
        )
        legacy_bytes = legacy_cache.read_bytes()
        legacy_mtime = legacy_cache.stat().st_mtime_ns

        alternative_video = self.root / "ALTERNATIVE.MOV"
        alternative_video.write_bytes(b"alternative-video")
        alternative_items: list[dict[str, object]] = [
            {
                "question": "Where is Exit?",
                "answer": ["right", "left"],
                "reference_type": "alternatives",
            }
        ]
        alternative_entries = [
            {
                "videoPath": alternative_video.name,
                "questions": alternative_items,
                "_manifest_source": str(self.root / "extension.json"),
            }
        ]
        alternative_cache = write_policy_cache(
            alternative_video, alternative_items, legacy=False
        )

        gemini_module = ModuleType(
            "src.understand.code.rpi_continuous_testing"
        )
        gemini_module.GeminiDirectionQA = (
            lambda **unused_kwargs: object()
        )
        stream_module = ModuleType(
            "src.pipeline.run_stream_video_pipeline"
        )
        stream_module.build_detection = lambda value: value
        multiview_module = ModuleType(
            "src.pipeline.run_multiview_pipeline"
        )

        def fake_multiview(**kwargs: object) -> dict[str, object]:
            output_root = Path(kwargs["output_root"])
            image_name = str(kwargs["image_output_name"])
            write_json_atomic(
                output_root / image_name / "gemini_results.json",
                {"gemini_attempts": 1},
            )
            return {"results": [{"predicted": "right"}]}

        multiview_module.run_multiview_from_detections = fake_multiview

        def execute(video: Path, entries: list[dict[str, object]]) -> Path:
            with patch.dict(
                sys.modules,
                {
                    "src.understand.code.rpi_continuous_testing": gemini_module,
                    "src.pipeline.run_stream_video_pipeline": stream_module,
                    "src.pipeline.run_multiview_pipeline": multiview_module,
                },
            ):
                return run_policy(
                    video,
                    entries,
                    self.output_root,
                    3,
                    "track_lifetime_best_view",
                    selection_only=False,
                    limit_targets=None,
                    root=".",
                    api_key_path="unused",
                    model_version="synthetic",
                    prompt_file=prompt_relative,
                    resume=True,
                    overwrite=False,
                )

        legacy_result = execute(self.video, legacy_entries)
        legacy_payload = json.loads(legacy_result.read_text(encoding="utf-8"))
        self.assertEqual(
            [item["reference_type"] for item in legacy_payload["results"]],
            ["scalar", "scalar"],
        )
        self.assertTrue(legacy_payload["results"][0]["candidate_retrieved"])
        self.assertEqual(
            legacy_payload["results"][0]["gemini_prediction"], "right"
        )
        self.assertFalse(
            legacy_payload["results"][1]["candidate_retrieved"]
        )
        self.assertEqual(
            legacy_payload["results"][1]["gemini_prediction"], "unknown"
        )
        self.assertEqual(legacy_cache.read_bytes(), legacy_bytes)
        self.assertEqual(legacy_cache.stat().st_mtime_ns, legacy_mtime)

        alternative_result = execute(
            alternative_video, alternative_entries
        )
        alternative_record = json.loads(
            alternative_result.read_text(encoding="utf-8")
        )["results"][0]
        self.assertEqual(
            alternative_record["expected_direction"], ["right", "left"]
        )
        self.assertEqual(
            alternative_record["reference_type"], "alternatives"
        )
        self.assertTrue(alternative_record["correct"])
        self.assertIn(
            b'"reference_type": "alternatives"',
            alternative_cache.read_bytes(),
        )

        # Resume a valid partial result, then verify a second resume is
        # byte-idempotent and does not call Gemini for completed questions.
        partial = dict(legacy_payload)
        partial["results"] = partial["results"][:1]
        partial["summary"] = {
            key: value
            for key, value in legacy_payload["summary"].items()
        }
        write_json_atomic(legacy_result, partial)
        resumed = execute(self.video, legacy_entries)
        resumed_bytes = resumed.read_bytes()
        self.assertEqual(len(json.loads(resumed_bytes)["results"]), 2)
        execute(self.video, legacy_entries)
        self.assertEqual(resumed.read_bytes(), resumed_bytes)

    def test_extension_resume_reuses_legacy_original_and_plans_only_extension(
        self,
    ) -> None:
        extension_video = self.root / "EXTENSION_VIDEO.MOV"
        extension_video.write_bytes(b"synthetic-extension-video")
        base_manifest = self.root / "base.json"
        extension_manifest = self.root / "extension.json"
        base_manifest.write_text(
            json.dumps(
                [{"videoPath": self.video.name, "questions": self.items}]
            ),
            encoding="utf-8",
        )
        extension_manifest.write_text(
            json.dumps(
                [{"videoPath": extension_video.name, "questions": self.items}]
            ),
            encoding="utf-8",
        )
        entries = load_qa_entries(base_manifest, (extension_manifest,))

        legacy = json.loads(self.cache_path.read_text(encoding="utf-8"))
        legacy["configuration"]["qa_sha256"] = "legacy-whole-manifest-hash"
        legacy["configuration"].pop("qa_video_sha256")
        for question in legacy["questions"]:
            question.pop("reference_type")
        write_json_atomic(self.cache_path, legacy)
        before_bytes = self.cache_path.read_bytes()
        before_mtime = self.cache_path.stat().st_mtime_ns

        planned = [
            cache_preparation_action(
                video,
                entries,
                self.output_root,
                6,
                resume=True,
                overwrite=False,
            )
            for video in (self.video, extension_video)
        ]
        self.assertEqual([action for action, _ in planned], ["reuse", "create"])
        self.assertEqual(self.cache_path.read_bytes(), before_bytes)
        self.assertEqual(self.cache_path.stat().st_mtime_ns, before_mtime)
        self.assertEqual(
            planned[1][1],
            self.output_root
            / "cache"
            / "stride_6"
            / extension_video.stem
            / "observations.json",
        )

        args = argparse.Namespace(
            video_dir=str(self.root),
            qa_extension=[extension_manifest],
            prepare_cache=True,
            policy=None,
            frame_stride=6,
            resume=True,
            overwrite=False,
        )
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            dry_run(
                args,
                [self.video, extension_video],
                base_manifest,
                entries,
                self.output_root,
            )
        report = json.loads(stream.getvalue())
        self.assertEqual(
            [item["cache_action"] for item in report["videos"]],
            ["reuse", "create"],
        )
        self.assertFalse(planned[1][1].exists())
        self.assertEqual(self.cache_path.read_bytes(), before_bytes)
        self.assertEqual(self.cache_path.stat().st_mtime_ns, before_mtime)

    def test_missing_extension_cache_real_branch_persists_canonical_questions(
        self,
    ) -> None:
        extension_video = self.root / "EXTENSION_VIDEO.MOV"
        extension_video.write_bytes(b"synthetic-extension-video")
        alternative_items = [
            {
                "question": "Where is Exit?",
                "answer": ["right", "left"],
                "reference_type": "alternatives",
            }
        ]
        entries = [
            {
                "videoPath": self.video.name,
                "questions": self.items,
                "_manifest_source": str(self.root / "base.json"),
            },
            {
                "videoPath": extension_video.name,
                "questions": alternative_items,
                "_manifest_source": str(self.root / "extension.json"),
            },
        ]

        legacy = json.loads(self.cache_path.read_text(encoding="utf-8"))
        legacy["configuration"]["qa_sha256"] = "legacy-whole-manifest-hash"
        legacy["configuration"].pop("qa_video_sha256")
        for question in legacy["questions"]:
            question.pop("reference_type")
        write_json_atomic(self.cache_path, legacy)
        original_bytes = self.cache_path.read_bytes()
        original_mtime = self.cache_path.stat().st_mtime_ns

        class FakeFrame:
            shape = (4, 4, 3)

        class FakeCapture:
            def __init__(self, unused_path: str) -> None:
                self.read_count = 0

            def isOpened(self) -> bool:
                return True

            def get(self, key: int) -> float:
                return {
                    1: 30.0,
                    2: 1.0,
                    3: 0.0,
                }.get(key, 0.0)

            def read(self) -> tuple[bool, object | None]:
                self.read_count += 1
                return (
                    (True, FakeFrame())
                    if self.read_count == 1
                    else (False, None)
                )

            def release(self) -> None:
                pass

        cv2 = ModuleType("cv2")
        cv2.CAP_PROP_FPS = 1
        cv2.CAP_PROP_FRAME_COUNT = 2
        cv2.CAP_PROP_POS_MSEC = 3
        cv2.VideoCapture = FakeCapture

        def fake_imwrite(path: str, unused_frame: object) -> bool:
            Path(path).write_bytes(b"synthetic-jpeg")
            return True

        cv2.imwrite = fake_imwrite

        paddleocr = ModuleType("paddleocr")
        paddleocr.PaddleOCR = lambda **unused_kwargs: object()
        ultralytics = ModuleType("ultralytics")

        class FakeModel:
            names: dict[int, str] = {}

            def __init__(self, unused_model: str) -> None:
                pass

            def set_classes(self, unused_classes: list[str]) -> None:
                pass

            def track(self, **unused_kwargs: object) -> list[object]:
                return [SimpleNamespace(boxes=[])]

        ultralytics.YOLOWorld = FakeModel
        stream_pipeline = ModuleType(
            "src.pipeline.run_stream_video_pipeline"
        )
        stream_pipeline.get_avg_confidence = lambda unused_lines: 0.0
        stream_pipeline.get_detection_text = lambda unused_lines: ""
        stream_pipeline.run_ocr = (
            lambda unused_ocr, unused_crop: []
        )
        stream_pipeline.score_match = (
            lambda unused_target, unused_text: 0.0
        )

        with patch.dict(
            sys.modules,
            {
                "cv2": cv2,
                "paddleocr": paddleocr,
                "ultralytics": ultralytics,
                "src.pipeline.run_stream_video_pipeline": stream_pipeline,
            },
        ):
            reused = prepare_cache(
                self.video,
                entries,
                self.output_root,
                6,
                resume=True,
                overwrite=False,
            )
            created = prepare_cache(
                extension_video,
                entries,
                self.output_root,
                6,
                resume=True,
                overwrite=False,
            )

        self.assertEqual(reused, self.cache_path)
        self.assertEqual(self.cache_path.read_bytes(), original_bytes)
        self.assertEqual(self.cache_path.stat().st_mtime_ns, original_mtime)
        persisted = json.loads(created.read_text(encoding="utf-8"))
        self.assertEqual(
            persisted["questions"],
            [
                {
                    "question_id": "question_0000",
                    "question": "Where is Exit?",
                    "destination": "Exit",
                    "expected_direction": ["right", "left"],
                    "reference_type": "alternatives",
                    "target_present": True,
                    "presence_source": "inferred_from_expected_answer",
                }
            ],
        )
        self.assertEqual(
            persisted["configuration"]["qa_video_sha256"],
            canonical_json_sha256(alternative_items),
        )

    def test_atomic_json_failure_leaves_no_partial_output(self) -> None:
        output = self.root / "atomic" / "observations.json"
        output.parent.mkdir(parents=True)
        output.write_bytes(b'{"previous": true}')
        previous = output.read_bytes()
        with patch(
            "src.pipeline.run_video_temporal_ablation.json.dump",
            side_effect=RuntimeError("synthetic serialization failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "synthetic serialization failure"
            ):
                write_json_atomic(output, {"incomplete": True})
        self.assertEqual(output.read_bytes(), previous)
        self.assertFalse(output.with_suffix(".json.tmp").exists())

    def test_dry_run_is_idempotent_and_writes_nothing(self) -> None:
        manifest = self.root / "base.json"
        manifest.write_text(
            json.dumps(
                [{"videoPath": self.video.name, "questions": self.items}]
            ),
            encoding="utf-8",
        )
        entries = load_qa_entries(manifest)
        args = argparse.Namespace(
            video_dir=str(self.root),
            qa_extension=[],
            prepare_cache=True,
            policy=None,
            frame_stride=6,
            resume=True,
            overwrite=False,
        )
        before = {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        outputs = []
        for _ in range(2):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                dry_run(
                    args,
                    [self.video],
                    manifest,
                    entries,
                    self.output_root,
                )
            outputs.append(stream.getvalue())
        after = {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(before, after)

    def test_partial_result_resumes_only_missing_and_is_idempotent(self) -> None:
        result_path = (
            self.output_root
            / "runs"
            / "stride_6"
            / "fixed_window"
            / self.video.stem
            / "per_target_results.json"
        )
        first = question_records(self.items)[0]
        completed_record = result_record(first, None, None, 0)
        write_json_atomic(
            result_path,
            {
                "video_filename": self.video.name,
                "policy": "fixed_window",
                "stride": 6,
                "selection_only": True,
                "results": [
                    completed_record
                ],
            },
        )

        run_policy(
            self.video,
            self.entries,
            self.output_root,
            6,
            "fixed_window",
            selection_only=True,
            limit_targets=None,
            root=".",
            api_key_path="unused",
            model_version="unused",
            prompt_file="unused",
            resume=True,
            overwrite=False,
        )
        first_resume = result_path.read_bytes()
        results = json.loads(first_resume)["results"]
        self.assertEqual([item["question_id"] for item in results], [
            "question_0000",
            "question_0001",
        ])

        run_policy(
            self.video,
            self.entries,
            self.output_root,
            6,
            "fixed_window",
            selection_only=True,
            limit_targets=None,
            root=".",
            api_key_path="unused",
            model_version="unused",
            prompt_file="unused",
            resume=True,
            overwrite=False,
        )
        self.assertEqual(result_path.read_bytes(), first_resume)

    def test_annotation_or_configuration_change_is_rejected(self) -> None:
        cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        changed_items = [
            {"question": "Where is changed?", "answer": "unknown"},
            self.items[1],
        ]
        changed_questions = question_records(changed_items)
        stored = [
            {key: value for key, value in item.items() if key != "question_item"}
            for item in changed_questions
        ]
        with self.assertRaisesRegex(ValueError, "annotations"):
            validate_cache(
                cache,
                cache_configuration(6, self.video, changed_items),
                stored,
                self.cache_path,
            )

        wrong_stride = cache_configuration(3, self.video, self.items)
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_cache(
                cache,
                wrong_stride,
                cache["questions"],
                self.cache_path,
            )


if __name__ == "__main__":
    unittest.main()

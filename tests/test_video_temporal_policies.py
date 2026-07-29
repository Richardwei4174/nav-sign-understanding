from __future__ import annotations

import unittest

from src.pipeline.run_video_temporal_ablation import (
    LAST_SEEN_GRACE_UPDATES,
    evaluate_track_lifetime_best_view,
    select_first_match,
    select_fixed_window,
    select_last_seen,
)


QUESTION_ID = "question_0000"


def detection(track: str, quality: float, match: float = 0.0, confidence: float = 0.5):
    return {
        "stable_track_id": track,
        "raw_crop_path": f"{track}_{quality}.jpg",
        "ocr_text": "target" if match >= 70 else "other",
        "average_ocr_confidence": confidence,
        "visual_quality": {"final_score": quality},
        "target_matches": {
            QUESTION_ID: {"match_score": match, "qualifies": match >= 70}
        },
    }


def frame(index: int, *detections):
    return {
        "processed_frame_index": index,
        "source_frame_index": index * 6,
        "source_timestamp_seconds": index * 0.2,
        "full_frame_path": f"frame_{index}.jpg",
        "detections": list(detections),
    }


class TrackLifetimeBestViewTests(unittest.TestCase):
    def test_binding_best_view_miss_reset_and_exact_close(self):
        frames = [
            frame(0, detection("track_a", 5.0)),
            frame(1, detection("track_a", 4.0, match=90.0, confidence=0.8)),
            # A stronger competing track must never steal the binding.
            frame(2, detection("track_b", 100.0, match=100.0, confidence=1.0)),
            # Better visual evidence may reuse the qualifying OCR evidence.
            frame(3, detection("track_a", 10.0)),
            frame(4),
            frame(5),
            # Reappearance resets the two accumulated misses.
            frame(6, detection("track_a", 11.0)),
        ]
        frames.extend(frame(index) for index in range(7, 15))
        # This would win, but it appears only after the eighth miss closed the window.
        frames.append(frame(15, detection("track_a", 1000.0)))

        result = evaluate_track_lifetime_best_view({"frames": frames}, QUESTION_ID)

        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.audit["bound_track_id"], "track_a")
        self.assertEqual(result.audit["binding_count"], 1)
        self.assertEqual(result.audit["track_switch_count"], 0)
        self.assertGreater(result.audit["visual_updates_after_binding"], 0)
        self.assertGreater(result.audit["candidate_improvements_after_initial"], 0)
        self.assertEqual(result.audit["miss_counter_resets"], 2)
        self.assertTrue(result.audit["closed_by_grace"])
        self.assertEqual(result.audit["misses_at_close"], LAST_SEEN_GRACE_UPDATES)
        self.assertFalse(result.audit["force_closed_at_video_end"])
        self.assertEqual(result.candidate.frame["processed_frame_index"], 6)
        self.assertEqual(result.candidate.ocr_text, "target")

    def test_pre_match_best_view_can_be_selected_and_video_end_force_closes(self):
        cache = {
            "frames": [
                frame(0, detection("track_a", 20.0)),
                frame(1, detection("track_a", 10.0, match=85.0, confidence=0.7)),
                frame(2, detection("track_a", 15.0)),
            ]
        }

        result = evaluate_track_lifetime_best_view(cache, QUESTION_ID)

        self.assertEqual(result.candidate.frame["processed_frame_index"], 0)
        self.assertTrue(result.audit["selected_before_first_match"])
        self.assertFalse(result.audit["closed_by_grace"])
        self.assertTrue(result.audit["force_closed_at_video_end"])

    def test_original_three_selectors_keep_their_existing_behavior(self):
        cache = {
            "frames": [
                frame(0, detection("track_a", 1.0, match=80.0, confidence=0.2)),
                frame(1, detection("track_a", 2.0, match=90.0, confidence=0.9)),
            ]
        }

        first, first_count = select_first_match(cache, QUESTION_ID)
        fixed, fixed_count = select_fixed_window(cache, QUESTION_ID)
        last, last_count = select_last_seen(cache, QUESTION_ID)

        self.assertEqual(first.frame["processed_frame_index"], 0)
        self.assertEqual(first_count, 1)
        self.assertEqual(fixed.frame["processed_frame_index"], 1)
        self.assertEqual(fixed_count, 2)
        self.assertEqual(last.frame["processed_frame_index"], 1)
        self.assertEqual(last_count, 2)


if __name__ == "__main__":
    unittest.main()

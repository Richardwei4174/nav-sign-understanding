import argparse
import csv
import json
import queue
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLOWorld
from paddleocr import PaddleOCR
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]

DEBUG_LOG_FILE = None
DEBUG_LOG_HANDLE = None
DEBUG_LOG_LOCK = threading.Lock()


PERF_LOG_FILE = None
PERF_LOG_LOCK = threading.Lock()


def update_runtime_stats(stats, stats_lock, **values):
    with stats_lock:
        stats.update(values)


def increment_runtime_stat(stats, stats_lock, key, amount=1):
    with stats_lock:
        stats[key] = stats.get(key, 0) + amount


def snapshot_runtime_stats(stats, stats_lock):
    with stats_lock:
        return dict(stats)


def append_perf_row(row):
    global PERF_LOG_FILE

    if PERF_LOG_FILE is None:
        return

    with PERF_LOG_LOCK:
        file_exists = PERF_LOG_FILE.exists()
        with open(PERF_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            f.flush()

# Track now stores best crop / best frame / best bbox
# Crop quality now considers size, sharpness, YOLO confidence, and edge cutoff
# OCR retries if early OCR was empty/weak and crop improves
# Gemini selection uses visual quality and closes after a last-seen grace buffer

def debug_log(message):
    """
    Print to terminal AND save to a debug log.

    NOTE: previously this opened, wrote, flushed, and closed the log file on
    every single call. With how chatty this pipeline is (a print per box,
    per OCR job, per YOLO frame, per track-state change), that meant one
    open+close syscall pair per message - real, synchronous disk I/O
    competing with the GPU/CPU-bound detection and OCR work on every frame.
    Under memory pressure (e.g. once the system starts swapping), that I/O
    gets dramatically slower and compounds the slowdown. We now keep a
    single file handle open for the duration of the run and just write to
    it, flushing so the file still stays readable live.
    """

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {message}"

    _original_print(line)

    global DEBUG_LOG_HANDLE

    if DEBUG_LOG_HANDLE is None:
        return

    with DEBUG_LOG_LOCK:
        DEBUG_LOG_HANDLE.write(line + "\n")
        DEBUG_LOG_HANDLE.flush()



import builtins
_original_print = builtins.print


def print(*args, **kwargs):
    """
    Route normal print() calls to both terminal and debug log.

    Notes:
    - We intentionally ignore print(file=...) for this pipeline's debug messages.
    - _original_print is used inside debug_log() to avoid infinite recursion.
    """
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    msg = sep.join(str(a) for a in args)

    # Keep behavior close to print(): if end is not a newline, preserve it in message.
    if end != "\n":
        msg += end

    debug_log(msg)

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SRC_ROOT))

from src.pipeline.run_multiview_pipeline import run_multiview_from_detections
from src.understand.code.rpi_continuous_testing import GeminiDirectionQA


def load_video_qa(qa_file, video_path):
    with open(qa_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    video_name = Path(video_path).name

    for item in qa_data:
        if item["videoPath"] == video_name:
            return item["questions"]

    raise ValueError(f"No QA found for video: {video_name}")

def load_live_qa(qa_file):
    with open(qa_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # If the JSON is a list of video/image entries, use the first entry for now.
    if isinstance(qa_data, list):
        return qa_data[0]["questions"]

    # If the JSON is already a live-format object, use its questions directly.
    return qa_data["questions"]

def extract_target_from_question(question_item):
    if "destination" in question_item:
        return question_item["destination"]

    question = question_item["question"].strip()
    question = re.sub(r"^where\s+is\s+", "", question, flags=re.IGNORECASE)
    question = question.rstrip("?").strip()

    return question


def has_digit_target(target):
    return any(ch.isdigit() for ch in target)


def extract_tokens(text):
    return re.findall(r"[A-Za-z]+|\d+", text.lower())


def score_match(target, text):
    target_lower = target.lower().strip()
    text_lower = text.lower().strip()

    target_tokens = extract_tokens(target_lower)
    text_tokens = extract_tokens(text_lower)

    # Allow single-character OCR only if it is an exact token in the target.
    # Example: target="platform A", OCR="A" should pass.
    # Example: target="exit", OCR="x" should fail.
    if len(text_lower) <= 1:
        return 100 if text_lower in target_tokens else 0
    
    if has_digit_target(target_lower):
        target_numbers = [tok for tok in target_tokens if tok.isdigit()]
        text_numbers = [tok for tok in text_tokens if tok.isdigit()]

        for num in target_numbers:
            if num not in text_numbers:
                return 0

        return fuzz.partial_ratio(target_lower, text_lower)

    # Short single-word targets require an exact token match.
    if len(target_tokens) == 1 and len(target_tokens[0]) <= 4:
        return 100 if target_tokens[0] in text_tokens else 0

    return fuzz.partial_ratio(target_lower, text_lower)


def resize_for_ocr(crop, max_side=480):
    """
    Downscale large crops before PaddleOCR.
    OCR cost grows with pixels, and sign crops usually do not need full resolution.
    """
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    longest = max(h, w)

    if longest <= max_side:
        return crop

    scale = max_side / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)


MIN_RETRY_GAP_FRAMES = 6

# Every track gets at least this many OCR passes before we fall back to
# confidence/quality-based retry logic. A sign is often only partially
# legible or truncated on the very first pass (e.g. a two-line or ranged
# sign like "5100-5119"), even when PaddleOCR reports high confidence on
# the piece it did read. One guaranteed extra look, on a slightly later
# (usually larger/sharper) crop, catches most of these without having to
# wait for a big quality jump.
GUARANTEED_OCR_ATTEMPTS = 2


# Remove stale per-track state so long live runs do not retain every historical
# best frame/crop forever. This is based on original camera-frame indices.
# At a 30 FPS camera, 300 frames is approximately 10 seconds.
TRACK_RETENTION_FRAMES = 300


# Cap the number of full annotated frames waiting for VideoWriter.
# A 960x540 BGR frame is about 1.56 MB, so 120 queued frames is roughly
# 187 MB rather than an unbounded amount of RAM.
WRITER_QUEUE_MAXSIZE = 120


# Hard ceiling on how many live tracks (confirmed + temporary) we will keep
# state for at once. This existed only implicitly before (bounded solely by
# TRACK_RETENTION_FRAMES, i.e. ~10s of not being seen) - under a low YOLO
# confidence threshold, noisy detections could spawn far more temporary
# tracks than that eviction policy could keep up with, and every track
# holds a full-resolution frame + crop copy (multi-MB each). This cap is a
# backstop: if we ever exceed it, we proactively evict the oldest,
# lowest-priority unprotected tracks immediately instead of waiting for the
# normal age-based cleanup.
MAX_ACTIVE_TRACKS = 60


def run_ocr(ocr, crop_path):
    crop_path = resize_for_ocr(crop_path)
    ocr_result = ocr.ocr(crop_path, cls=True)

    ocr_lines = []

    if ocr_result and ocr_result[0]:
        for line in ocr_result[0]:
            ocr_lines.append({
                "text": line[1][0],
                "confidence": round(float(line[1][1]), 3),
            })

    return ocr_lines


def get_detection_text(ocr_lines):
    return " ".join([line["text"] for line in ocr_lines]).strip()


def get_avg_confidence(ocr_lines):
    if not ocr_lines:
        return 0.0

    return sum(line["confidence"] for line in ocr_lines) / len(ocr_lines)


def build_detection(candidate):
    return [
        {
            "index": 0,
            "box": candidate["bbox_xyxy"],
            "label": "navigation sign",
            "confidence": 1.0,
        }
    ]


def make_unknown_result(question_item):
    return {
        "question": question_item["question"],
        "expected": question_item["answer"],
        "predicted": "unknown",
        "correct": question_item["answer"] == "unknown",
    }

# save as we go
def save_stream_summary(summary_path, video_path, evaluation_results):
    total_questions_answered = len(evaluation_results)

    total_correct = sum(
        result["correct"] for result in evaluation_results
    )

    accuracy = (
        total_correct / total_questions_answered
        if total_questions_answered > 0 else 0
    )

    summary = {
        "summary": {
            "total_questions": total_questions_answered,
            "total_correct": total_correct,
            "accuracy": accuracy,
        },
        "results": [
            {
                "videoPath": video_path.name,
                "results": evaluation_results,
            }
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary

def create_target_states(
    question_items,
    match_threshold,
    last_seen_grace_updates,
):
    """
    Create per-question state.

    Candidate collection is no longer closed after a fixed number of seconds.
    Once OCR matches a target, the target stays bound to that tracked sign.
    Collection closes only after that sign has been absent for a configurable
    number of processed detector updates.
    """
    target_states = {}

    for i, question_item in enumerate(question_items):
        target = extract_target_from_question(question_item)

        target_states[target] = {
            "target": target,
            "question_item": question_item,
            "question_index": i,
            "match_threshold": match_threshold,
            "first_match_frame": None,
            "first_match_time": None,
            "matched_track_id": None,
            "last_seen_frame": None,
            "consecutive_misses": 0,
            "last_seen_grace_updates": last_seen_grace_updates,
            "best_candidate": None,
            "candidate_observation_count": 0,
            "status": "searching",
            "submitted_to_gemini": False,
        }

    return target_states


def all_targets_finished(target_states):
    return all(
        state["status"] in ["window_done", "answered"]
        for state in target_states.values()
    )


def update_target_with_detection(
    state,
    frame_idx,
    frame_path,
    crop_path,
    bbox_xyxy,
    ocr_text,
    ocr_lines,
    avg_conf,
    save_debug_images,
    frame,
    crop,
    track_id,
    visual_quality,
):
    """
    Add or improve a target candidate.

    The first passing OCR match binds the target to one stable tracked object.
    Later candidates are accepted only from that same object. Among valid
    candidates, visual quality is the primary ranking signal; OCR confidence
    and fuzzy-match score are tie-breakers.
    """
    if state["status"] not in ["searching", "window_open"]:
        return

    match_score = score_match(state["target"], ocr_text) if ocr_text else 0

    if match_score < state["match_threshold"]:
        return

    # Once a target is bound to a physical track, do not let another duplicate
    # or nearby detection replace its candidate history.
    if (
        state["matched_track_id"] is not None
        and state["matched_track_id"] != track_id
    ):
        return

    print(
        f"MATCH PASSED | "
        f"target={repr(state['target'])} | "
        f"track={track_id} | "
        f"ocr={repr(ocr_text)} | "
        f"score={match_score}"
    )

    now = time.monotonic()

    if state["first_match_frame"] is None:
        state["first_match_frame"] = frame_idx
        state["first_match_time"] = now
        state["matched_track_id"] = track_id
        state["last_seen_frame"] = frame_idx
        state["consecutive_misses"] = 0
        state["status"] = "window_open"

        print(f"\nFIRST MATCH FOUND for target: {state['target']}")
        print(f"Matched track: {track_id}")
        print(f"First match frame: {state['first_match_frame']}")
        print(
            "Collection will close after "
            f"{state['last_seen_grace_updates']} missed detector updates."
        )
    else:
        state["last_seen_frame"] = frame_idx
        state["consecutive_misses"] = 0

    candidate_rank = (
        float(visual_quality),
        float(avg_conf),
        float(match_score),
    )

    candidate = {
        "frame_index": frame_idx,
        "frame_path": frame_path if save_debug_images else None,
        "crop_path": crop_path if save_debug_images else None,
        "frame_image": frame.copy(),
        "crop_image": crop.copy(),
        "bbox_xyxy": list(bbox_xyxy),
        "track_id": track_id,
        "ocr_text": ocr_text,
        "ocr": ocr_lines,
        "match_score": match_score,
        "avg_ocr_confidence": avg_conf,
        "visual_quality": float(visual_quality),
        "candidate_rank": list(candidate_rank),
        # Kept for compatibility with existing selection JSON consumers.
        "final_score": match_score + (avg_conf * 10),
    }

    # Do not retain every historical frame/crop. Each candidate can contain
    # multi-megabyte NumPy arrays, so keeping the entire history causes memory
    # growth during long live runs. Retain only the current best candidate.
    state["candidate_observation_count"] = (
        state.get("candidate_observation_count", 0) + 1
    )

    best_candidate = state["best_candidate"]
    best_rank = (
        tuple(best_candidate.get("candidate_rank", [0.0, 0.0, 0.0]))
        if best_candidate is not None
        else None
    )

    if best_candidate is None or candidate_rank > best_rank:
        state["best_candidate"] = candidate
        print(
            f"  New best candidate for {state['target']}! "
            f"quality={visual_quality:.1f}, "
            f"ocr_conf={avg_conf:.3f}"
        )


def update_target_visibility(target_states, seen_track_ids, frame_idx):
    """
    Update open target windows after one processed detector frame.

    Seeing the matched track resets the miss counter. Missing it increments
    the counter. A target closes only after the configured grace buffer is
    exhausted, allowing brief detector misses or occlusions.
    """
    for state in target_states.values():
        if state["status"] != "window_open":
            continue

        matched_track_id = state["matched_track_id"]

        if matched_track_id in seen_track_ids:
            state["last_seen_frame"] = frame_idx
            state["consecutive_misses"] = 0
            continue

        state["consecutive_misses"] += 1
        print(
            f"  Target {state['target']} missed matched track "
            f"{matched_track_id}: "
            f"{state['consecutive_misses']}/"
            f"{state['last_seen_grace_updates']}"
        )

        if (
            state["consecutive_misses"]
            >= state["last_seen_grace_updates"]
        ):
            state["status"] = "window_done"
            print(
                f"\nFinished last-seen window for target: "
                f"{state['target']}"
            )


def save_selection_for_target(run_dir, target, state):
    safe_target = re.sub(r"[^A-Za-z0-9_-]+", "_", target).strip("_")
    target_dir = run_dir / "selected" / safe_target
    target_dir.mkdir(parents=True, exist_ok=True)

    best_candidate = state["best_candidate"]

    selected_frame_path = target_dir / "best_frame.jpg"
    selected_crop_path = target_dir / "best_crop.jpg"

    if best_candidate["frame_path"] is not None:
        shutil.copy(best_candidate["frame_path"], selected_frame_path)
    else:
        cv2.imwrite(str(selected_frame_path), best_candidate["frame_image"])

    if best_candidate["crop_path"] is not None:
        shutil.copy(best_candidate["crop_path"], selected_crop_path)
    else:
        cv2.imwrite(str(selected_crop_path), best_candidate["crop_image"])

    # The raw NumPy frame/crop arrays have now been persisted to disk (the
    # only thing that still needs them). Drop the references immediately so
    # this multi-MB memory can be reclaimed right away instead of lingering
    # for the rest of this candidate's lifetime (which, if it gets reopened
    # via "Reopened target for later search" below, could otherwise be a
    # while).
    best_candidate["frame_image"] = None
    best_candidate["crop_image"] = None

    selection_result = {
        "target": target,
        "match_threshold": state["match_threshold"],
        "last_seen_grace_updates": state["last_seen_grace_updates"],
        "first_match_frame": state["first_match_frame"],
        "candidate_observation_count": state.get(
            "candidate_observation_count", 0
        ),
        "selected": {
            "frame_index": best_candidate["frame_index"],
            "frame_path": str(selected_frame_path),
            "crop_path": str(selected_crop_path),
            "bbox_xyxy": best_candidate["bbox_xyxy"],
            "track_id": best_candidate.get("track_id"),
            "visual_quality": best_candidate.get("visual_quality", 0.0),
            "candidate_rank": best_candidate.get("candidate_rank", []),
            "ocr_text": best_candidate["ocr_text"],
            "ocr": best_candidate["ocr"],
            "match_score": best_candidate["match_score"],
            "avg_ocr_confidence": best_candidate["avg_ocr_confidence"],
            "final_score": best_candidate["final_score"],
        },
    }

    selection_path = target_dir / "stream_selection.json"

    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection_result, f, indent=2, ensure_ascii=False)

    return selected_frame_path, selected_crop_path, selection_path

def put_latest_frame(
    frame_queue,
    item,
    runtime_stats=None,
    runtime_stats_lock=None,
):
    """
    Keep only the newest frame in the queue.
    This prevents the detector from falling behind the live camera.
    """
    try:
        frame_queue.put_nowait(item)
    except queue.Full:
        try:
            frame_queue.get_nowait()
            if runtime_stats is not None and runtime_stats_lock is not None:
                increment_runtime_stat(
                    runtime_stats,
                    runtime_stats_lock,
                    "dropped_detection_frames",
                )
        except queue.Empty:
            pass

        frame_queue.put_nowait(item)

def put_latest_detection(detection_result_queue, item):
    """
    Keep only the newest detection result in the queue.
    This prevents the detection thread from blocking if the GUI falls behind.
    """
    try:
        detection_result_queue.put_nowait(item)
    except queue.Full:
        try:
            detection_result_queue.get_nowait()
        except queue.Empty:
            pass

        detection_result_queue.put_nowait(item)


def put_ocr_job(ocr_queue, item):
    """
    Add an OCR job without freezing the detection thread.
    If the OCR queue is full, drop the oldest OCR job.
    """
    try:
        ocr_queue.put_nowait(item)
    except queue.Full:
        try:
            old_item = ocr_queue.get_nowait()
            ocr_queue.task_done()
        except queue.Empty:
            pass

        ocr_queue.put_nowait(item)


def put_video_frame(
    writer_queue,
    frame,
    runtime_stats,
    runtime_stats_lock,
):
    """
    Queue an annotated video frame without allowing unlimited RAM growth.

    When encoding falls behind and the queue is full, discard the oldest
    unwritten frame and keep the newest frame. This preserves near-live output
    instead of accumulating an ever-growing delay.
    """
    try:
        writer_queue.put_nowait(frame)
        return
    except queue.Full:
        pass

    try:
        old_frame = writer_queue.get_nowait()
        writer_queue.task_done()
        del old_frame
        increment_runtime_stat(
            runtime_stats,
            runtime_stats_lock,
            "dropped_writer_frames",
        )
    except queue.Empty:
        pass

    writer_queue.put_nowait(frame)


def video_writer_worker(
    writer_queue,
    video_writer,
    runtime_stats,
    runtime_stats_lock,
):
    """
    Writes annotated frames to disk on its own thread.

    cv2.VideoWriter.write() does real encoding work (CPU-bound), and doing
    it inline in the main capture/display loop is exactly what makes the
    live view feel laggy: every frame, the GUI loop stalls until the
    encoder finishes before it can read the next camera frame or refresh
    the window. Moving it here means the main loop only has to hand off a
    frame and move on - the on-screen video stays smooth even if encoding
    briefly falls behind.
    """
    while True:
        frame = writer_queue.get()

        if frame is None:
            writer_queue.task_done()
            break

        write_start = time.perf_counter()
        video_writer.write(frame)
        write_seconds = time.perf_counter() - write_start

        update_runtime_stats(
            runtime_stats,
            runtime_stats_lock,
            last_video_write_seconds=write_seconds,
            writer_queue_size=writer_queue.qsize(),
        )
        increment_runtime_stat(
            runtime_stats,
            runtime_stats_lock,
            "written_video_frames",
        )

        writer_queue.task_done()


def gemini_worker(
    gemini_queue,
    result_queue,
    stop_event,
    run_dir,
    output_root,
    qa,
    summary_path,
    live_video_path,
    evaluation_results,
    results_lock,
    video_name,
    target_states,
    target_states_lock,
):
    """
    Process each selected sign crop with all currently unresolved questions.

    Non-unknown answers are accumulated immediately. Unknown answers are not
    finalized during capture because the destination may appear on a later sign.
    Remaining unanswered questions are marked unknown only after all queued
    Gemini work has finished.
    """
    while True:
        try:
            trigger_target, trigger_state = gemini_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if trigger_target is None:
            gemini_queue.task_done()
            break

        # Another crop may already have answered this target while this job
        # waited in the queue. In that case there is no reason to process it.
        with target_states_lock:
            if trigger_state["status"] == "answered":
                print(
                    f"\n[Gemini thread] Skipping already-answered trigger: "
                    f"{trigger_target}"
                )
                gemini_queue.task_done()
                continue

            unresolved_question_items = [
                state["question_item"]
                for state in target_states.values()
                if state["status"] != "answered"
            ]

        if not unresolved_question_items:
            print("\n[Gemini thread] No unresolved questions remain.")
            gemini_queue.task_done()
            continue

        print(
            f"\n[Gemini thread] Processing sign triggered by: "
            f"{trigger_target}"
        )
        print(
            "[Gemini thread] Asking unresolved questions: "
            f"{[item['question'] for item in unresolved_question_items]}"
        )

        selected_frame_path, selected_crop_path, selection_path = (
            save_selection_for_target(
                run_dir=run_dir,
                target=trigger_target,
                state=trigger_state,
            )
        )

        print(f"[Gemini thread] Saved selection: {selection_path}")

        best_candidate = trigger_state["best_candidate"]
        detections = build_detection(best_candidate)

        # If this call exhausts its retries (see GeminiDirectionQA's
        # _create_completion), it raises. Without this try/except, that
        # exception would propagate out of this while-loop and kill
        # gemini_worker's thread permanently - every future submission
        # would then just sit in gemini_queue forever with nothing ever
        # processing it, and there would be no error visible anywhere
        # except a thread that silently stopped logging "[Gemini thread]"
        # lines. We catch it, log it, and let this one target be retried
        # on a later/better crop instead of losing the whole worker.
        try:
            result = run_multiview_from_detections(
                image_path=selected_frame_path,
                output_root=output_root,
                qa=qa,
                question_items=unresolved_question_items,
                detections=detections,
                crop_paths=[selected_crop_path],
                image_output_name=f"{video_name}_{trigger_target}",
            )
        except Exception as e:
            print(
                f"[Gemini thread] Call failed for target "
                f"{trigger_target!r}, will retry on a later crop: {e!r}"
            )

            with target_states_lock:
                if trigger_state["status"] != "answered":
                    trigger_state["status"] = "searching"
                    trigger_state["submitted_to_gemini"] = False
                    trigger_state["first_match_frame"] = None
                    trigger_state["first_match_time"] = None
                    trigger_state["matched_track_id"] = None
                    trigger_state["last_seen_frame"] = None
                    trigger_state["consecutive_misses"] = 0
                    trigger_state["best_candidate"] = None
                    trigger_state["candidate_observation_count"] = 0

            gemini_queue.task_done()
            continue

        if isinstance(result, dict) and "results" in result:
            returned_results = result["results"]
        elif isinstance(result, list):
            returned_results = result
        else:
            returned_results = [result]

        accepted_results = []

        with target_states_lock:
            question_to_state = {
                state["question_item"]["question"]: state
                for state in target_states.values()
            }

            for returned_result in returned_results:
                if not isinstance(returned_result, dict):
                    continue

                question = returned_result.get("question", "")
                predicted = returned_result.get("predicted", "unknown")
                matching_state = question_to_state.get(question)

                if matching_state is None:
                    print(
                        "[Gemini thread] Unrecognized returned question: "
                        f"{question!r}"
                    )
                    continue

                # Unknown on this crop is not final. A later crop may contain
                # the destination, so keep the target available for matching.
                if predicted == "unknown":
                    print(
                        f"[Gemini thread] {question} -> unknown "
                        "(not finalized yet)"
                    )
                    continue

                # A previous Gemini job may have answered this question while
                # the current request was in flight. Do not add duplicates.
                if matching_state["status"] == "answered":
                    print(
                        f"[Gemini thread] Duplicate answer ignored: {question}"
                    )
                    continue

                matching_state["status"] = "answered"
                matching_state["submitted_to_gemini"] = True
                accepted_results.append(returned_result)

                print(
                    f"[Gemini thread] Accepted: "
                    f"{question} -> {predicted}"
                )

            # If the OCR-triggering target was not answered by this crop, let
            # a later/better crop trigger it again.
            if trigger_state["status"] != "answered":
                trigger_state["status"] = "searching"
                trigger_state["submitted_to_gemini"] = False
                trigger_state["first_match_frame"] = None
                trigger_state["first_match_time"] = None
                trigger_state["matched_track_id"] = None
                trigger_state["last_seen_frame"] = None
                trigger_state["consecutive_misses"] = 0
                trigger_state["best_candidate"] = None
                trigger_state["candidate_observation_count"] = 0

                print(
                    f"[Gemini thread] Reopened target for later search: "
                    f"{trigger_target}"
                )

        if accepted_results:
            with results_lock:
                # Re-check by question before extending. This protects the
                # summary from duplicate answers produced by overlapping jobs.
                already_saved_questions = {
                    item.get("question")
                    for item in evaluation_results
                    if isinstance(item, dict)
                }

                fresh_results = [
                    item
                    for item in accepted_results
                    if item.get("question") not in already_saved_questions
                ]

                if fresh_results:
                    evaluation_results.extend(fresh_results)
                    save_stream_summary(
                        summary_path=summary_path,
                        video_path=live_video_path,
                        evaluation_results=evaluation_results,
                    )

                    for accepted_result in fresh_results:
                        result_queue.put({
                            "target": extract_target_from_question({
                                "question": accepted_result["question"]
                            }),
                            "result": [accepted_result],
                        })

        gemini_queue.task_done()

def ocr_worker(
    ocr_queue,
    detection_result_queue,
    gemini_queue,
    stop_event,
    ocr,
    target_states,
    tracks_lock,
    target_states_lock,
    save_debug_images,
    runtime_stats,
    runtime_stats_lock,
):
    """
    Background PaddleOCR worker.

    Important design choice:
    - It receives a saved crop from the detection thread.
    - It does NOT read the latest camera frame.
    - So even if the camera moves away later, OCR still uses the good crop.
    """
    # NOTE: this loop deliberately does NOT check stop_event, for the same
    # reason as gemini_worker above: it must finish every queued OCR job
    # (which can still produce a target match and feed Gemini) before
    # exiting. Shutdown is signaled purely by the None sentinel.
    while True:
        try:
            item = ocr_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if item is None:
            ocr_queue.task_done()
            break

        track = item["track"]
        track_id = item["track_id"]
        frame_idx = item["frame_idx"]
        crop = item["crop"]
        frame = item["frame"]
        bbox_xyxy = item["bbox_xyxy"]
        frame_path = item["frame_path"]
        crop_path = item["crop_path"]
        quality = item["quality"]

        start = time.perf_counter()
        ocr_lines = run_ocr(ocr, crop)
        ocr_seconds = time.perf_counter() - start
        print(f"OCR took {ocr_seconds:.3f}s")
        update_runtime_stats(
            runtime_stats,
            runtime_stats_lock,
            last_ocr_seconds=ocr_seconds,
            last_ocr_wall_time=time.time(),
        )
        increment_runtime_stat(
            runtime_stats,
            runtime_stats_lock,
            "completed_ocr_jobs",
        )

        ocr_text = get_detection_text(ocr_lines)
        avg_conf = get_avg_confidence(ocr_lines)

        with tracks_lock:
            # Only attach this OCR result if this is still the same track object.
            # This prevents the worker from blindly writing into a stale object
            # if the detection thread has moved on.
            track["ocr_status"] = "done"
            track["ocr_attempted"] = True
            track["ocr_attempt_count"] = track.get("ocr_attempt_count", 0) + 1
            track["last_ocr_frame"] = frame_idx
            track["last_ocr_quality"] = quality
            track["ocr_text"] = ocr_text
            track["ocr_lines"] = ocr_lines
            track["avg_ocr_confidence"] = avg_conf

            # Keep the strongest OCR evidence seen so far.
            # A later, clearer crop can correct an early weak/empty OCR result.
            if avg_conf >= track.get("best_ocr_confidence", 0.0) or (ocr_text and not track.get("best_ocr_text")):
                track["best_ocr_text"] = ocr_text
                track["best_ocr_lines"] = ocr_lines
                track["best_ocr_confidence"] = avg_conf
                track["best_ocr_frame_idx"] = frame_idx

            track["best_quality"] = max(track["best_quality"], quality)

        print(
            f"  OCR done | track {track_id} | "
            f"text={repr(ocr_text)} | conf={avg_conf:.3f}"
        )

        with tracks_lock:
            candidate_frame = (
                track["best_frame"].copy()
                if track.get("best_frame") is not None
                else frame.copy()
            )
            candidate_crop = (
                track["best_crop"].copy()
                if track.get("best_crop") is not None
                else crop.copy()
            )
            candidate_bbox = list(track.get("best_bbox_xyxy", bbox_xyxy))
            candidate_frame_idx = track.get("best_frame_idx", frame_idx)
            candidate_frame_path = track.get("best_frame_path", frame_path)
            candidate_crop_path = track.get("best_crop_path", crop_path)

        latest_match_text = ""

        with target_states_lock:
            for target, state in target_states.items():
                before_status = state["status"]

                update_target_with_detection(
                    state=state,
                    frame_idx=candidate_frame_idx,
                    frame_path=candidate_frame_path,
                    crop_path=candidate_crop_path,
                    bbox_xyxy=candidate_bbox,
                    ocr_text=ocr_text,
                    ocr_lines=ocr_lines,
                    avg_conf=avg_conf,
                    save_debug_images=save_debug_images,
                    frame=candidate_frame,
                    crop=candidate_crop,
                    track_id=track.get("stable_id", track["id"]),
                    visual_quality=track.get("best_quality", quality),
                )

                match_score = score_match(target, ocr_text) if ocr_text else 0

                if match_score >= state["match_threshold"]:
                    latest_match_text = f"{target} ({match_score:.1f})"

                if before_status != state["status"]:
                    print(f"  Target state changed: {target} -> {state['status']}")

        # NOTE: we intentionally do NOT send "boxes" here. item["latest_boxes"]
        # is a snapshot taken at the moment this job was queued (with an
        # "OCR pending" placeholder label baked in). By the time OCR finishes,
        # that snapshot is stale, and pushing it would clobber the fresher
        # boxes that detection_worker is already sending every processed
        # frame (with the correct OCR text read straight from the track).
        # Only the text fields need to come from this worker.
        put_latest_detection(
            detection_result_queue,
            {
                "ocr_text": ocr_text,
                "match_text": latest_match_text,
            },
        )

        ocr_queue.task_done()

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter / (area_a + area_b - inter)


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def normalized_center_distance(a, b):
    ax, ay = box_center(a)
    bx, by = box_center(b)

    dx = ax - bx
    dy = ay - by
    dist = (dx * dx + dy * dy) ** 0.5

    aw = max(1, a[2] - a[0])
    ah = max(1, a[3] - a[1])
    bw = max(1, b[2] - b[0])
    bh = max(1, b[3] - b[1])

    diag = max((aw * aw + ah * ah) ** 0.5, (bw * bw + bh * bh) ** 0.5)
    return dist / diag


def crop_quality(crop, bbox_xyxy=None, frame_shape=None, yolo_conf=1.0):
    """
    Estimate whether a sign crop is useful for OCR/Gemini.

    Higher is better. This is still a heuristic, but it now considers:
    - crop area
    - sharpness / blur
    - YOLO confidence
    - whether the sign is cut off by the frame edge
    """
    if crop is None or crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    area = crop.shape[0] * crop.shape[1]

    sharpness_factor = 1.0 + min(sharpness, 500.0) / 500.0
    confidence_factor = 0.5 + max(0.0, min(float(yolo_conf), 1.0))

    edge_factor = 1.0
    if bbox_xyxy is not None and frame_shape is not None:
        x1, y1, x2, y2 = bbox_xyxy
        h, w = frame_shape[:2]
        margin = 4

        touches_left = x1 <= margin
        touches_top = y1 <= margin
        touches_right = x2 >= w - margin
        touches_bottom = y2 >= h - margin
        edge_touches = sum([touches_left, touches_top, touches_right, touches_bottom])

        # Penalize likely partial signs. Do not make it zero, because sometimes
        # a partially visible crop can still provide useful OCR evidence.
        if edge_touches == 1:
            edge_factor = 0.65
        elif edge_touches >= 2:
            edge_factor = 0.40

    return area * sharpness_factor * confidence_factor * edge_factor


def update_best_track_view(
    track,
    crop,
    frame,
    bbox_xyxy,
    frame_idx,
    frame_path,
    crop_path,
    quality,
):
    """
    Keep the best visual evidence for this tracked sign.

    OCR may run on an earlier crop, but Gemini should use the best crop/frame
    observed during the track lifetime or matching window.
    """
    if quality > track.get("best_quality", 0.0):
        track["best_quality"] = quality
        track["best_crop"] = crop.copy()
        track["best_frame"] = frame.copy()
        track["best_bbox_xyxy"] = list(bbox_xyxy)
        track["best_frame_idx"] = frame_idx
        track["best_frame_path"] = frame_path
        track["best_crop_path"] = crop_path
        return True

    return False


def find_best_track(sign_tracks, bbox, frame_idx, used_track_ids, max_age=30):
    best_track = None
    best_score = -1

    for track in sign_tracks:
        if track["id"] in used_track_ids:
            continue

        age = frame_idx - track["last_seen_frame"]
        if age > max_age:
            continue

        iou = box_iou(bbox, track["bbox_xyxy"])
        center_dist = normalized_center_distance(bbox, track["bbox_xyxy"])

        # Stricter same-object rule.
        # Require overlap, but allow more center movement for partial views.
        if iou < 0.25:
            continue

        if center_dist > 0.50:
            continue

        score = iou - (0.25 * center_dist)

        if score > best_score:
            best_score = score
            best_track = track

    return best_track



def create_track_state(track_id, bbox_xyxy, frame_idx):
    """
    Create the OCR/best-view state associated with either a temporary ID
    or a confirmed ByteTrack ID.
    """
    return {
        "id": track_id,
        # Never changes, even if TEMP_n is promoted to BT_n.
        "stable_id": track_id,
        "bbox_xyxy": list(bbox_xyxy),
        "last_seen_frame": frame_idx,
        "ocr_status": "not_started",
        "ocr_attempted": False,
        "ocr_attempt_count": 0,
        "last_ocr_frame": -1,
        "last_ocr_quality": 0.0,
        "ocr_text": "",
        "ocr_lines": [],
        "avg_ocr_confidence": 0.0,
        "best_ocr_text": "",
        "best_ocr_lines": [],
        "best_ocr_confidence": 0.0,
        "best_ocr_frame_idx": -1,
        "best_quality": 0.0,
        "best_crop": None,
        "best_frame": None,
        "best_bbox_xyxy": list(bbox_xyxy),
        "best_frame_idx": frame_idx,
        "best_frame_path": None,
        "best_crop_path": None,
    }


def find_best_temporary_track(
    sign_tracks,
    bbox,
    frame_idx,
    used_track_ids,
    max_age=30,
):
    """
    Match an unconfirmed detection, or a newly confirmed ByteTrack detection,
    to a recent temporary track using the original geometric association rule.
    """
    best_track = None
    best_score = -1.0

    for track_id, track in sign_tracks.items():
        if not str(track_id).startswith("TEMP_"):
            continue

        if track_id in used_track_ids:
            continue

        age = frame_idx - track["last_seen_frame"]
        if age > max_age:
            continue

        iou = box_iou(bbox, track["bbox_xyxy"])
        center_dist = normalized_center_distance(
            bbox,
            track["bbox_xyxy"],
        )

        if iou < 0.25:
            continue

        if center_dist > 0.50:
            continue

        score = iou - (0.25 * center_dist)

        if score > best_score:
            best_score = score
            best_track = track

    return best_track


def promote_temporary_track(
    sign_tracks,
    temporary_track,
    bytetrack_key,
    bbox_xyxy,
    frame_idx,
):
    """
    Re-key the same track object from TEMP_n to BT_n.

    Mutating and reusing the same dictionary preserves references held by
    pending OCR jobs, so early OCR work and best-view history are not lost.
    """
    old_key = temporary_track["id"]

    if old_key in sign_tracks:
        del sign_tracks[old_key]

    temporary_track["id"] = bytetrack_key
    temporary_track["bbox_xyxy"] = list(bbox_xyxy)
    temporary_track["last_seen_frame"] = frame_idx
    sign_tracks[bytetrack_key] = temporary_track

    print(
        f"  Promoted temporary track {old_key} "
        f"to ByteTrack ID {bytetrack_key}"
    )

    return temporary_track



def cleanup_stale_tracks(
    sign_tracks,
    target_states,
    frame_idx,
    retention_frames=TRACK_RETENTION_FRAMES,
    max_active_tracks=MAX_ACTIVE_TRACKS,
):
    """
    Remove old track state that has not been observed recently.

    Tracks are retained when:
    - they were seen within retention_frames,
    - an OCR job is still pending for them, or
    - an open target-selection window is bound to their stable identity.

    Before deletion, large NumPy frame/crop references are cleared so memory
    can be reclaimed promptly.

    In addition to the normal age-based sweep, this also enforces a hard cap
    (max_active_tracks) on how many tracks can exist at once. Each track can
    hold a full-resolution frame + crop copy (multi-MB), and a low YOLO
    confidence threshold can spawn temporary tracks faster than the ~10s
    age-based sweep reclaims them, causing memory to climb steadily over a
    long run. If we're still over the cap after the normal sweep, evict the
    oldest unprotected tracks (temporary tracks first) immediately.
    """
    protected_stable_ids = {
        state.get("matched_track_id")
        for state in target_states.values()
        if state.get("status") == "window_open"
        and state.get("matched_track_id") is not None
    }

    stale_keys = []

    for track_key, track in sign_tracks.items():
        age = frame_idx - track.get("last_seen_frame", frame_idx)
        stable_id = track.get("stable_id", track_key)

        if age <= retention_frames:
            continue

        if track.get("ocr_status") == "pending":
            continue

        if stable_id in protected_stable_ids:
            continue

        stale_keys.append(track_key)

    for track_key in stale_keys:
        track = sign_tracks.pop(track_key)

        # Explicitly release the largest retained objects.
        track["best_frame"] = None
        track["best_crop"] = None

        print(
            f"  Removed stale track {track_key} "
            f"(last seen {frame_idx - track.get('last_seen_frame', frame_idx)} "
            f"frames ago)"
        )

    removed_count = len(stale_keys)

    overflow = len(sign_tracks) - max_active_tracks

    if overflow > 0:
        evictable = [
            (track_key, track)
            for track_key, track in sign_tracks.items()
            if track.get("ocr_status") != "pending"
            and track.get("stable_id", track_key) not in protected_stable_ids
        ]

        # Evict temporary (unconfirmed) tracks first, oldest-seen first,
        # since those are the most likely to be noise rather than a sign
        # we're actively trying to read.
        evictable.sort(
            key=lambda item: (
                0 if not str(item[0]).startswith("TEMP_") else -1,
                item[1].get("last_seen_frame", frame_idx),
            )
        )

        for track_key, track in evictable[:overflow]:
            sign_tracks.pop(track_key, None)
            track["best_frame"] = None
            track["best_crop"] = None
            removed_count += 1
            print(
                f"  Evicted track {track_key} to enforce "
                f"max_active_tracks={max_active_tracks} "
                f"(active={len(sign_tracks) + 1})"
            )

    return removed_count


def should_run_ocr_for_track(track, quality, frame_idx):
    """
    Decide whether this track deserves another OCR pass.

    Important idea:
    - New signs should get OCR.
    - Do not queue duplicate OCR while one is already pending.
    - Retry if OCR was empty/weak and the crop improved.
    - Retry if the crop becomes substantially better, even if OCR had text.
    """
    if track.get("ocr_status") == "pending":
        return False

    # First time seeing this tracked object.
    if not track["ocr_attempted"]:
        return True

    frames_since_ocr = frame_idx - track["last_ocr_frame"]

    # Hard cooldown: prevents one improving track from flooding the OCR queue.
    if frames_since_ocr < MIN_RETRY_GAP_FRAMES:
        return False

    # Guaranteed retry: give every track at least GUARANTEED_OCR_ATTEMPTS
    # passes regardless of how confident the first read looked. This is
    # what catches signs where OCR read part of the text cleanly (high
    # confidence) but missed another part entirely, since confidence-based
    # retry logic below would never fire for a "confident but incomplete"
    # result.
    if track.get("ocr_attempt_count", 0) < GUARANTEED_OCR_ATTEMPTS:
        return True

    last_ocr_quality = max(1.0, track.get("last_ocr_quality", 1.0))
    text = track.get("ocr_text", "")
    conf = track.get("avg_ocr_confidence", 0.0)

    # If OCR saw nothing, try again fairly soon once we have a better crop.
    if not text and frames_since_ocr >= 6 and quality > last_ocr_quality * 1.10:
        return True

    # If OCR confidence was weak, retry when the crop is at least 10% better.
    if (
        conf < 0.75
        and frames_since_ocr >= 6
        and quality > last_ocr_quality * 1.10
    ):
        return True

    # Even if OCR looked confident, retry when the crop is at least 10% better.
    # This helps when OCR correctly reads only part of a sign while far away,
    # then gets another chance as the camera moves closer.
    if (
        frames_since_ocr >= 6
        and quality > last_ocr_quality * 1.10
    ):
        return True
    

    return False




def detection_worker(
    frame_queue,
    detection_result_queue,
    ocr_queue,
    gemini_queue,
    stop_event,
    model,
    target_states,
    process_every_n_frames,
    display_width,
    display_height,
    save_debug_images,
    frames_dir,
    crops_dir,
    tracks_lock,
    target_states_lock,
    runtime_stats,
    runtime_stats_lock,
    track_retention_frames,
    yolo_conf,
):
    # ByteTrack assigns persistent IDs. We keep our own per-ID state for
    # OCR history, retry scheduling, and best-view selection.
    sign_tracks = {}
    next_temp_id = 1

    # NOTE: this loop deliberately does NOT check stop_event. Shutdown is
    # signaled purely by the None sentinel put onto frame_queue once the
    # camera loop stops, so any frame already queued still gets processed.
    while True:
        try:
            item = frame_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if item is None:
            frame_queue.task_done()
            break

        frame_idx = item["frame_idx"]
        raw_frame = item["raw_frame"]

        if frame_idx % process_every_n_frames != 0:
            frame_queue.task_done()
            continue

        start = time.perf_counter()
        results = model.track(
            source=raw_frame,
            persist=True,
            tracker=str(PROJECT_ROOT / "config" / "bytetrack_navigation.yaml"),
            conf=yolo_conf,
            iou=0.3,
            agnostic_nms=True,
            verbose=False,
        )
        yolo_seconds = time.perf_counter() - start
        print(f"YOLO took {yolo_seconds:.3f}s")
        update_runtime_stats(
            runtime_stats,
            runtime_stats_lock,
            last_yolo_seconds=yolo_seconds,
            last_detection_frame_idx=frame_idx,
            last_detection_wall_time=time.time(),
        )
        increment_runtime_stat(
            runtime_stats,
            runtime_stats_lock,
            "processed_detection_frames",
        )

        latest_boxes = []
        latest_ocr_text = ""
        latest_match_text = ""
        seen_track_ids = set()

        if results and len(results) > 0:
            result = results[0]
            h, w = raw_frame.shape[:2]

            print(f"\n[Detection thread] Frame {frame_idx}: {len(result.boxes)} detections")
            used_track_ids = set()

            for box_idx, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = raw_frame[y1:y2, x1:x2]

                crop_w = x2 - x1
                crop_h = y2 - y1

                # if crop_w < 80 or crop_h < 30:
                #     continue

                conf = float(box.conf[0])

                # ByteTrack may leave a detection unassigned for a frame.
                tracker_id = None
                if box.id is not None and len(box.id) > 0:
                    tracker_id = int(box.id[0].item())

                scale_x = display_width / w
                scale_y = display_height / h

                dx1 = int(x1 * scale_x)
                dy1 = int(y1 * scale_y)
                dx2 = int(x2 * scale_x)
                dy2 = int(y2 * scale_y)

                frame_path = frames_dir / f"frame_{frame_idx:04d}.jpg"
                crop_path = crops_dir / f"frame_{frame_idx:04d}_box_{box_idx:02d}.jpg"

                if save_debug_images:
                    cv2.imwrite(str(frame_path), raw_frame)
                    cv2.imwrite(str(crop_path), crop)

                quality = crop_quality(crop, bbox_xyxy=[x1, y1, x2, y2], frame_shape=raw_frame.shape, yolo_conf=conf)

                with tracks_lock:
                    if tracker_id is None:
                        # ByteTrack has not confirmed an identity yet. Reuse a
                        # geometrically matching temporary track, or create one
                        # immediately so OCR can start without waiting.
                        track = find_best_temporary_track(
                            sign_tracks=sign_tracks,
                            bbox=[x1, y1, x2, y2],
                            frame_idx=frame_idx,
                            used_track_ids=used_track_ids,
                        )

                        if track is None:
                            temporary_key = f"TEMP_{next_temp_id}"
                            next_temp_id += 1

                            track = create_track_state(
                                track_id=temporary_key,
                                bbox_xyxy=[x1, y1, x2, y2],
                                frame_idx=frame_idx,
                            )
                            sign_tracks[temporary_key] = track
                            print(
                                f"  Created temporary track "
                                f"{temporary_key} for box {box_idx}"
                            )
                        else:
                            track["bbox_xyxy"] = [x1, y1, x2, y2]
                            track["last_seen_frame"] = frame_idx

                    else:
                        bytetrack_key = f"BT_{tracker_id}"
                        track = sign_tracks.get(bytetrack_key)

                        if track is None:
                            # ByteTrack has just confirmed this object. First
                            # try to migrate an existing temporary state so its
                            # OCR history and best crop are preserved.
                            temporary_track = find_best_temporary_track(
                                sign_tracks=sign_tracks,
                                bbox=[x1, y1, x2, y2],
                                frame_idx=frame_idx,
                                used_track_ids=used_track_ids,
                            )

                            if temporary_track is not None:
                                track = promote_temporary_track(
                                    sign_tracks=sign_tracks,
                                    temporary_track=temporary_track,
                                    bytetrack_key=bytetrack_key,
                                    bbox_xyxy=[x1, y1, x2, y2],
                                    frame_idx=frame_idx,
                                )
                            else:
                                track = create_track_state(
                                    track_id=bytetrack_key,
                                    bbox_xyxy=[x1, y1, x2, y2],
                                    frame_idx=frame_idx,
                                )
                                sign_tracks[bytetrack_key] = track
                                print(
                                    f"  ByteTrack created confirmed "
                                    f"track {bytetrack_key}"
                                )
                        else:
                            track["bbox_xyxy"] = [x1, y1, x2, y2]
                            track["last_seen_frame"] = frame_idx

                    used_track_ids.add(track["id"])
                    seen_track_ids.add(track.get("stable_id", track["id"]))

                    best_view_updated = update_best_track_view(
                        track=track,
                        crop=crop,
                        frame=raw_frame,
                        bbox_xyxy=[x1, y1, x2, y2],
                        frame_idx=frame_idx,
                        frame_path=frame_path if save_debug_images else None,
                        crop_path=crop_path if save_debug_images else None,
                        quality=quality,
                    )

                    if best_view_updated:
                        print(
                            f"  track {track['id']} box {box_idx}: "
                            f"updated best view quality={quality:.1f}"
                        )

                    run_ocr_now = should_run_ocr_for_track(track, quality, frame_idx)

                    if run_ocr_now:
                        track["ocr_status"] = "pending"
                        track["ocr_attempted"] = True
                        track["last_ocr_frame"] = frame_idx

                        # Use the most recent detection result for display while OCR is pending.
                        pending_label = f"ID {track['id']}: OCR pending"

                        latest_boxes_for_job = latest_boxes.copy()
                        latest_boxes_for_job.append({
                            "bbox": [dx1, dy1, dx2, dy2],
                            "confidence": conf,
                            "ocr_text": pending_label,
                        })

                        put_ocr_job(
                            ocr_queue,
                            {
                                "track": track,
                                "track_id": track["id"],
                                "frame_idx": frame_idx,
                                "crop": (track["best_crop"].copy() if track["best_crop"] is not None else crop.copy()),
                                "frame": (track["best_frame"].copy() if track["best_frame"] is not None else raw_frame.copy()),
                                "bbox_xyxy": list(track.get("best_bbox_xyxy", [x1, y1, x2, y2])),
                                "frame_path": track.get("best_frame_path"),
                                "crop_path": track.get("best_crop_path"),
                                "quality": track.get("best_quality", quality),
                                "latest_boxes": latest_boxes_for_job,
                            },
                        )

                        print(f"  track {track['id']} box {box_idx}: queued for OCR quality={track.get('best_quality', quality):.1f}")
                    else:
                        print(
                            f"  track {track['id']} box {box_idx}: "
                            f"reused OCR='{track['ocr_text']}' "
                            f"conf={track['avg_ocr_confidence']:.3f} "
                            f"status={track['ocr_status']}"
                        )

                    ocr_text = track["ocr_text"]
                    avg_conf = track["avg_ocr_confidence"]
                    ocr_status = track["ocr_status"]
                    track_id = track["id"]

                if ocr_text:
                    box_label = f"ID {track_id}: {ocr_text}"
                    latest_ocr_text = ocr_text
                elif ocr_status == "pending":
                    box_label = f"ID {track_id}: OCR pending"
                else:
                    box_label = f"ID {track_id}"

                latest_boxes.append({
                    "bbox": [dx1, dy1, dx2, dy2],
                    "confidence": conf,
                    "ocr_text": box_label,
                })

                # If this track already has OCR text and a target window is open,
                # keep feeding better visual candidates to the target selector.
                # This lets Gemini receive the clearest view, not just the OCR trigger crop.
                if ocr_text:
                    with target_states_lock:
                        for target, state in target_states.items():
                            if state["status"] == "window_open":
                                update_target_with_detection(
                                    state=state,
                                    frame_idx=frame_idx,
                                    frame_path=track.get("best_frame_path"),
                                    crop_path=track.get("best_crop_path"),
                                    bbox_xyxy=track.get("best_bbox_xyxy", [x1, y1, x2, y2]),
                                    ocr_text=ocr_text,
                                    ocr_lines=track.get("ocr_lines", []),
                                    avg_conf=avg_conf,
                                    save_debug_images=save_debug_images,
                                    frame=(track["best_frame"] if track.get("best_frame") is not None else raw_frame),
                                    crop=(track["best_crop"] if track.get("best_crop") is not None else crop),
                                    track_id=track.get("stable_id", track["id"]),
                                    visual_quality=track.get("best_quality", quality),
                                )

        with target_states_lock:
            update_target_visibility(
                target_states=target_states,
                seen_track_ids=seen_track_ids,
                frame_idx=frame_idx,
            )

            for target, state in target_states.items():
                if (
                    state["status"] == "window_done"
                    and not state["submitted_to_gemini"]
                    and state["best_candidate"] is not None
                ):
                    print(f"\nSubmitting target to Gemini thread: {target}")
                    state["submitted_to_gemini"] = True
                    gemini_queue.put((target, state))

        with tracks_lock:
            with target_states_lock:
                removed_count = cleanup_stale_tracks(
                    sign_tracks=sign_tracks,
                    target_states=target_states,
                    frame_idx=frame_idx,
                    retention_frames=track_retention_frames,
                )

        if removed_count:
            increment_runtime_stat(
                runtime_stats,
                runtime_stats_lock,
                "removed_stale_tracks",
                removed_count,
            )

        update_runtime_stats(
            runtime_stats,
            runtime_stats_lock,
            active_tracks=len(sign_tracks),
        )

        put_latest_detection(
            detection_result_queue,
            {
                "boxes": latest_boxes,
                "ocr_text": latest_ocr_text,
                "match_text": latest_match_text,
            },
        )

        frame_queue.task_done()


# make sure to make save_debug_images to false when live
def run_live_pipeline(
    camera_index,
    output_root,
    qa_file,
    root,
    api_key_path,
    model_version,
    prompt_file,
    match_threshold=70,
    last_seen_grace_updates=8,
    track_retention_frames=TRACK_RETENTION_FRAMES,
    save_debug_images=True,
    process_every_n_frames=3,
    display_width=960,
    display_height=540,
    yolo_conf=0.15,
):
    output_root = Path(output_root)
    video_name = "live_camera"
    live_video_path = Path("live_camera.mp4")
    annotated_video_path = output_root / "live_annotated_output.mp4"

    run_dir = output_root
    global DEBUG_LOG_FILE
    global DEBUG_LOG_HANDLE
    global PERF_LOG_FILE

    runtime_stats = {
        "dropped_detection_frames": 0,
        "processed_detection_frames": 0,
        "completed_ocr_jobs": 0,
        "written_video_frames": 0,
        "removed_stale_tracks": 0,
        "dropped_writer_frames": 0,
        "active_tracks": 0,
        "last_yolo_seconds": 0.0,
        "last_ocr_seconds": 0.0,
        "last_video_write_seconds": 0.0,
    }
    runtime_stats_lock = threading.Lock()

    output_root.mkdir(parents=True, exist_ok=True)

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    DEBUG_LOG_FILE = run_dir / (
        "debug_" + timestamp_tag + ".txt"
    )
    PERF_LOG_FILE = run_dir / (
        "performance_" + timestamp_tag + ".csv"
    )

    # Open the debug log once for the whole run instead of per-message. See
    # the note in debug_log() for why this matters.
    DEBUG_LOG_HANDLE = open(DEBUG_LOG_FILE, "w", encoding="utf-8")
    DEBUG_LOG_HANDLE.write("===== LIVE PIPELINE DEBUG LOG =====\n\n")
    DEBUG_LOG_HANDLE.flush()

    frames_dir = run_dir / "stream_frames"
    crops_dir = run_dir / "stream_crops"

    summary_path = output_root / "stream_summary.json"
    evaluation_results = []

    print(f"run_dir = {run_dir}")
    print(f"output_root = {output_root}")

    (run_dir / "selected").mkdir(parents=True, exist_ok=True)

    if save_debug_images:
        frames_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

    save_stream_summary(
        summary_path=summary_path,
        video_path=live_video_path,
        evaluation_results=evaluation_results,
    )

    question_items = load_live_qa(qa_file)

    target_states = create_target_states(
        question_items=question_items,
        match_threshold=match_threshold,
        last_seen_grace_updates=last_seen_grace_updates,
    )

    print("\n==============================")
    print("LIVE CAMERA PIPELINE - BYTETRACK + LAST-SEEN WINDOW")
    print(
        "Tracker config: "
        f"{PROJECT_ROOT / 'config' / 'bytetrack_navigation.yaml'}"
    )
    print("==============================")
    print(f"Camera index: {camera_index}")
    print(f"Targets: {list(target_states.keys())}")
    print(f"Questions: {[q['question'] for q in question_items]}")
    print(
        f"Track retention: {track_retention_frames} camera frames "
        f"(~{track_retention_frames / 30.0:.1f}s at 30 FPS)"
    )
    print(f"YOLO confidence threshold: {yolo_conf}")
    print(f"Max active tracks: {MAX_ACTIVE_TRACKS}")
    print("Press q to stop live capture.")
    print("==============================\n")

    print("Loading YOLOWorld...")
    model = YOLOWorld("yolov8s-world.pt")

    classes = [
        "navigation sign",
        "directional sign",
        "wayfinding sign",
        "sign with arrow",
        "directional arrow sign",
        "hallway directional sign",
        "exit sign",
        "",
    ]

    model.set_classes(classes)
    print("YOLOWorld loaded.")

    print("Loading PaddleOCR...")
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=True,
        show_log=False,
    )

    qa = GeminiDirectionQA(
        root=root,
        api_key_path=api_key_path,
        model_version=model_version,
        prompt_file=prompt_file,
    )

    # -----------------------------
    # Thread communication
    # -----------------------------
    frame_queue = queue.Queue(maxsize=2)
    ocr_queue = queue.Queue(maxsize=4)
    detection_result_queue = queue.Queue(maxsize=2)
    gemini_queue = queue.Queue()
    result_queue = queue.Queue()

    stop_event = threading.Event()
    tracks_lock = threading.Lock()
    target_states_lock = threading.Lock()
    results_lock = threading.Lock()
    gemini_thread = threading.Thread(
        target=gemini_worker,
        args=(
            gemini_queue,
            result_queue,
            stop_event,
            run_dir,
            output_root,
            qa,
            summary_path,
            live_video_path,
            evaluation_results,
            results_lock,
            video_name,
            target_states,
            target_states_lock,
        ),
        daemon=True,
    )

    gemini_thread.start()

    ocr_thread = threading.Thread(
        target=ocr_worker,
        args=(
            ocr_queue,
            detection_result_queue,
            gemini_queue,
            stop_event,
            ocr,
            target_states,
            tracks_lock,
            target_states_lock,
            save_debug_images,
            runtime_stats,
            runtime_stats_lock,
        ),
        daemon=True,
    )

    ocr_thread.start()

    detection_thread = threading.Thread(
        target=detection_worker,
        args=(
            frame_queue,
            detection_result_queue,
            ocr_queue,
            gemini_queue,
            stop_event,
            model,
            target_states,
            process_every_n_frames,
            display_width,
            display_height,
            save_debug_images,
            frames_dir,
            crops_dir,
            tracks_lock,
            target_states_lock,
            runtime_stats,
            runtime_stats_lock,
            track_retention_frames,
            yolo_conf,
        ),
        daemon=True,
    )

    detection_thread.start()

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)


    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    # This is both the container's declared frame rate AND the rate we pace
    # writes to, in wall-clock time (see the pacing loop below). Keeping
    # these the same value is what makes played-back duration match real
    # elapsed capture time, instead of the output silently speeding up
    # whenever the pipeline falls behind and writes fewer frames than the
    # declared fps expects.
    OUTPUT_VIDEO_FPS = 20.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(annotated_video_path),
        fourcc,
        OUTPUT_VIDEO_FPS,
        (display_width, display_height),
    )

    if not video_writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {annotated_video_path}")

    # Bounded so a slow encoder cannot retain unlimited full video frames
    # and exhaust RAM during a long live run.
    writer_queue = queue.Queue(maxsize=WRITER_QUEUE_MAXSIZE)
    writer_thread = threading.Thread(
        target=video_writer_worker,
        args=(
            writer_queue,
            video_writer,
            runtime_stats,
            runtime_stats_lock,
        ),
        daemon=True,
    )
    writer_thread.start()

    frame_idx = 0
    latest_boxes = []

    capture_start_wall = time.time()
    last_metrics_wall = capture_start_wall
    last_metrics_frame_idx = 0
    last_loop_wall = time.perf_counter()
    smoothed_loop_fps = 0.0

    # Wall-clock video pacing. The container is stamped at OUTPUT_VIDEO_FPS,
    # so played-back duration = (frames written / OUTPUT_VIDEO_FPS). Without
    # this, one frame gets pushed per loop iteration regardless of how much
    # real time that iteration took - so if the pipeline falls behind (GPU
    # contention, GC pauses, etc.) and produces fewer frames than
    # OUTPUT_VIDEO_FPS expects for that stretch of real time, the resulting
    # video plays back sped up relative to what was actually recorded. To
    # keep video duration honest, we track how many output frames "should"
    # have been written by now given real elapsed time, and write that many
    # copies of the current frame (duplicating when we're behind, skipping
    # writes entirely when we're ahead). MAX_PACING_CATCHUP caps how many
    # duplicate frames a single stall can inject, so a multi-second freeze
    # doesn't cause a multi-second burst of identical frames.
    output_frame_duration = 1.0 / OUTPUT_VIDEO_FPS
    MAX_PACING_CATCHUP = 10
    output_frames_written = 0

    latest_ocr_text = ""
    latest_match_text = ""
    latest_gemini_text = ""
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read camera frame.")
            break

        raw_frame = frame.copy()
        display_frame = cv2.resize(frame, (display_width, display_height))

        put_latest_frame(
            frame_queue,
            {
                "frame_idx": frame_idx,
                "raw_frame": raw_frame.copy(),
            },
            runtime_stats=runtime_stats,
            runtime_stats_lock=runtime_stats_lock,
        )

        try:
            while True:
                gemini_update = result_queue.get_nowait()
                target = gemini_update["target"]
                results = gemini_update["result"]

                if results:
                    predicted = results[0].get("predicted", "unknown")
                    latest_gemini_text = f"{target}: {predicted}"
                    print(f"\nGemini result received: {latest_gemini_text}")

                result_queue.task_done()

        except queue.Empty:
            pass

        try:
            while True:
                detection_update = detection_result_queue.get_nowait()

                if "boxes" in detection_update:
                    latest_boxes = detection_update["boxes"]
                if "ocr_text" in detection_update:
                    latest_ocr_text = detection_update["ocr_text"]
                if "match_text" in detection_update:
                    latest_match_text = detection_update["match_text"]

                detection_result_queue.task_done()

        except queue.Empty:
            pass        

        now_perf = time.perf_counter()
        loop_dt = max(1e-6, now_perf - last_loop_wall)
        instant_loop_fps = 1.0 / loop_dt
        last_loop_wall = now_perf

        if smoothed_loop_fps <= 0:
            smoothed_loop_fps = instant_loop_fps
        else:
            smoothed_loop_fps = (
                0.90 * smoothed_loop_fps
                + 0.10 * instant_loop_fps
            )

        now_wall = time.time()
        if now_wall - last_metrics_wall >= 1.0:
            elapsed = max(1e-6, now_wall - last_metrics_wall)
            capture_fps = (
                (frame_idx - last_metrics_frame_idx) / elapsed
            )

            stats_snapshot = snapshot_runtime_stats(
                runtime_stats,
                runtime_stats_lock,
            )

            perf_row = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "elapsed_seconds": round(now_wall - capture_start_wall, 3),
                "frame_idx": frame_idx,
                "capture_fps_1s": round(capture_fps, 3),
                "gui_loop_fps_smoothed": round(smoothed_loop_fps, 3),
                "frame_queue_size": frame_queue.qsize(),
                "ocr_queue_size": ocr_queue.qsize(),
                "detection_result_queue_size": detection_result_queue.qsize(),
                "gemini_queue_size": gemini_queue.qsize(),
                "result_queue_size": result_queue.qsize(),
                "writer_queue_size": writer_queue.qsize(),
                "active_tracks": stats_snapshot.get("active_tracks", 0),
                "processed_detection_frames": stats_snapshot.get(
                    "processed_detection_frames", 0
                ),
                "completed_ocr_jobs": stats_snapshot.get(
                    "completed_ocr_jobs", 0
                ),
                "written_video_frames": stats_snapshot.get(
                    "written_video_frames", 0
                ),
                "dropped_detection_frames": stats_snapshot.get(
                    "dropped_detection_frames", 0
                ),
                "removed_stale_tracks": stats_snapshot.get(
                    "removed_stale_tracks", 0
                ),
                "dropped_writer_frames": stats_snapshot.get(
                    "dropped_writer_frames", 0
                ),
                "last_yolo_seconds": round(
                    stats_snapshot.get("last_yolo_seconds", 0.0), 4
                ),
                "last_ocr_seconds": round(
                    stats_snapshot.get("last_ocr_seconds", 0.0), 4
                ),
                "last_video_write_seconds": round(
                    stats_snapshot.get("last_video_write_seconds", 0.0), 4
                ),
            }

            try:
                import psutil

                process_rss_mb = round(
                    psutil.Process().memory_info().rss / (1024 * 1024), 1
                )
                system_available_mb = round(
                    psutil.virtual_memory().available / (1024 * 1024), 1
                )
                perf_row["process_rss_mb"] = process_rss_mb
                perf_row["system_available_mb"] = system_available_mb
            except ImportError:
                pass

            append_perf_row(perf_row)

            print(
                "[PERF] "
                f"capture={capture_fps:.1f} FPS | "
                f"gui={smoothed_loop_fps:.1f} FPS | "
                f"frameQ={frame_queue.qsize()} | "
                f"ocrQ={ocr_queue.qsize()} | "
                f"geminiQ={gemini_queue.qsize()} | "
                f"writerQ={writer_queue.qsize()} | "
                f"tracks={stats_snapshot.get('active_tracks', 0)} | "
                f"removed={stats_snapshot.get('removed_stale_tracks', 0)} | "
                f"dropDet={stats_snapshot.get('dropped_detection_frames', 0)} | "
                f"dropVid={stats_snapshot.get('dropped_writer_frames', 0)}"
                + (
                    f" | rssMB={perf_row['process_rss_mb']}"
                    if "process_rss_mb" in perf_row
                    else ""
                )
            )

            last_metrics_wall = now_wall
            last_metrics_frame_idx = frame_idx

        cv2.putText(
            display_frame,
            "LIVE Navigation Pipeline",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            f"Frame: {frame_idx}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            f"Gemini: {latest_gemini_text}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            f"OCR: {latest_ocr_text[:60]}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            f"Best match: {latest_match_text}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            (
                f"Perf: {smoothed_loop_fps:.1f} FPS | "
                f"FQ {frame_queue.qsize()} | "
                f"OQ {ocr_queue.qsize()} | "
                f"WQ {writer_queue.qsize()} | "
                f"VD {snapshot_runtime_stats(runtime_stats, runtime_stats_lock).get('dropped_writer_frames', 0)}"
            ),
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        for item in latest_boxes:
            x1, y1, x2, y2 = item["bbox"]
            conf = item["confidence"]
            ocr_text = item["ocr_text"]

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"sign {conf:.2f}"
            if ocr_text:
                label += f" | {ocr_text[:25]}"

            cv2.putText(
                display_frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )        

        # Handed off to video_writer_worker instead of writing inline here -
        # see that function for why. display_frame isn't touched again after
        # this point in the loop (nothing mutates it in-place), so it's safe
        # to push the same array reference more than once below.
        #
        # Wall-clock pacing: figure out how many output frames "should"
        # exist by now given real elapsed time, and top up to that count.
        # If we're behind (pipeline was slow), this duplicates the current
        # frame a few times so played-back duration keeps matching real
        # time instead of compressing. If we're ahead, it writes zero
        # frames this iteration.
        pacing_now = time.time()
        frames_due = int(
            (pacing_now - capture_start_wall) / output_frame_duration
        ) - output_frames_written
        frames_due = max(0, min(frames_due, MAX_PACING_CATCHUP))

        for _ in range(frames_due):
            put_video_frame(
                writer_queue=writer_queue,
                frame=display_frame,
                runtime_stats=runtime_stats,
                runtime_stats_lock=runtime_stats_lock,
            )
            output_frames_written += 1

        cv2.imshow("Live Navigation Pipeline", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nStopping live capture (finishing pending OCR/Gemini work)...")
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("Finishing annotated video file...")
    # A blocking put is intentional for the shutdown sentinel. The writer
    # drains enough space and then receives None.
    writer_queue.put(None)
    writer_thread.join()
    video_writer.release()

    print(f"Saved annotated video: {annotated_video_path}")

    # Graceful shutdown. Order matters here:
    # 1) Tell detection_worker there are no more frames, and wait for it to
    #    fully exit. It may still queue OCR jobs based on the last frames it
    #    processed, so we must not touch ocr_queue until it's done.
    print("Stopping detection thread (no more incoming frames)...")
    put_latest_frame(
        frame_queue,
        None,
        runtime_stats=runtime_stats,
        runtime_stats_lock=runtime_stats_lock,
    )
    detection_thread.join()

    # 2) Now it's safe to drain whatever OCR jobs are left. These can still
    #    produce a fresh target match, so we let ocr_thread run to
    #    completion before deciding what needs to go to Gemini.
    print("Draining OCR queue...")
    ocr_queue.put(None)
    ocr_thread.join()

    # 3) Any target whose last-seen collection window was still open when the user
    #    pressed "q" would otherwise be silently dropped, since window
    #    closing normally only happens inside the (now-stopped) frame loop.
    #    Force those windows closed and submit anything with a real
    #    candidate to Gemini before we shut that queue down too.
    with target_states_lock:
        for target, state in target_states.items():
            if state["status"] == "window_open":
                state["status"] = "window_done"
                print(f"  Force-closed matching window for target: {target}")

            if (
                state["status"] == "window_done"
                and not state["submitted_to_gemini"]
                and state["best_candidate"] is not None
            ):
                print(f"  Submitting target to Gemini thread: {target}")
                state["submitted_to_gemini"] = True
                gemini_queue.put((target, state))

    # 4) Only now signal Gemini to stop, and actually wait for it to finish
    #    (including any in-flight API call) rather than a short timeout.
    #    gemini_thread is a daemon thread, so without this join, Python
    #    would kill it mid-request the instant this function returns.
    print("Waiting for Gemini thread to finish pending requests...")
    gemini_queue.put((None, None))
    gemini_thread.join()

    stop_event.set()

    for target, state in target_states.items():
        question_item = state["question_item"]

        if state["status"] != "answered":
            print(f"\nNo match found for target: {target}. Marking unknown.")

            unknown_result = make_unknown_result(question_item)
            evaluation_results.append(unknown_result)

            save_stream_summary(
                summary_path=summary_path,
                video_path=live_video_path,
                evaluation_results=evaluation_results,
            )

    with results_lock:
        summary = save_stream_summary(
            summary_path=summary_path,
            video_path=live_video_path,
            evaluation_results=evaluation_results,
        )

    print("\nLIVE PIPELINE COMPLETE.")
    print(f"Saved summary: {summary_path}")
    print(f"Saved performance metrics: {PERF_LOG_FILE}")

    if DEBUG_LOG_HANDLE is not None:
        DEBUG_LOG_HANDLE.close()
        DEBUG_LOG_HANDLE = None

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera-index",
        type=int,
        default=2,
        help="Camera index",
    )

    parser.add_argument(
        "--output-root",
        default="outputs/live_pipeline_builtin_tracker",
        help="Output root",
    )

    parser.add_argument(
        "--qa-file",
        default="src/understand/qa_test_set/rpi_video_test_set.json",
        help="QA JSON file",
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

    parser.add_argument(
        "--process-every-n-frames",
        type=int,
        default=2,
        help="Run YOLO/ByteTrack every N frames.",
    )

    parser.add_argument(
        "--last-seen-grace-updates",
        type=int,
        default=8,
        help=(
            "Number of processed detector updates to wait after the matched "
            "track disappears before sending its best candidate to Gemini."
        ),
    )

    parser.add_argument(
        "--track-retention-frames",
        type=int,
        default=TRACK_RETENTION_FRAMES,
        help=(
            "Delete track state after this many original camera frames "
            "without seeing it, unless OCR is pending or a target window "
            "still depends on it."
        ),
    )

    parser.add_argument(
        "--display-width",
        type=int,
        default=960,
        help="Display width.",
    )

    parser.add_argument(
        "--display-height",
        type=int,
        default=540,
        help="Display height.",
    )

    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.15,
        help=(
            "YOLO-World detection confidence threshold. Raised from the "
            "original 0.05 default - a very low threshold lets through a "
            "lot of noisy low-confidence boxes, which spawns extra "
            "temporary tracks (each holding a full-resolution frame+crop "
            "copy) faster than age-based cleanup can reclaim them."
        ),
    )

    args = parser.parse_args()

    run_live_pipeline(
        camera_index=args.camera_index,
        output_root=PROJECT_ROOT / args.output_root,
        qa_file=PROJECT_ROOT / args.qa_file,
        root=args.root,
        api_key_path=args.api_key_path,
        model_version=args.model_version,
        prompt_file=args.prompt_file,
        save_debug_images=False,
        process_every_n_frames=args.process_every_n_frames,
        last_seen_grace_updates=args.last_seen_grace_updates,
        track_retention_frames=args.track_retention_frames,
        display_width=args.display_width,
        display_height=args.display_height,
        yolo_conf=args.yolo_conf,
    )

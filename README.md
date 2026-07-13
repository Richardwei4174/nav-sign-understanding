Navigation Sign Understanding Pipeline

This repository contains three different pipelines for understanding indoor navigation signs using YOLO-World, MobileSAM, OpenCV Perspective Rectification, PaddleOCR, and Gemini.

Pipeline Overview
1. Image Pipeline
      ↓
Single image → Gemini

2. Offline Video Pipeline
      ↓
Entire video → Best frame → Gemini

3. Simulated Live Video Pipeline
      ↓
Process video while playing
Stop once destination is found
→ Gemini


# 1. Image Pipeline
Purpose

Process one image (or an entire image folder) and answer navigation questions.

Run
python -m src.pipeline.run_multiview_pipeline

or

python -m src.pipeline.run_multiview_pipeline \
    --image data/test_images/IMG_0001.JPG
Pipeline
Image
    │
    ▼
YOLO-World
    │
    ▼
Crop navigation signs
    │
    ▼
MobileSAM
    │
    ▼
Perspective Rectification
    │
    ▼
Annotated Original Image
    │
    ▼
Gemini Multiview Prompt
    │
    ▼
QA Results
Main Python File
src/pipeline/run_multiview_pipeline.py
Output
outputs/pipeline/

    IMG_0001/

        annotated_original.jpg

        crops/

        rectified/

        gemini_results.json

# 2. Offline Video Pipeline

The offline pipeline processes an entire video before sending a single selected frame to Gemini.

This is primarily used for benchmarking and controlled experiments.

Step 1 — YOLO + OCR

Run

python -m src.preprocess.yolo_crop_ocr_video
Purpose
Video
    │
    ▼
YOLO-World
    │
    ▼
Crop each detected sign
    │
    ▼
PaddleOCR
    │
    ▼
ocr_results.json

Python File
src/preprocess/yolo_crop_ocr_video.py
Output
outputs/yolo_ocr_video/

    frames/

    crops/

    ocr_results.json
Step 2 — Best Frame Selection

Run

python -m src.preprocess.select_best_frame
Purpose
ocr_results.json
        │
        ▼
RapidFuzz
        │
        ▼
Find destination
        │
        ▼
Search only 1-second window
        │
        ▼
Best frame
        │
        ▼
selection.json
Matching Strategy

Numeric destinations

5200
1105
2117

require exact numeric matches.

Text destinations use fuzzy matching.

Final score

RapidFuzz Match Score
+
Average OCR Confidence
Python File
src/preprocess/select_best_frame.py
Output
outputs/frame_selection/

    candidates/

    selected/

    selection.json
Step 3 — Multiview Backend

Run

python -m src.pipeline.run_video_multiview_pipeline
Purpose

This stage loads selection.json and calls run_multiview_from_detections(...) from 
src/pipeline/run_multiview_pipeline.py

The shared backend performs

Selected Frame
        │
        ▼
MobileSAM
        │
        ▼
Perspective Rectification
        │
        ▼
Annotated Original Image
        │
        ▼
Gemini Multiview
        │
        ▼
QA Results
Python File
src/pipeline/run_video_multiview_pipeline.py
Output
outputs/video_multiview_pipeline/

    IMG_3772/

        annotated_original.jpg

        crops/

        rectified/

        gemini_results.json


Complete Offline Video Pipeline

Run these three commands in order

python -m src.preprocess.yolo_crop_ocr_video

↓

python -m src.preprocess.select_best_frame

↓

python -m src.pipeline.run_video_multiview_pipeline

Pipeline

Video
    │
    ▼
YOLO + OCR
    │
    ▼
ocr_results.json
    │
    ▼
Frame Selection
    │
    ▼
selection.json
    │
    ▼
Shared Multiview Backend
    │
    ▼
Gemini
# 3. Simulated Live Video Pipeline

The simulated live pipeline processes the video while it is being read.

Instead of waiting for the entire video to finish, the pipeline continuously searches for the requested destination.

Once a match is found, it immediately performs multiview reasoning.

This simulates how a future live camera system will operate.

Run
python -m src.pipeline.run_stream_video_pipeline
Pipeline
Video
    │
    ▼
YOLO-World
    │
    ▼
PaddleOCR
    │
    ▼
RapidFuzz
    │
    ▼
Maintain Best Candidate
    │
    ▼
Destination Found?
    │
    ├── No
    │       │
    │       ▼
    │   Continue Reading Video
    │
    └── Yes
            │
            ▼
    Search 1-second Window
            │
            ▼
    Best Candidate
            │
            ▼
run_multiview_from_detections(...)
            │
            ▼
MobileSAM
            │
            ▼
Perspective Rectification
            │
            ▼
Annotated Original Image
            │
            ▼
Gemini
            │
            ▼
QA Results
Python File
src/pipeline/run_stream_video_pipeline.py
Output
outputs/stream_video_pipeline/

    IMG_3772/

        stream_frames/

        stream_crops/

        candidates/

        selected/

        annotated_original.jpg

        crops/

        rectified/

        gemini_results.json
Shared Backend

Both the image pipeline and the two video pipelines reuse the same multiview backend.

run_multiview_from_detections(...)

located in

src/pipeline/run_multiview_pipeline.py

This backend performs

MobileSAM
↓

Perspective Rectification

↓

Annotated Original Image

↓

Gemini Multiview Prompt

↓

QA Evaluation

Because all pipelines share this backend, improvements automatically apply to image, offline video, and simulated live video processing.

Current Project Architecture
                     +------------------------+
                     | run_multiview_pipeline |
                     |  (Shared Backend)      |
                     +-----------+------------+
                                 ^
                                 |
                  run_multiview_from_detections()
                                 ^
          +----------------------+----------------------+
          |                                             |
          |                                             |
+---------+----------+                     +------------+-------------+
| Image Pipeline     |                     | Video Pipelines          |
+--------------------+                     +--------------------------+
| run_multiview_     |                     | Offline                  |
| pipeline.py        |                     | yolo_crop_ocr_video.py   |
|                    |                     | select_best_frame.py     |
| Image              |                     | run_video_multiview_     |
| → YOLO             |                     | pipeline.py              |
| → MobileSAM        |                     |                          |
| → Gemini           |                     | Simulated Live           |
+--------------------+                     | run_stream_video_        |
                                           | pipeline.py              |
                                           +--------------------------+
Future Work

The simulated live pipeline is designed to transition directly into a true live system.

The only planned change is replacing

cv2.VideoCapture("IMG_3772.MOV")

with

cv2.VideoCapture(0)

(or another live camera source).

The remainder of the pipeline—including YOLO detection, OCR, frame selection, MobileSAM rectification, and Gemini multiview reasoning—can remain unchanged, enabling real-time navigation sign understanding using a live camera feed
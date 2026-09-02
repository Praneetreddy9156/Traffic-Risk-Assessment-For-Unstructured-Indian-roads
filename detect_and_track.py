"""
STEP 3 — Vehicle Detection + Tracking on Traffic Videos

What this script does:
  - Loads the fine-tuned detector (models/fgvd_finetuned/weights/best.pt)
  - Loops through every video in the configured video folder
  - Detects vehicles in every frame
  - Tracks each vehicle across frames using BoT-SORT
    (BoT-SORT has Camera Motion Compensation, which suits moving/vlog footage)
  - Saves one CSV per video with all trajectories
  - Records each video's real fps AND frame size to outputs/video_meta.json

Output:
  outputs/csv/tracking/<video_id>.csv
      frame, track_id, class_id, class_name, cx, cy, width, height, confidence
  outputs/video_meta.json
      {video_id: {fps, width, height, frames}}

WHAT CHANGED AND WHY
--------------------
1. CLASS NAMES ARE READ FROM THE MODEL, NOT HARD-CODED.
   This file used to carry its own CLASS_NAMES dict that had to "match
   finetune.py order exactly", and it listed six classes including two the
   detector was never trained on. Any retrain that changes the taxonomy would
   silently mislabel every row. The names now come from the loaded model, so
   they cannot disagree with the weights.

2. FRAME SIZE AND FPS ARE NOW WRITTEN OUT.
   The script always read fps and frame size, then printed them and threw them
   away. Everything downstream needs both - fps to convert frame gaps into
   seconds, and frame WIDTH to express distances as a fraction of the frame so
   the risk features stop depending on video resolution (see pipeline_core).
   Recovering this after the fact required re-opening every source video.

3. DETECTION SETTINGS COME FROM pipeline_core.DETECTION,
   which the dashboard's live upload path also uses, so an uploaded clip is
   measured exactly like the training corpus rather than at different
   thresholds.

4. Tracking CSVs go in their own subdirectory, away from the derived analysis
   CSVs that used to sit beside them and get swept up by "*.csv" globs.

All error handling is built in so one bad video won't stop the rest.

IMPORTANT: Run from project root with venv activated
  python src/detect_and_track.py
"""

import os

# Must be set BEFORE torch is imported (ultralytics pulls it in below).
#
# CUDA eagerly loads every kernel module at init, committing ~2.6-3.4 GB of
# virtual address space per process. This machine has a 2 GB page file, so the
# commit limit is ~33 GB with ~8 GB free — enough for two detection workers.
# Four and six both died with:
#     OSError: [WinError 1455] The paging file is too small for this operation
# before torch had even finished importing.
#
# Lazy loading defers those modules until first use and drops per-process commit
# to ~1.3 GB, which is what makes 5-6 parallel workers possible here. Enlarging
# the page file would be the other fix, but that needs admin rights.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import csv
import sys
import json
import traceback
from pathlib import Path

import pipeline_core
from pipeline_core import PATHS, DETECTION, safe_video_id

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_PATH = PATHS["yolo_weights"]
VIDEO_DIR  = PATHS["video_dir"]
OUTPUT_DIR = PATHS["tracking_dir"]
META_PATH  = PATHS["video_meta"]

# Detection/tracking settings live in pipeline_core so the dashboard shares them
CONF_THRESHOLD = DETECTION["conf"]
IOU_THRESHOLD  = DETECTION["iou"]
IMG_SIZE       = DETECTION["imgsz"]
TRACKER        = DETECTION["tracker"]
MIN_TRACK_LEN  = DETECTION["min_track_len"]

# Existing CSVs are skipped by default so an interrupted run can resume.
# Set True after retraining the detector - otherwise the old model's results
# would be kept and silently mixed with the new model's.
FORCE_REPROCESS = False

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]


# ─────────────────────────────────────────────────────────────
# WINDOWS MULTIPROCESSING GUARD — required on Windows
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Import here (inside guard) to avoid Windows multiprocessing issues
    import argparse
    from ultralytics import YOLO
    import cv2

    # ── Sharding ──
    # One process cannot saturate this machine: YOLOv8n inference uses ~163MB of
    # 8GB VRAM and a single core's worth of video decoding, leaving the GPU at
    # ~25% and most cores idle. The real cost is CPU-side decoding and tracker
    # association, which parallelises perfectly because every video is
    # independent (the same reason persist=False is correct).
    #
    # Each shard takes every Nth video, so results are byte-identical to a
    # single run - the work is only distributed, never changed.
    ap = argparse.ArgumentParser(description="Detect + track vehicles across videos")
    ap.add_argument("--shard", type=int, default=0, help="this worker's index")
    ap.add_argument("--num-shards", type=int, default=1, help="total workers")
    ap.add_argument("--merge-meta", action="store_true",
                    help="merge per-shard metadata files and exit")
    cli = ap.parse_args()

    # Sharded workers each write their own metadata file. Sharing one path would
    # race: several processes rewriting the same JSON can interleave and leave a
    # truncated, unparseable file.
    if cli.num_shards > 1:
        META_PATH = META_PATH.replace(".json", f".shard{cli.shard}.json")

    if cli.merge_meta:
        merged = {}
        base = PATHS["video_meta"]
        for part in sorted(Path(os.path.dirname(base)).glob("video_meta.shard*.json")):
            try:
                with open(part) as f:
                    merged.update(json.load(f))
            except Exception as e:
                print(f"  skipping {part.name}: {e}")
        if os.path.exists(base):
            try:
                with open(base) as f:
                    existing = json.load(f)
                existing.update(merged)
                merged = existing
            except Exception:
                pass
        with open(base, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"merged metadata for {len(merged)} videos -> {base}")
        for part in Path(os.path.dirname(base)).glob("video_meta.shard*.json"):
            part.unlink()
        sys.exit(0)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("   Please run finetune.py first!")
        sys.exit(1)

    if not os.path.exists(VIDEO_DIR):
        print(f"❌ Video folder not found at: {VIDEO_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)

    all_videos = sorted(
        f for f in Path(VIDEO_DIR).iterdir() if f.suffix in VIDEO_EXTENSIONS
    )
    # Stride-based split keeps long and short videos spread evenly across
    # workers; a contiguous split would hand one worker all the 4K clips.
    video_files = all_videos[cli.shard::cli.num_shards]

    if not video_files:
        print(f"❌ No video files found in: {VIDEO_DIR}")
        print(f"   Looking for: {VIDEO_EXTENSIONS}")
        sys.exit(1)

    # ── Load model ONCE (not inside the loop — saves time) ──
    print("\nLoading detector...")
    model = YOLO(MODEL_PATH)
    class_names = dict(model.names)   # authoritative: comes with the weights
    print("✅ Model loaded")

    print("=" * 62)
    print("  STEP 3 — Vehicle Detection + Tracking")
    print("=" * 62)
    print(f"  Model:        {MODEL_PATH}")
    print(f"  Classes:      {len(class_names)} → {list(class_names.values())}")
    print(f"  Videos:       {VIDEO_DIR}  ({len(all_videos)} found)")
    if cli.num_shards > 1:
        print(f"  Shard:        {cli.shard+1}/{cli.num_shards}  "
              f"-> {len(video_files)} videos for this worker")
    print(f"  Output:       {OUTPUT_DIR}")
    print(f"  conf={CONF_THRESHOLD}  iou={IOU_THRESHOLD}  imgsz={IMG_SIZE}  tracker={TRACKER}")
    print(f"  Reprocess existing CSVs: {FORCE_REPROCESS}")
    print("=" * 62)

    # Metadata is kept for EVERY video, including ones whose tracking is
    # skipped or fails, because the risk features still need the frame size.
    video_meta = {}
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH) as f:
                video_meta = json.load(f)
        except Exception:
            video_meta = {}

    summary = []
    success_count = 0
    fail_count = 0

    # ─────────────────────────────────────────────────────────
    # MAIN LOOP — process each video
    # ─────────────────────────────────────────────────────────
    for video_idx, video_path in enumerate(video_files, 1):

        video_id = safe_video_id(video_path.stem)
        csv_path = os.path.join(OUTPUT_DIR, video_id + ".csv")

        print(f"\n[{video_idx}/{len(video_files)}] {video_path.name}")

        try:
            # ── Read metadata first, so it is recorded even if tracking is skipped ──
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print("  ❌ Cannot open video — skipping")
                fail_count += 1
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = cap.get(cv2.CAP_PROP_FPS) or pipeline_core.DEFAULT_FPS
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            video_meta[video_id] = {
                "fps": round(float(fps), 3),
                "width": width,
                "height": height,
                "frames": total_frames,
            }
            print(f"  Frames: {total_frames} | FPS: {fps:.1f} | Size: {width}x{height}")

            if os.path.exists(csv_path) and not FORCE_REPROCESS:
                print("  ⏭ Already processed — skipping (set FORCE_REPROCESS=True to redo)")
                success_count += 1
                continue

            if total_frames < 30:
                print(f"  ⚠ Video too short ({total_frames} frames) — skipping")
                fail_count += 1
                continue

            # ── Run detection + BoT-SORT tracking ──
            # stream=True processes frame by frame — memory efficient for long videos.
            #
            # persist=False is deliberate, and was a real bug when it was True.
            # persist keeps the SAME tracker instance alive across .track() calls.
            # That is right for feeding frames of one live stream, but here each
            # call is a different video, and BoT-SORT's global motion compensation
            # holds a previous-frame buffer. When the next video has a different
            # resolution the optical-flow pyramids mismatch and GMC raises BEFORE
            # refreshing that buffer, so the stale frame is never replaced and
            # every remaining frame of that video fails too:
            #
            #   persist=True   1080p video -> 0 failures, then 4K video -> 308/308
            #   persist=False  1080p video -> 0 failures, then 4K video ->   0/308
            #
            # In the previous corpus run the buffer stuck at 1920x1080, so every
            # 1080p clip worked and all 4K and 720p clips silently lost camera
            # motion compensation - including the moving-camera vlog footage that
            # contributed the most High-Risk windows. Uncompensated camera motion
            # reads as vehicle motion and inflates speed variance, which is the
            # feature that defines the High Risk cluster.
            # Track IDs are per-video here anyway, since each video gets its own CSV.
            results = model.track(
                source  = str(video_path),
                tracker = TRACKER,
                conf    = CONF_THRESHOLD,
                iou     = IOU_THRESHOLD,
                imgsz   = IMG_SIZE,
                persist = False,
                stream  = True,
                verbose = False,
                device  = 0,
            )

            all_rows = []
            frame_idx = 0
            total_dets = 0

            for result in results:

                if result.boxes is not None and result.boxes.id is not None:
                    boxes   = result.boxes.xywh.cpu().numpy()     # cx, cy, w, h
                    ids     = result.boxes.id.cpu().numpy().astype(int)
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    confs   = result.boxes.conf.cpu().numpy()

                    for box, track_id, cls_id, conf in zip(boxes, ids, classes, confs):
                        cx, cy, w, h = box

                        if w <= 0 or h <= 0:
                            continue

                        all_rows.append({
                            "frame":      frame_idx,
                            "track_id":   int(track_id),
                            "class_id":   int(cls_id),
                            "class_name": class_names.get(int(cls_id), str(cls_id)),
                            "cx":         round(float(cx), 2),
                            "cy":         round(float(cy), 2),
                            "width":      round(float(w), 2),
                            "height":     round(float(h), 2),
                            "confidence": round(float(conf), 4),
                        })
                        total_dets += 1

                frame_idx += 1

                if frame_idx % 300 == 0:
                    print(f"  ... frame {frame_idx}/{total_frames}")

            # ── Filter out very short tracks (likely false detections) ──
            track_frame_counts = {}
            for row in all_rows:
                tid = row["track_id"]
                track_frame_counts[tid] = track_frame_counts.get(tid, 0) + 1

            valid_tracks = {
                tid for tid, count in track_frame_counts.items()
                if count >= MIN_TRACK_LEN
            }
            filtered_rows = [r for r in all_rows if r["track_id"] in valid_tracks]

            removed = len(all_rows) - len(filtered_rows)
            print(f"  ✅ {frame_idx} frames | {total_dets} detections | "
                  f"{len(valid_tracks)} tracks kept, {removed} short-track rows dropped")

            # Per-class breakdown makes a taxonomy problem visible immediately,
            # rather than only showing up as odd numbers much later on.
            if filtered_rows:
                per_class = {}
                dominant = {}
                for r in filtered_rows:
                    dominant.setdefault(r["track_id"], r["class_name"])
                for cname in dominant.values():
                    per_class[cname] = per_class.get(cname, 0) + 1
                print(f"     vehicles by class: {per_class}")
            else:
                print("  ⚠ No valid tracks found — CSV will be empty but saved")

            fieldnames = ["frame", "track_id", "class_id", "class_name",
                          "cx", "cy", "width", "height", "confidence"]

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(filtered_rows)

            summary.append({
                "video": video_id,
                "frames": frame_idx,
                "detections": total_dets,
                "valid_tracks": len(valid_tracks),
                "status": "OK",
            })
            success_count += 1

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            print(f"  {traceback.format_exc()}")
            fail_count += 1
            summary.append({"video": video_id, "status": f"FAILED: {str(e)}"})
            continue

        finally:
            # Written every iteration so an interrupted run still leaves usable
            # metadata behind rather than losing the whole table.
            with open(META_PATH, "w") as f:
                json.dump(video_meta, f, indent=2)

    # ─────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  STEP 3 COMPLETE")
    print("=" * 62)
    print(f"  ✅ Successful: {success_count}/{len(video_files)}")
    print(f"  ❌ Failed:     {fail_count}/{len(video_files)}")
    print(f"\n  Tracking CSVs: {OUTPUT_DIR}")
    print(f"  Video metadata: {META_PATH}  ({len(video_meta)} videos)")

    failures = [s for s in summary if s["status"] != "OK"]
    if failures:
        print("\n  Videos that failed:")
        for s in failures:
            print(f"  ❌ {s['video'][:44]:44s} | {s['status']}")

    print("\n  Next step: run features_risk.py to extract the risk features")
    print("=" * 62)

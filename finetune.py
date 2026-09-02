"""
STEP 2 — Fine-tune YOLOv8n on FGVD (Indian Vehicle Detection)

Trains on the 5-class taxonomy produced by convert_fgvd.py:
    0 auto_rickshaw   1 motorcycle   2 car   3 bus   4 truck

WHAT CHANGED AND WHY
--------------------
1. This file previously did not parse at all. Line 24 was a bare row of box-
   drawing characters left outside any comment, so `python src/finetune.py`
   died with `SyntaxError: invalid character '─' (U+2500)` before running
   a single line. It is now a proper comment banner.

2. stdout is forced to UTF-8. The status lines below use emoji, and on Windows
   a redirected stdout defaults to cp1252 and crashes on the first print.

3. The old version trained a 6-class head on a dataset where two of those
   classes (auto_rickshaw, bicycle) had zero annotations, because the label
   converter silently dropped them. Retraining is only meaningful now that
   convert_fgvd.py maps every FGVD prefix — see its docstring.

The previous run's weights and curves are preserved under
    baseline_backup/models/fgvd_finetuned/
so this run can overwrite models/fgvd_finetuned/ without losing the baseline,
and every downstream path (detect_and_track.py, dashboard.py) keeps working
unchanged.

Run from the project root with the venv active:
    python src/finetune.py
"""

import os
import sys
import argparse

from ultralytics import YOLO

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
YAML_PATH   = r"E:\Traffic_Intelligence_Project\datasets\IDD_FGVD\dataset.yaml"
OUTPUT_DIR  = r"E:\Traffic_Intelligence_Project\models"
# Default matches the model the pipeline actually uses (see pipeline_core.PATHS),
# so re-running this script reproduces the promoted detector rather than a
# different one. Pass --name/--freeze to train a variant elsewhere.
RUN_NAME    = "fgvd_frozen"

EPOCHS      = 50
IMG_SIZE    = 640
BATCH_SIZE  = 16
PATIENCE    = 15
SEED        = 42

# FREEZE THE BACKBONE — the fix for catastrophic forgetting.
#
# Fine-tuning the whole network on FGVD's 3,535 street-level images overwrote
# the COCO backbone's general vehicle features. The result scored mAP50 0.886 on
# FGVD's own test split while collapsing on anything outside that narrow domain:
# on real elevated-view phone footage it found 6-8 vehicles per frame where
# stock yolov8n found 30-34, and it under-detected even on this project's own
# corpus clips. A benchmark measured on the same distribution it overfit to
# cannot reveal that.
#
# YOLOv8's first 10 modules are the backbone; 10+ is the detection head.
# freeze=10 keeps the pretrained backbone intact and trains only the head, so
# the model learns the 5 Indian vehicle classes (auto_rickshaw included) without
# discarding what COCO taught it about what a vehicle looks like.
#
# Set to 0 for a full fine-tune.
FREEZE      = 10
LR0         = 0.001
BASE_MODEL  = "yolo11m.pt"

# GEOMETRIC AUGMENTATION — the targeted fix for viewpoint generalisation.
#
# Ultralytics defaults these to zero: degrees=0.0, perspective=0.0, shear=0.0.
# Every FGVD image is shot from street level, so with rotation and perspective
# disabled the detector only ever sees vehicles from one viewing angle. Feed it
# an elevated or overhead clip - exactly what a phone recording from a building
# looks like - and cars, autos and scooters no longer resemble anything in
# training. Measured on real footage: stock COCO yolov8n found 34 vehicles in a
# frame where our fine-tune found 14, and the fine-tune is the model that was
# specifically trained on Indian traffic.
#
# Rotating and perspective-warping the training images synthesises the viewing
# angles FGVD lacks. Values are deliberately moderate - too much perspective
# warps boxes into nonsense and hurts the in-domain score for no gain.
AUG = {
    "degrees":     10.0,    # rotation, simulates camera tilt
    "perspective": 0.0005,  # perspective warp, simulates elevated viewpoints
    "shear":       2.0,
    # SCALE IS THE CRITICAL ONE — this is what teaches small distant vehicles.
    #
    # Measured on real elevated footage, the previous model (scale=0.6) could
    # not detect ANY vehicle under ~73 px tall, while stock COCO detected down
    # to 36 px. Every vehicle beyond mid-frame fell below that floor and was
    # missed entirely, which is why auto-rickshaws up the street were never
    # counted. Raising the inference size did not help and actually made it
    # worse — the objects moved further from the scale distribution the head
    # had learned, not closer to it.
    #
    # FGVD is entirely street-level close-ups: vehicles fill a large part of the
    # frame and the head never sees a small one. scale=0.9 shrinks training
    # images to as little as 0.1x, synthesising the distant vehicles the dataset
    # lacks. This is a training-data problem and only training can fix it.
    "scale":       0.9,
    "fliplr":      0.5,
    "mosaic":      1.0,
    "close_mosaic": 10,     # disable mosaic for the last N epochs to settle
}

# Varies the input size by +/-50% each batch. In principle it complements
# scale=0.9; in practice it is off because it does not fit on 8 GB.
#
# With imgsz=960 it pushes batches up to ~1440 px. At batch=8 that filled 7,866
# of 8,188 MiB and iteration time degraded from 1.0 to 2.5 s/it within a single
# epoch as memory pressure built — roughly 18 min/epoch, or 12 hours for the
# run, versus ~3 min/epoch without it.
#
# scale=0.9 already shrinks training objects to as little as 0.1x, which is the
# lever that actually teaches small-object detection; multi_scale only varies
# the input resolution around it. Not worth 4x the training time here. Enable it
# only with batch<=4, or on a GPU with more memory.
MULTI_SCALE = False


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # CLI overrides let a variant be trained into its own directory, so an
    # experiment never overwrites the model currently in use before it has been
    # shown to be better.
    ap = argparse.ArgumentParser(description="Fine-tune YOLOv8n on FGVD")
    ap.add_argument("--name", default=RUN_NAME, help="output run name")
    ap.add_argument("--freeze", type=int, default=FREEZE,
                    help="freeze the first N modules (10 = backbone, 0 = full fine-tune)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr0", type=float, default=LR0)
    ap.add_argument("--model", default=BASE_MODEL, help="base checkpoint")
    ap.add_argument("--imgsz", type=int, default=IMG_SIZE)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--no-aug", action="store_true",
                    help="disable the geometric augmentation described above")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from its last.pt checkpoint")
    args = ap.parse_args()

    # Resume path: Ultralytics reloads every hyperparameter from the run's saved
    # args.yaml, so the settings below must NOT be passed again - doing so would
    # silently start a different run from the same weights.
    if args.resume:
        ckpt = os.path.join(OUTPUT_DIR, args.name, "weights", "last.pt")
        if not os.path.exists(ckpt):
            print(f"❌ No checkpoint to resume from: {ckpt}")
            sys.exit(1)
        print(f"Resuming from {ckpt}")
        YOLO(ckpt).train(resume=True)
        print("✅ Resumed run complete")
        sys.exit(0)

    RUN_NAME, FREEZE, EPOCHS, LR0 = args.name, args.freeze, args.epochs, args.lr0
    BASE_MODEL, IMG_SIZE, BATCH_SIZE = args.model, args.imgsz, args.batch
    if args.no_aug:
        AUG = {k: (0.0 if k in ("degrees", "perspective", "shear") else v)
               for k, v in AUG.items()}

    if not os.path.exists(YAML_PATH):
        print(f"❌ dataset.yaml not found at: {YAML_PATH}")
        print("   Please run convert_fgvd.py first!")
        sys.exit(1)

    # Surface the class list being trained, so a mismatch between the yaml and
    # the converter is obvious here rather than 40 minutes later.
    with open(YAML_PATH) as f:
        yaml_text = f.read()

    print("=" * 60)
    print("  YOLOv8n Fine-tuning on FGVD Dataset")
    print("=" * 60)
    print(f"  Dataset:    {YAML_PATH}")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMG_SIZE}")
    print(f"  Output:     {OUTPUT_DIR}/{RUN_NAME}")
    print("-" * 60)
    for line in yaml_text.splitlines():
        if line.strip() and not line.strip().startswith("#"):
            print(f"  {line}")
    print("=" * 60)
    print()

    print(f"Loading base model: {BASE_MODEL} ...")
    model = YOLO(BASE_MODEL)
    print("✅ Base model loaded")
    print(f"   freeze={FREEZE}  imgsz={IMG_SIZE}  batch={BATCH_SIZE}")
    print(f"   augmentation: {AUG}\n")

    print("Starting fine-tuning... (this will take 30-60 minutes)")
    print("You will see progress printed for each epoch below.\n")

    results = model.train(
        data         = YAML_PATH,
        epochs       = EPOCHS,
        imgsz        = IMG_SIZE,
        batch        = BATCH_SIZE,
        patience     = PATIENCE,
        device       = 0,
        project      = OUTPUT_DIR,
        name         = RUN_NAME,
        exist_ok     = True,
        pretrained   = True,
        freeze       = FREEZE if FREEZE > 0 else None,
        optimizer    = "AdamW",
        lr0          = LR0,
        lrf          = 0.01,
        weight_decay = 0.0005,
        seed         = SEED,
        val          = True,
        save         = True,
        plots        = True,
        verbose      = True,
        # cache="ram" looks like free speed but is a trap on Windows. Workers
        # are SPAWNED, not forked, so every worker pickles the entire cached
        # dataset from the parent. With 3,535 images at imgsz=960 and 8 workers
        # that is tens of GB, and it died with MemoryError the moment
        # close_mosaic rebuilt the dataloader mid-run (epoch 30 of 40) — after
        # an hour of training. Caching to disk gets most of the benefit with a
        # bounded memory cost.
        workers      = 4,
        cache        = False,
        multi_scale  = MULTI_SCALE,
        **AUG
    )

    print("\n" + "=" * 60)
    print("  FINE-TUNING COMPLETE")
    print("=" * 60)

    best_model_path = os.path.join(OUTPUT_DIR, RUN_NAME, "weights", "best.pt")

    if os.path.exists(best_model_path):
        print(f"✅ Best model saved at:\n   {best_model_path}\n")
        print("Key metrics:")
        # .get() can return a string sentinel, which would blow up a float
        # format — so format defensively rather than assuming a number.
        for label, key in [("mAP50   ", "metrics/mAP50(B)"),
                           ("mAP50-95", "metrics/mAP50-95(B)"),
                           ("precision", "metrics/precision(B)"),
                           ("recall  ", "metrics/recall(B)")]:
            v = results.results_dict.get(key)
            print(f"  {label}: {v:.3f}" if isinstance(v, (int, float)) else f"  {label}: n/a")
        print()
        print("Next step: run detect_and_track.py to reprocess your traffic videos")
    else:
        print("⚠ Could not find best.pt — check the output folder manually")

    print("=" * 60)

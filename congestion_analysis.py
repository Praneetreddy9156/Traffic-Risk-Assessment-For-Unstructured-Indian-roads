"""
congestion_analysis.py

Step 4 (congestion half). Risk features are handled in a separate script
once this stage is validated.

Pipeline: read the 124 tracking CSVs from Step 3 (YOLOv8n + BoT-SORT) ->
split into fixed-DURATION windows -> average vehicle count per window (the
raw congestion value) -> cluster those values with K-means -> derive fixed
Low/Medium/High-style thresholds from the cluster centroids.

Three methodology decisions worth being able to explain out loud:
  1. Windows are 3 SECONDS of real time, not a fixed 90 frames. The videos
     in this dataset are a mix of 25/30/60fps (encoded in the filenames),
     so a fixed 90-frame window would represent 3 seconds for a 30fps clip
     but only 1.5 seconds for a 60fps clip - not a fair comparison across
     videos. Frame count per window is now computed per video from its own
     fps, so every window represents the same amount of real time.
  2. k is not assumed to be 3. It's chosen by comparing silhouette scores
     across k=2..5 on the training data, so there's a real number backing
     the cluster count rather than "3 is the usual convention."
  3. The train/val/test split happens at the VIDEO level, not the window
     level, to avoid leaking information between windows from the same
     clip (see split_by_video for the reasoning).

Outputs:
  - congestion_windows_labeled.csv   one row per window, includes the fps
                                      used and whether it was parsed from
                                      the filename or assumed
  - congestion_video_summary.csv     one row per video (video-level label)
  - congestion_thresholds.json       k, centroids, thresholds, silhouette
                                      scores for every k tested
"""

import os
import re
import sys
import glob
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pipeline_core
from pipeline_core import PATHS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Tracking CSVs now live in their own subdirectory. They used to sit in
# outputs/csv/ next to the derived analysis files, so this glob also picked up
# risk_predictions.csv and friends and relied on a column check to reject them.
CSV_DIR = PATHS["tracking_dir"]
OUT_DIR = PATHS["congestion_dir"]

# Written by detect_and_track.py, which records each video's real fps and frame
# size at tracking time. Previously fps had to be recovered afterwards by
# re-opening every source video, because the tracker read it and threw it away.
# Missing file isn't fatal - resolve_fps falls back to the filename, then a
# default, and reports which was used.
VIDEO_META_PATH = PATHS["video_meta"]

WINDOW_SECONDS = pipeline_core.DEFAULT_WINDOW_SECONDS   # shared with the risk features
DEFAULT_FPS = pipeline_core.DEFAULT_FPS                 # last-resort fallback only
K_RANGE = (2, 3, 4, 5)
RANDOM_SEED = 42
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20                # remaining ~0.20 goes to test

# set to an integer to skip silhouette-based selection and force that k
# instead (e.g. keeping Low/Medium/High for interpretability even if k=2
# scores higher). None lets the silhouette score decide.
FORCE_K = None

LABEL_SCHEMES = {
    2: ["Low", "High"],
    3: ["Low", "Medium", "High"],
    4: ["Low", "Medium", "High", "Very High"],
    5: ["Very Low", "Low", "Medium", "High", "Very High"],
}

os.makedirs(OUT_DIR, exist_ok=True)


def load_fps_lookup(path=None):
    """
    Loads the video_id -> metadata table written by detect_and_track.py.

    Falls back to the older flat {video_id: fps} lookup so an existing checkout
    keeps working before detect_and_track.py has been re-run. Missing entirely
    isn't fatal either - every video then falls back to filename-parsing / the
    default, exactly as before.
    """
    if path is None:
        path = VIDEO_META_PATH

    meta = pipeline_core.load_video_meta(path)
    if meta:
        print(f"loaded metadata for {len(meta)} videos from {path}\n")
        return meta

    meta = pipeline_core.load_video_meta(PATHS["legacy_fps_lookup"])
    if meta:
        print(f"no {path} - using legacy fps lookup ({len(meta)} videos)")
        print("run detect_and_track.py to record real fps and frame sizes\n")
        return meta

    print(f"no video metadata found at {path}")
    print("continuing with filename-parsing / default fallback only\n")
    return {}


def resolve_fps(video_id, fps_lookup, default_fps=DEFAULT_FPS):
    """
    Three tiers, in order of trust: the measured value recorded at tracking
    time, then an '_NNfps' pattern in the filename, then the default. Returns
    (fps, source) where source is "measured", "filename" or "assumed" - kept in
    the output so an assumption stays visible per video.

    Delegates to pipeline_core so this file and features_risk.py cannot end up
    resolving the same video to different frame rates.
    """
    return pipeline_core.resolve_fps(video_id, fps_lookup, default_fps)


def windows_for_one_video(csv_path, fps_lookup, window_seconds=WINDOW_SECONDS):
    """
    Converts one video's frame-by-frame tracking CSV into window-level
    rows, where each window covers window_seconds of real time - the
    frame count per window is derived from this video's own fps, so a
    'window' means the same duration everywhere in the dataset.

    Each window's congestion value is the average vehicle count across
    all frames in that window, not a single frame's count - a single
    frame is noisy (a bus briefly blocking the camera shouldn't define
    the whole window), averaging smooths that out.

    start_frame/end_frame are kept per window so the risk script can slice
    the original tracking CSV down to exactly this window's rows. fps and
    fps_source are kept too since the risk features (speed, deceleration)
    will need real fps, and it's worth knowing how trustworthy that number
    is for each video.
    """
    video_id = os.path.splitext(os.path.basename(csv_path))[0]
    fps, fps_source = resolve_fps(video_id, fps_lookup)
    window_frames = max(1, round(window_seconds * fps))

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  skipping {video_id}: couldn't open it ({e})")
        return pd.DataFrame(), fps_source

    # the tracking script (detect_and_track.py) outputs this column as
    # 'frame', not 'frame_id' - normalising the name here
    if "frame" in df.columns and "frame_id" not in df.columns:
        df = df.rename(columns={"frame": "frame_id"})

    if not {"frame_id", "track_id"}.issubset(df.columns):
        print(f"  skipping {video_id}: missing frame_id/track_id")
        return pd.DataFrame(), fps_source

    # integer division buckets frames into windows of window_frames length -
    # same idea as before, just using a per-video frame count now instead
    # of one fixed number for every video
    df["window_id"] = df["frame_id"] // window_frames

    per_frame_counts = (
        df.groupby("frame_id")["track_id"].nunique()
        .rename("vehicle_count").reset_index()
    )
    per_frame_counts["window_id"] = per_frame_counts["frame_id"] // window_frames

    window_rows = (
        per_frame_counts.groupby("window_id")
        .agg(raw_congestion=("vehicle_count", "mean"),
             frames_in_window=("vehicle_count", "count"),
             start_frame=("frame_id", "min"),
             end_frame=("frame_id", "max"))
        .reset_index()
    )
    window_rows["video_id"] = video_id
    window_rows["fps"] = fps
    window_rows["fps_source"] = fps_source
    window_rows["window_frames_used"] = window_frames

    # a trailing window with fewer than half the expected frames is
    # usually just the leftover tail of a video that doesn't divide evenly
    # by window_frames - not comparable to a complete window, so dropped
    window_rows = window_rows[window_rows["frames_in_window"] >= window_frames / 2]

    cols = ["video_id", "window_id", "start_frame", "end_frame",
            "raw_congestion", "frames_in_window", "fps", "fps_source",
            "window_frames_used"]
    return window_rows[cols], fps_source


def build_all_windows(csv_dir, fps_lookup):
    all_csvs = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    print(f"found {len(all_csvs)} tracking CSVs in {csv_dir}")

    pieces = []
    fps_source_counts = {"measured": 0, "filename": 0, "assumed": 0}
    assumed_videos = []

    for path in all_csvs:
        rows, fps_source = windows_for_one_video(path, fps_lookup)
        pieces.append(rows)
        if not rows.empty:
            fps_source_counts[fps_source] += 1
            if fps_source == "assumed":
                assumed_videos.append(os.path.splitext(os.path.basename(path))[0])

    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.dropna(subset=["raw_congestion"])

    print(f"{len(combined)} congestion windows total, across {combined['video_id'].nunique()} videos")
    print(f"fps source breakdown: {fps_source_counts['measured']} measured, "
          f"{fps_source_counts['filename']} from filename, "
          f"{fps_source_counts['assumed']} assumed ({DEFAULT_FPS}fps default)")

    if assumed_videos:
        print(f"\nthese {len(assumed_videos)} videos had no measured or filename fps, "
              f"assumed {DEFAULT_FPS}fps:")
        for v in assumed_videos:
            print(f"  - {v}")
        print("run build_fps_lookup.py if these need a real fps value instead")

    return combined


def split_by_video(all_windows, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=RANDOM_SEED):
    """
    Splitting whole videos, not individual windows, into train/val/test.
    Windows from the same video sit close together in time and tend to
    look alike (same road, same traffic, same camera) - splitting
    window-by-window could put some windows from a video into training and
    others from that same video into test, which isn't a genuinely unseen
    test set. Keeping whole videos together avoids that.
    """
    video_ids = np.array(all_windows["video_id"].unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(video_ids)

    n = len(video_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_videos = set(video_ids[:n_train])
    val_videos = set(video_ids[n_train:n_train + n_val])
    test_videos = set(video_ids[n_train + n_val:])

    train_df = all_windows[all_windows["video_id"].isin(train_videos)].copy()
    val_df = all_windows[all_windows["video_id"].isin(val_videos)].copy()
    test_df = all_windows[all_windows["video_id"].isin(test_videos)].copy()

    print(f"video split -> train: {len(train_videos)}, val: {len(val_videos)}, test: {len(test_videos)}")
    return train_df, val_df, test_df


def pick_best_k(train_values, k_range=K_RANGE, seed=RANDOM_SEED):
    """
    Fits K-means for every candidate k on the TRAINING split only, scores
    each with the silhouette coefficient (-1 to +1, higher = better
    separated clusters), and returns the best-scoring k. Exists so the
    cluster count is a measured choice, not an assumption.
    """
    X = train_values.reshape(-1, 1)
    scores = {}
    for k in k_range:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
        labels = model.fit_predict(X)
        scores[k] = silhouette_score(X, labels)

    best_k = max(scores, key=scores.get)

    print("silhouette score by k:")
    for k, s in scores.items():
        tag = "  <- best score" if k == best_k else ""
        print(f"  k={k}: {s:.3f}{tag}")

    return best_k, scores


def fit_congestion_kmeans(train_values, k, seed=RANDOM_SEED):
    """
    Final K-means fit, training split only, so val/test stay genuinely
    held-out. init='k-means++' with n_init=10 tries 10 different starting
    configurations and keeps the tightest result, guarding against one bad
    random start producing a poor grouping.
    """
    X = train_values.reshape(-1, 1)
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
    model.fit(X)

    centroids = sorted(model.cluster_centers_.flatten().tolist())

    # boundary between adjacent labels = midpoint between their centroids -
    # a fixed number derived from the data, reused at inference time
    # without re-running K-means
    thresholds = [
        (centroids[i] + centroids[i + 1]) / 2
        for i in range(len(centroids) - 1)
    ]

    return model, centroids, thresholds


def get_labels_for_k(k):
    if k in LABEL_SCHEMES:
        return LABEL_SCHEMES[k]
    return [f"Level {i+1}" for i in range(k)]


def label_from_thresholds(value, thresholds, labels):
    for i, t in enumerate(thresholds):
        if value < t:
            return labels[i]
    return labels[-1]


def main():
    fps_lookup = load_fps_lookup()
    all_windows = build_all_windows(CSV_DIR, fps_lookup)

    if all_windows.empty:
        print("no windows produced - check CSV_DIR and the column names before anything else")
        return

    train_df, val_df, test_df = split_by_video(all_windows)

    print("\nselecting cluster count (k) via silhouette comparison, k=2..5...")
    best_k, scores = pick_best_k(train_df["raw_congestion"].to_numpy())

    if FORCE_K is not None:
        print(f"\nFORCE_K override active: using k={FORCE_K} instead of the silhouette winner (k={best_k})")
        k_used = FORCE_K
    else:
        k_used = best_k
        print(f"\nusing k={k_used} (silhouette winner)")

    labels = get_labels_for_k(k_used)

    print(f"fitting final model with k={k_used} on the training split...")
    model, centroids, thresholds = fit_congestion_kmeans(train_df["raw_congestion"].to_numpy(), k_used)

    print(f"centroids, low to high: {[round(c, 2) for c in centroids]}")
    print(f"thresholds derived from centroid midpoints: {[round(t, 2) for t in thresholds]}")
    print(f"labels in use: {labels}")

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df["congestion_label"] = split_df["raw_congestion"].apply(
            lambda v: label_from_thresholds(v, thresholds, labels)
        )

    full = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print("\nfinal label counts (per window):")
    print(full["congestion_label"].value_counts())

    class_share = full["congestion_label"].value_counts(normalize=True)
    thin = class_share[class_share < 0.05]
    if not thin.empty:
        print(f"\nlabels under 5% of all windows: {thin.to_dict()}")
        print("worth checking before building risk features on top of this")

    out_windows = os.path.join(OUT_DIR, "congestion_windows_labeled.csv")
    full.to_csv(out_windows, index=False)
    print(f"\nwritten: {out_windows}")

    # per-video summary - video-level label comes from the video's MEAN
    # congestion run through the same thresholds, not a majority vote of
    # its windows' labels, so it stays consistent with how window labels
    # were derived in the first place
    video_summary = (
        full.groupby("video_id")
        .agg(num_windows=("window_id", "count"),
             mean_congestion=("raw_congestion", "mean"),
             min_congestion=("raw_congestion", "min"),
             max_congestion=("raw_congestion", "max"),
             fps=("fps", "first"),
             fps_source=("fps_source", "first"))
        .reset_index()
    )
    video_summary["video_congestion_label"] = video_summary["mean_congestion"].apply(
        lambda v: label_from_thresholds(v, thresholds, labels)
    )
    out_summary = os.path.join(OUT_DIR, "congestion_video_summary.csv")
    video_summary.to_csv(out_summary, index=False)
    print(f"written: {out_summary}")

    print("\nvideo-level label counts:")
    print(video_summary["video_congestion_label"].value_counts())

    threshold_record = {
        "window_seconds": WINDOW_SECONDS,
        "k": k_used,
        "silhouette_scores_by_k": {str(k): round(s, 4) for k, s in scores.items()},
        "random_seed": RANDOM_SEED,
        "centroids": centroids,
        "thresholds": thresholds,
        "labels_low_to_high": labels,
    }
    out_json = os.path.join(OUT_DIR, "congestion_thresholds.json")
    with open(out_json, "w") as f:
        json.dump(threshold_record, f, indent=2)
    print(f"written: {out_json}")


if __name__ == "__main__":
    main()
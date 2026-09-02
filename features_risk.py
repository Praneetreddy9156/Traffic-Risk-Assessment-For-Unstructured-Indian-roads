"""
features_risk.py

Computes the 5 risk features per vehicle per 3-second window across every
tracking CSV, using each video's real fps AND real frame width so the numbers
mean the same thing for a 4K clip and a phone clip.

THE BUG THIS VERSION FIXES
--------------------------
Speeds and distances used to be measured in RAW PIXELS of the source video, and
the thresholds derived from them were frozen as pixel constants
(deceleration >= 313.7 px/s, proximity < 642.2 px). The corpus is mostly
1080p-4K, so those constants encode "large frame". Feed in a 478-px-wide phone
clip and every distance shrinks by the same factor - and speed VARIANCE by the
square of it - so nothing ever crosses the threshold and every window comes back
Safe or Moderate.

Measured on real corpus footage, downscaling coordinates alone (identical video,
identical driving) took a clip from 21 High-Risk windows at 1280x720 to 0 at
320x180. Risk was tracking resolution.

pipeline_core now divides every coordinate by the frame width first, so speed is
in frame-widths/second and the thresholds derived here are resolution-free.

The window is also a true 3 SECONDS now, derived from each video's own fps,
instead of a flat 90 frames - that flat count was 3 seconds at 30fps but only
1.5 seconds at 60fps, so congestion counts were not comparable across the corpus
(congestion_analysis.py already did this correctly; the two now agree).

Two-pass design, unchanged in spirit:
  Pass 1 - walk every video once and collect the RAW distributions needed to
           derive two data-driven thresholds:
             - nearest-neighbour distance -> K-means(k=2) + silhouette
               -> proximity_threshold
             - frame-to-frame speed drop  -> mean + 1.5*std
               -> deceleration_threshold
  Pass 2 - walk every video again and compute the 5 features per
           (video, window, vehicle) using those thresholds.

Output: outputs/csv/risk_features.csv          (one row per video-window-vehicle)
        outputs/risk_feature_thresholds.json   (derivation summary + units)
"""

import os
import sys
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pipeline_core
from pipeline_core import PATHS, DEFAULT_WINDOW_SECONDS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONFIG = {
    "tracking_dir": PATHS["tracking_dir"],
    "video_meta": PATHS["video_meta"],
    "output_csv": PATHS["risk_features"],
    "output_thresholds": PATHS["risk_thresholds"],
    "window_seconds": DEFAULT_WINDOW_SECONDS,
    # Silhouette on every nearest-neighbour sample is O(n^2) in memory; the
    # threshold is a 1-D split, so a large random subsample scores it just as
    # well without trying to allocate an 80k x 80k matrix.
    "silhouette_sample": 5000,
    "random_state": 42,
}


def load_meta():
    meta = pipeline_core.load_video_meta(CONFIG["video_meta"])
    if meta:
        print(f"loaded metadata for {len(meta)} videos from {CONFIG['video_meta']}")
        return meta

    # Fall back to the older flat fps lookup so this still runs on an existing
    # checkout that has not re-run detect_and_track.py yet. Frame width then
    # falls back to the filename or the observed box extent (see pipeline_core).
    legacy = pipeline_core.load_video_meta(PATHS["legacy_fps_lookup"])
    if legacy:
        print(f"WARNING: {CONFIG['video_meta']} not found - falling back to "
              f"{PATHS['legacy_fps_lookup']} ({len(legacy)} videos, fps only).")
        print("         Re-run detect_and_track.py to record real frame sizes.")
        return legacy

    print("WARNING: no video metadata found - fps and frame width will be inferred "
          "from filenames and observed box extents.")
    return {}


def main():
    meta = load_meta()

    csv_files = sorted(glob.glob(os.path.join(CONFIG["tracking_dir"], "*.csv")))
    print(f"found {len(csv_files)} tracking CSVs in {CONFIG['tracking_dir']}")
    if not csv_files:
        print("nothing to do - run detect_and_track.py first")
        return

    # ------------------------------------------------------------------
    # PASS 1 - collect raw distributions across the WHOLE dataset
    # ------------------------------------------------------------------
    all_nn_distances = []
    all_decel_magnitudes = []
    video_cache = {}     # video_id -> (df, fps, frame_width), reused in pass 2
    width_sources = {}

    for csv_path in csv_files:
        video_id = Path(csv_path).stem
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            fps, fps_src = pipeline_core.resolve_fps(video_id, meta)
            frame_width, w_src = pipeline_core.resolve_frame_width(video_id, meta, df)
            width_sources[w_src] = width_sources.get(w_src, 0) + 1

            video_cache[video_id] = (df, fps, frame_width)

            norm = pipeline_core.normalize_positions(df, frame_width)
            norm = pipeline_core.assign_windows(norm, fps, CONFIG["window_seconds"])

            # Whole-video passes over numpy arrays rather than a pandas groupby
            # per frame and per (window, track). Profiling showed that grouping
            # overhead, not the arithmetic, dominated this stage.
            nn = pipeline_core.all_nearest_neighbor_distances(norm)
            if len(nn):
                all_nn_distances.append(nn)

            frames_arr = norm["frame"].to_numpy()
            xs_arr = norm["nx"].to_numpy()
            ys_arr = norm["ny"].to_numpy()

            pairs = np.column_stack((norm["window_id"].to_numpy(),
                                     norm["track_id"].to_numpy()))
            _, pair_ids = np.unique(pairs, axis=0, return_inverse=True)
            _, groups = pipeline_core.group_indices(pair_ids)

            for idx in groups:
                sel = idx[np.argsort(frames_arr[idx], kind="stable")]
                positions = np.column_stack((frames_arr[sel], xs_arr[sel], ys_arr[sel]))
                speeds = pipeline_core.compute_speeds(positions, fps)
                if len(speeds) >= 2:
                    drops = speeds[:-1] - speeds[1:]
                    all_decel_magnitudes.append(drops[drops > 0])

        except Exception as e:
            print(f"  [pass 1] skipping {video_id}: {e}")
            continue

    # Collected per video as arrays, joined once - cheaper than growing a
    # Python list element by element across the whole corpus.
    nn_all = np.concatenate(all_nn_distances) if all_nn_distances else np.array([])
    decel_all = np.concatenate(all_decel_magnitudes) if all_decel_magnitudes else np.array([])

    print(f"\npass 1 done - {len(nn_all)} nearest-neighbour samples, "
          f"{len(decel_all)} deceleration samples")
    print(f"frame-width sources: {width_sources}")

    if not len(nn_all) or not len(decel_all):
        print("not enough data to derive thresholds - aborting")
        return

    # ------------------------------------------------------------------
    # Derive proximity_threshold via K-means(k=2) + silhouette
    # ------------------------------------------------------------------
    nn_array = nn_all[np.isfinite(nn_all)].reshape(-1, 1)
    km = KMeans(n_clusters=2, random_state=CONFIG["random_state"], n_init=10).fit(nn_array)

    sample = min(CONFIG["silhouette_sample"], len(nn_array))
    sil = silhouette_score(nn_array, km.labels_, sample_size=sample,
                           random_state=CONFIG["random_state"])

    centroids = sorted(km.cluster_centers_.flatten())
    proximity_threshold = float(np.mean(centroids))
    print(f"\nproximity_threshold = {proximity_threshold:.4f} frame-widths "
          f"(centroids {centroids[0]:.4f}, {centroids[1]:.4f}; silhouette {sil:.3f})")

    # ------------------------------------------------------------------
    # Derive deceleration_threshold via mean + 1.5*std
    # ------------------------------------------------------------------
    decel_array = decel_all
    decel_threshold = float(decel_array.mean() + 1.5 * decel_array.std())
    print(f"deceleration_threshold = {decel_threshold:.4f} frame-widths/sec drop "
          f"(mean {decel_array.mean():.4f}, std {decel_array.std():.4f})")

    os.makedirs(os.path.dirname(CONFIG["output_thresholds"]), exist_ok=True)
    with open(CONFIG["output_thresholds"], "w") as f:
        json.dump({
            # "units" is recorded explicitly because these numbers used to be
            # pixels; anything reading this file must not mix the two.
            "units": "frame_widths",
            "window_seconds": CONFIG["window_seconds"],
            "proximity_threshold": proximity_threshold,
            "proximity_centroids": centroids,
            "proximity_silhouette": float(sil),
            "deceleration_threshold": decel_threshold,
            "deceleration_mean": float(decel_array.mean()),
            "deceleration_std": float(decel_array.std()),
            "max_plausible_speed": pipeline_core.MAX_PLAUSIBLE_SPEED,
            "n_nn_samples": int(len(nn_array)),
            "n_decel_samples": int(len(decel_array)),
        }, f, indent=2)

    # ------------------------------------------------------------------
    # PASS 2 - compute the 5 features per video/window/vehicle
    # ------------------------------------------------------------------
    frames = []
    for video_id, (df, fps, frame_width) in video_cache.items():
        try:
            feat = pipeline_core.compute_features(
                df, fps, frame_width,
                proximity_threshold, decel_threshold,
                window_seconds=CONFIG["window_seconds"],
            )
            if feat.empty:
                continue
            feat.insert(0, "video_id", video_id)
            frames.append(feat)
        except Exception as e:
            print(f"  [pass 2] skipping {video_id}: {e}")
            continue

    out_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    os.makedirs(os.path.dirname(CONFIG["output_csv"]), exist_ok=True)
    out_df.to_csv(CONFIG["output_csv"], index=False)
    print(f"\nwrote {len(out_df)} vehicle-window rows to {CONFIG['output_csv']}")

    if not out_df.empty:
        print("\nfeature summary:")
        print(out_df[pipeline_core.FEATURES].describe().round(4))


if __name__ == "__main__":
    main()

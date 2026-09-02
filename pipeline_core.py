"""
pipeline_core.py

Single source of truth for the geometry and feature math used by the whole
project. features_risk.py, dashboard.py and generate_charts.py all import from
here instead of keeping their own copies.

WHY THIS MODULE EXISTS
----------------------
Two problems made it necessary.

1. THE SAME MATH WAS COPY-PASTED INTO THREE FILES.
   compute_speeds / compute_displacement_ratio / nearest_neighbor_distances
   existed independently in features_risk.py, dashboard.py and
   generate_charts.py. Three copies of a formula drift, and when they drift the
   dashboard silently stops agreeing with the model that was trained offline.

2. THE FEATURES WERE MEASURED IN RAW PIXELS, SO THEY DEPENDED ON RESOLUTION.
   Speed was pixels/second of the source video. The training corpus is mostly
   1080p-4K, so the thresholds derived from it (deceleration >= 313.7 px/s)
   were calibrated for large frames. Feed in a 478-px-wide phone clip and every
   motion measurement shrinks by the same factor - and speed VARIANCE shrinks by
   the square of it - so the clip can never reach the threshold.

   Measured on real corpus footage, downscaling the coordinates alone (same
   video, same driving, same model) moved the result from 21 High-Risk windows
   at 1280x720 to 0 at 320x180. "High Risk" was effectively measuring video
   resolution.

   The fix is to express every distance as a fraction of the FRAME WIDTH before
   any speed or proximity calculation. Both x and y are divided by the width
   (not by their own axis) so the aspect ratio is preserved and geometry is not
   distorted. Speed then has units of "frame-widths per second", which means the
   same thing for a phone clip and a 4K clip.

UNITS AFTER THIS MODULE
-----------------------
    position                normalised, 1.0 == one frame width
    speed                   frame-widths / second
    speed_variance          (frame-widths / second)^2
    proximity threshold     frame-widths
    deceleration threshold  frame-widths / second
    displacement_ratio      already dimensionless, unchanged
    congestion_score        distinct vehicles in a fixed-DURATION window
"""

import os
import re
import json

import numpy as np
import pandas as pd

# Windows are a fixed amount of REAL TIME, not a fixed frame count. The corpus
# mixes 25/30/60fps, so a flat 90-frame window would be 3 seconds of a 30fps
# clip but only 1.5 seconds of a 60fps one - congestion counts and speed
# statistics from those two are not comparable.
DEFAULT_WINDOW_SECONDS = 3.0
DEFAULT_FPS = 30.0

# A tracker ID-switch teleports a box across the frame in one step, producing a
# speed no real vehicle can reach. Left unfiltered these dominate the variance:
# the corpus contained a speed_variance of 11,498,238 px^2/s^2, which is jitter,
# not driving. 3.0 frame-widths/sec means crossing the entire frame three times
# in a second - comfortably above any genuine vehicle, so anything faster is
# discarded as a tracking artefact rather than fed into the features.
MAX_PLAUSIBLE_SPEED = 3.0

FEATURES = [
    "displacement_ratio",
    "speed_variance",
    "proximity_interactions",
    "deceleration_events",
    "congestion_score",
]

# ----------------------------------------------------------------------
# Canonical paths
# ----------------------------------------------------------------------
# Every script used to hard-code the same absolute paths as string literals,
# so moving anything meant editing five files and hoping none were missed.
#
# Per-video tracking CSVs now live in their own subdirectory. They previously
# sat in outputs/csv/ alongside the derived files (risk_features.csv,
# risk_predictions.csv, dashboard_uploads.csv), and every consumer globbed
# "*.csv" and then tried to filter the derived ones back out by name - a list
# that was already incomplete and silently swallowed the mistakes in a
# try/except. Separate directories make that class of bug impossible.
PROJECT_ROOT = r"E:\Traffic_Intelligence_Project"

PATHS = {
    "tracking_dir":       os.path.join(PROJECT_ROOT, "outputs", "csv", "tracking"),
    "uploads_dir":        os.path.join(PROJECT_ROOT, "outputs", "uploads"),
    "video_meta":         os.path.join(PROJECT_ROOT, "outputs", "video_meta.json"),
    "legacy_fps_lookup":  os.path.join(PROJECT_ROOT, "outputs", "video_fps_lookup.json"),
    "risk_features":      os.path.join(PROJECT_ROOT, "outputs", "csv", "risk_features.csv"),
    "risk_predictions":   os.path.join(PROJECT_ROOT, "outputs", "csv", "risk_predictions.csv"),
    "uploads_store":      os.path.join(PROJECT_ROOT, "outputs", "csv", "dashboard_uploads.csv"),
    "risk_thresholds":    os.path.join(PROJECT_ROOT, "outputs", "risk_feature_thresholds.json"),
    "risk_metrics":       os.path.join(PROJECT_ROOT, "outputs", "risk_model_metrics.json"),
    "congestion_dir":     os.path.join(PROJECT_ROOT, "outputs", "congestion"),
    "charts_dir":         os.path.join(PROJECT_ROOT, "outputs", "charts"),
    "risk_models_dir":    os.path.join(PROJECT_ROOT, "models", "risk_model"),
    # YOLO11m, frozen backbone, viewpoint augmentation
    # (degrees=10, perspective=0.0005, shear=2, scale=0.6).
    #
    # models/fgvd_smallobj/ (same but scale=0.9) was tried and REVERTED. On one
    # test clip it looked better - more auto_rickshaws, far fewer phantom buses,
    # 107 vehicles vs 91 - so it was promoted. In use across multiple different
    # videos it inflated counts badly, which a single-video comparison could not
    # reveal. The aggressive scale=0.9 augmentation trains on objects shrunk to
    # 0.1x, and the resulting sensitivity to tiny features appears to fire on
    # background texture in footage unlike the one clip it was validated on.
    #
    # The lesson is the same one that hid the original 34% annotation loss: a
    # detector change must be judged on a RANGE of videos, not one. One clip
    # said promote, four sampled frames said reject, and real use across many
    # videos said reject. Trust the broadest evidence.
    #
    # Kept for comparison: models/fgvd_smallobj/ (scale=0.9, reverted),
    # models/fgvd_frozen/ (YOLOv8n frozen), models/fgvd_finetuned/ (full
    # fine-tune), baseline_backup/models/fgvd_finetuned/ (original 6-class).
    "yolo_weights":       os.path.join(PROJECT_ROOT, "models", "fgvd_v11m", "weights", "best.pt"),
    "video_dir":          r"E:\TrafficVideos",
}

# ----------------------------------------------------------------------
# Detection / tracking settings
# ----------------------------------------------------------------------
# One definition, used by BOTH detect_and_track.py (the offline corpus) and the
# dashboard's live upload path. They previously disagreed - the corpus ran at
# conf=0.35 / iou=0.45 / imgsz=640 while the dashboard ran conf=0.25 with
# library defaults for the rest - so an uploaded clip was never measured the
# same way as the data the model was trained on.
DETECTION = {
    # Lowered from 0.35. At 0.35 the detector missed small and distant vehicles
    # in dense traffic, so counts came in well under a manual count. The current
    # model is confident enough that 0.25 adds real vehicles rather than noise.
    "conf": 0.25,
    # Raised from 0.45. In jammed traffic vehicles genuinely overlap, and an
    # aggressive NMS threshold suppressed the ones behind.
    "iou": 0.6,
    # Matches the resolution the detector was TRAINED at (960). Inferring at the
    # training resolution is the principled default - the anchors and receptive
    # fields are tuned for it - and it is ~1.8x faster than 1280. The dashboard
    # exposes this per-upload, so low-resolution phone footage with very small
    # vehicles can still be pushed to 1280 where that genuinely helps.
    "imgsz": 960,
    # Project-local BoT-SORT config, not the library default. The default
    # retires a lost track after 30 frames, so any vehicle briefly occluded in
    # dense traffic returned as a NEW id and got counted twice. See
    # src/botsort_traffic.yaml for the measurements behind track_buffer=150.
    "tracker": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "botsort_traffic.yaml"),
    # 2 is the MATHEMATICAL minimum, not a quality filter. It was 5, which
    # silently deleted vehicles a person watching the video would still count.
    #
    # It cannot go below 2: speed, displacement ratio and deceleration are all
    # computed from differences between consecutive positions, so a track seen
    # in a single frame has no motion to measure at all - it would enter the
    # feature table as zeros and be indistinguishable from a stationary vehicle.
    # Single-frame detections are still REPORTED in the detection breakdown, so
    # nothing is hidden; they are only excluded from the motion features that
    # are undefined for them. Adjustable per-upload in the dashboard.
    "min_track_len": 2,
}


def safe_video_id(video_name):
    """
    Filename -> video_id, the sanitisation used for tracking CSV names.

    Shared so the CSV names, the video_meta.json keys and the upload store all
    agree; when this logic was duplicated the lookup keys silently stopped
    lining up with the CSVs they were meant to describe.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in video_name)


# ----------------------------------------------------------------------
# Video metadata (fps + frame size)
# ----------------------------------------------------------------------
def load_video_meta(path):
    """
    Loads {video_id: {"fps":, "width":, "height":, ...}} written by
    detect_and_track.py (and by build_fps_lookup.py as a standalone fallback).

    Also accepts the older flat {video_id: fps} format so existing
    video_fps_lookup.json files keep working - those entries simply have no
    frame size, and the caller falls back to inferring it.
    """
    if not os.path.exists(path):
        return {}

    with open(path) as f:
        raw = json.load(f)

    meta = {}
    for vid, val in raw.items():
        if isinstance(val, dict):
            meta[vid] = val
        else:  # legacy flat {video_id: fps}
            meta[vid] = {"fps": float(val)}
    return meta


def resolve_fps(video_id, meta, default_fps=DEFAULT_FPS):
    """
    Trust order: measured value from metadata, then an '_NNfps' pattern in the
    filename, then the default. Returns (fps, source) so an assumed value stays
    a visible assumption rather than a silent one.
    """
    entry = meta.get(video_id) or {}
    fps = entry.get("fps")
    if fps:
        return float(fps), "measured"

    match = re.search(r"(\d{2,3})\s*fps", video_id, re.IGNORECASE)
    if match:
        return float(match.group(1)), "filename"

    return float(default_fps), "assumed"


def resolve_frame_width(video_id, meta, tracking_df=None):
    """
    Frame width in pixels, used as the normalisation scale.

    Trust order:
      1. the real width recorded alongside the tracking run
      2. a WIDTH_HEIGHT pattern in the filename (the corpus uses names like
         '13020032_3840_2160_30fps')
      3. the furthest right-hand box edge actually observed in the tracking data

    Falling back to (3) slightly UNDER-estimates the width, since no vehicle is
    guaranteed to touch the frame edge. That is acceptable: it is a consistent
    per-video scale, which is all the features need. Returns (width, source).
    """
    entry = meta.get(video_id) or {}
    if entry.get("width"):
        return float(entry["width"]), "measured"

    match = re.search(r"_(\d{3,4})_(\d{3,4})_", video_id)
    if match:
        # Filenames encode WIDTH_HEIGHT; a vertical clip has the smaller first.
        return float(match.group(1)), "filename"

    if tracking_df is not None and not tracking_df.empty:
        observed = float((tracking_df["cx"] + tracking_df["width"] / 2).max())
        if observed > 0:
            return observed, "observed"

    return 1920.0, "assumed"


# ----------------------------------------------------------------------
# Windowing
# ----------------------------------------------------------------------
def frames_per_window(fps, window_seconds=DEFAULT_WINDOW_SECONDS):
    """Frame count covering window_seconds of real time for this video's fps."""
    return max(1, int(round(window_seconds * fps)))


def assign_windows(df, fps, window_seconds=DEFAULT_WINDOW_SECONDS):
    """Adds a window_id column bucketing frames into fixed-duration windows."""
    out = df.copy()
    out["window_id"] = out["frame"] // frames_per_window(fps, window_seconds)
    return out


# ----------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------
def normalize_positions(df, frame_width):
    """
    Adds nx/ny columns: pixel coordinates divided by the FRAME WIDTH.

    Both axes use the width so the aspect ratio survives - dividing y by the
    height instead would stretch vertical motion on non-square frames and make
    a lane change look like a different manoeuvre depending on orientation.
    """
    out = df.copy()
    scale = float(frame_width) if frame_width else 1.0
    out["nx"] = out["cx"] / scale
    out["ny"] = out["cy"] / scale
    return out


# ----------------------------------------------------------------------
# Per-vehicle geometry
# ----------------------------------------------------------------------
def compute_speeds(positions, fps, max_speed=MAX_PLAUSIBLE_SPEED):
    """
    positions: sorted (frame, nx, ny) rows for ONE vehicle in ONE window,
    already normalised - accepts a list of tuples or an (N, 3) array.
    Returns per-step speed in frame-widths/second.

    Elapsed time uses the real frame gap rather than assuming consecutive
    frames, since a track can drop out for a frame or two and reappear.
    Steps faster than max_speed are dropped as tracker ID-switches.

    Vectorised: this runs once per vehicle per window - tens of thousands of
    calls over the corpus - so the original per-step Python loop dominated the
    feature-extraction time.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or len(pos) < 2:
        return np.array([])

    dt = np.diff(pos[:, 0]) / fps
    dist = np.hypot(np.diff(pos[:, 1]), np.diff(pos[:, 2]))

    valid = dt > 0
    if not valid.any():
        return np.array([])

    speeds = dist[valid] / dt[valid]
    return speeds[speeds <= max_speed]


def compute_displacement_ratio(positions):
    """
    Straight-line distance / total path length: 1.0 for a vehicle travelling
    dead straight, approaching 0 for one weaving or circling. A ratio, so it was
    already resolution-independent and is unchanged by normalisation.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or len(pos) < 2:
        return 1.0

    xs, ys = pos[:, 1], pos[:, 2]

    straight = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    total_path = float(np.hypot(np.diff(xs), np.diff(ys)).sum())

    if total_path < 1e-9:
        return 0.0  # never moved - stuck, which is a risk signal, not efficiency

    value = straight / total_path
    return float(value) if np.isfinite(value) else 0.5


def group_indices(values):
    """
    Row indices grouped by value: returns (unique_values, [index_array, ...]).

    A sort-and-split over a numpy array, used instead of pandas .groupby() on
    the hot paths. Profiling compute_features showed 41% of its runtime inside
    nearest_neighbor_distances, and almost all of that was pandas column
    extraction (DataFrame.__getitem__) repeated across ~11k tiny per-frame
    groups - not the distance arithmetic. Pulling the columns out to numpy once
    and grouping by index removes that per-group overhead entirely.
    """
    values = np.asarray(values)
    if len(values) == 0:
        return np.array([]), []
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    cuts = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
    uniques = ordered[np.concatenate(([0], cuts))]
    return uniques, np.split(order, cuts)


def nearest_neighbors_from_arrays(ids, xs, ys):
    """
    Core nearest-neighbour computation on plain arrays.
    Returns (ids, min_distance) for one frame, or (empty, empty) below 2 vehicles.
    """
    n = len(ids)
    if n < 2:
        return ids[:0], np.array([])

    # Pairwise distances in one shot, then mask the diagonal so a vehicle is
    # never its own nearest neighbour.
    d = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
    np.fill_diagonal(d, np.inf)
    return ids, d.min(axis=1)


def nearest_neighbor_distances(frame_group):
    """
    frame_group: rows for ONE frame with track_id/nx/ny.
    Returns {track_id: distance to nearest other vehicle}, in frame-widths.
    Empty when fewer than two vehicles are present.

    DataFrame-shaped convenience wrapper around nearest_neighbors_from_arrays.
    """
    ids, mins = nearest_neighbors_from_arrays(
        frame_group["track_id"].to_numpy(),
        frame_group["nx"].to_numpy(),
        frame_group["ny"].to_numpy(),
    )
    return dict(zip(ids, mins))


def all_nearest_neighbor_distances(df):
    """
    Every per-frame nearest-neighbour distance for a whole video, as one array.

    Used to build the distribution the proximity threshold is derived from.
    Doing it in one pass avoids a pandas groupby per frame.
    """
    frames = df["frame"].to_numpy()
    ids = df["track_id"].to_numpy()
    xs = df["nx"].to_numpy()
    ys = df["ny"].to_numpy()

    out = []
    _, index_groups = group_indices(frames)
    for idx in index_groups:
        if len(idx) < 2:
            continue
        _, mins = nearest_neighbors_from_arrays(ids[idx], xs[idx], ys[idx])
        out.append(mins)

    return np.concatenate(out) if out else np.array([])


def guard_value(value, fallback=0.0):
    """Replace NaN/Inf with a safe fallback."""
    if value is None or not np.isfinite(value):
        return fallback
    return float(value)


# ----------------------------------------------------------------------
# The 5 features, for one video's tracking data
# ----------------------------------------------------------------------
def compute_features(track_df, fps, frame_width, proximity_threshold,
                     deceleration_threshold, window_seconds=DEFAULT_WINDOW_SECONDS):
    """
    One row per (window, vehicle) with the 5 risk features.

    track_df needs: frame, track_id, cx, cy (the tracking CSV schema).
    Thresholds are in normalised units - frame-widths and frame-widths/sec.

    This is the ONLY implementation of these features in the project; the
    offline pipeline and the dashboard's live upload path both call it, so a
    clip analysed in the dashboard is measured exactly like the training corpus.
    """
    df = normalize_positions(track_df, frame_width)
    df = assign_windows(df, fps, window_seconds)

    # Columns are pulled into numpy ONCE here. Everything below groups by index
    # instead of calling pandas .groupby()/[] per window, per frame and per
    # track - see group_indices() for the profiling result that motivated it.
    frames = df["frame"].to_numpy()
    track_ids = df["track_id"].to_numpy()
    nxs = df["nx"].to_numpy()
    nys = df["ny"].to_numpy()
    windows = df["window_id"].to_numpy()

    rows = []
    window_values, window_groups = group_indices(windows)

    for window_id, widx in zip(window_values, window_groups):
        w_frames = frames[widx]
        w_tracks = track_ids[widx]
        w_x = nxs[widx]
        w_y = nys[widx]

        congestion_score = len(np.unique(w_tracks))

        # Flat set of (frame, track_id) pairs closer than the threshold, so the
        # per-position check below is a single hash lookup. Ints are cast
        # explicitly so a numpy scalar can never miss a match.
        close_pairs = set()
        frame_values, frame_groups = group_indices(w_frames)
        for frame_num, fidx in zip(frame_values, frame_groups):
            if len(fidx) < 2:
                continue
            ids, mins = nearest_neighbors_from_arrays(w_tracks[fidx], w_x[fidx], w_y[fidx])
            near = mins < proximity_threshold
            for tid in ids[near]:
                close_pairs.add((int(frame_num), int(tid)))

        track_values, track_groups = group_indices(w_tracks)
        for track_id, tidx in zip(track_values, track_groups):
            # A track's rows must be in time order before differencing.
            sel = tidx[np.argsort(w_frames[tidx], kind="stable")]
            positions = np.column_stack((w_frames[sel], w_x[sel], w_y[sel]))

            speeds = compute_speeds(positions, fps)

            speed_variance = guard_value(float(np.var(speeds))) if len(speeds) >= 2 else 0.0

            if len(speeds) >= 2:
                drops = speeds[:-1] - speeds[1:]
                deceleration_events = int(np.sum(drops >= deceleration_threshold))
            else:
                deceleration_events = 0

            tid_int = int(track_id)
            proximity_interactions = sum(
                1 for f in w_frames[sel]
                if (int(f), tid_int) in close_pairs
            )

            rows.append({
                "window_id": int(window_id),
                "track_id": track_id,
                "displacement_ratio": compute_displacement_ratio(positions),
                "speed_variance": speed_variance,
                "proximity_interactions": proximity_interactions,
                "deceleration_events": deceleration_events,
                "congestion_score": int(congestion_score),
            })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Clustering transform
# ----------------------------------------------------------------------
def clustering_frame(df):
    """
    The feature matrix used for K-means pseudo-labelling.

    Two adjustments versus feeding the raw features straight in:

    * displacement_ratio is flipped, so every column points the same way
      (larger == riskier). Otherwise the cluster ranking, which averages the
      scaled columns, would treat smooth straight-line driving as risky.

    * speed_variance is log1p'd. It spans several orders of magnitude and is
      extremely heavy-tailed; StandardScaler does not tame that, so without the
      log the single column dominates the composite and "High Risk" collapses
      into "whatever had the largest variance" - which, before normalisation,
      mostly meant "whatever was filmed at the highest resolution".
    """
    out = pd.DataFrame(index=df.index)
    out["risk_displacement"] = 1.0 - df["displacement_ratio"]
    out["log_speed_variance"] = np.log1p(df["speed_variance"])
    out["proximity_interactions"] = df["proximity_interactions"]
    out["deceleration_events"] = df["deceleration_events"]
    out["congestion_score"] = df["congestion_score"]
    return out


CLUSTER_FEATURES = ["risk_displacement", "log_speed_variance",
                    "proximity_interactions", "deceleration_events",
                    "congestion_score"]

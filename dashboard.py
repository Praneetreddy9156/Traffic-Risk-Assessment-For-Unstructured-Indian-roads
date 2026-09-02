"""
dashboard.py

Smart-city-style dashboard for the Traffic Risk & Congestion Intelligence
project. Every panel is driven by real pipeline output. Visual system: theme.py.
Feature math: pipeline_core.py (shared with the offline pipeline).

Pages:
  Command Center  - KPI cluster + risk distribution, all real totals
  Model Lab       - feature importance, confusion matrices, cluster validation
  Congestion      - the congestion_analysis.py stage, previously not surfaced
  Video Explorer  - per-video gauges, risk/congestion timeline, heatmap, radar
  Data Explorer   - filterable table + CSV export
  Analyze Video   - upload -> live detection + tracking -> classification ->
                    gauges, incident feed, annotated replay, report card

WHAT CHANGED AND WHY
--------------------
1. ANALYSIS RESULTS NOW SURVIVE A RERUN.
   The page was gated on `if not st.button("Run Analysis"): return`. st.button
   is True only on the single rerun straight after the click, and Streamlit
   reruns the whole script on EVERY interaction - so touching anything
   afterwards (the replay button, the report download, the raw-predictions
   expander) re-ran the script, the button read False, the function returned
   early, and the entire analysis vanished. The annotated replay button could
   never work, not once. Results now live in st.session_state and the controls
   act on stored state.

2. THE CONTROLS ACTUALLY DO SOMETHING.
   Detection confidence, IoU, window length and the choice of classifier are
   real inputs now, not constants buried in a dict. Detection settings are
   batched in a form so moving a slider does not kick off a re-analysis.

3. UPLOADS ARE MEASURED LIKE THE CORPUS.
   The upload path ran at conf=0.25 with library defaults for iou/imgsz while
   the corpus ran conf=0.35/iou=0.45/imgsz=640, and it computed its own copy of
   the feature formulas. Both now come from pipeline_core, so a clip analysed
   here is measured exactly like the training data - and in resolution-invariant
   units, which is what stopped phone footage from ever reaching High Risk.

4. Assorted correctness fixes: incident snapshots merge on (window_id, track_id)
   rather than track_id alone (which multiplied rows for any vehicle appearing
   in more than one window); box colours are no longer BGR-swapped after the
   RGB conversion; uploads appear in Video Explorer; deprecated
   use_container_width replaced with width="stretch".

Run:
    streamlit run src/dashboard.py
"""

import os
import io
import re
import sys
import json
import shutil
import tempfile
import subprocess

import numpy as np
import pandas as pd
import joblib
import cv2
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")          # headless: Streamlit has no GUI event loop
import matplotlib.pyplot as plt
from ultralytics import YOLO

# `streamlit run src/dashboard.py` puts src/ on sys.path automatically, but
# other entry points (streamlit.testing's AppTest, importing this module, a
# different working directory) do not - and then the sibling imports below fail
# with ModuleNotFoundError. Adding the file's own directory makes the import
# work regardless of how the app is launched.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pipeline_core
from pipeline_core import PATHS, DETECTION, FEATURES, safe_video_id

import theme
from theme import (PALETTE, RISK_COLORS, RF_COLOR, XGB_COLOR, FONT_COLOR, MUTED,
                   GRID, CYCLE)

FEATURE_DISPLAY_NAMES = {
    "displacement_ratio": "Displacement Ratio",
    "speed_variance": "Speed Variance",
    "proximity_interactions": "Proximity Interactions",
    "deceleration_events": "Deceleration Events",
    "congestion_score": "Congestion Score",
}

STATUS_MESSAGES = [
    "Detecting vehicles...", "Tracking motion across frames...",
    "Measuring speed & spacing...", "Flagging risky moments...",
]

# Vehicle-class icons for KPI cards (best-effort match on class name substrings)
CLASS_ICONS = [
    ("motor", "🏍"), ("scooter", "🛵"), ("bike", "🏍"), ("cycle", "🚲"),
    ("car", "🚗"), ("bus", "🚌"), ("truck", "🚚"),
    ("auto", "🛺"), ("rick", "🛺"), ("person", "🚶"),
]

# Which column drives the risk shown across the app. The K-means label is the
# target the classifiers were trained on, so it is offered as the reference.
CLASSIFIER_COLUMNS = {
    "Random Forest": "rf_prediction",
    "XGBoost": "xgb_prediction",
    "K-means label": "risk_label",
}

st.set_page_config(page_title="Traffic Risk & Congestion Intelligence",
                   layout="wide", page_icon="🚦", initial_sidebar_state="collapsed")

# --- Session state, initialised in one place ---
st.session_state.setdefault("analysis", None)     # stored upload analysis
st.session_state.setdefault("accent", PALETTE["indigo"])
st.session_state.setdefault("classifier", "Random Forest")


# ===========================================================================
# Cached loading
# ===========================================================================
@st.cache_data
def load_predictions():
    return pd.read_csv(PATHS["risk_predictions"])


@st.cache_data
def load_metrics():
    with open(PATHS["risk_metrics"]) as f:
        return json.load(f)


@st.cache_data
def load_video_meta():
    return pipeline_core.load_video_meta(PATHS["video_meta"]) or \
           pipeline_core.load_video_meta(PATHS["legacy_fps_lookup"])


@st.cache_data
def load_risk_thresholds():
    with open(PATHS["risk_thresholds"]) as f:
        return json.load(f)


@st.cache_data
def load_congestion():
    """congestion_analysis.py output. Optional - the page degrades gracefully."""
    windows = os.path.join(PATHS["congestion_dir"], "congestion_windows_labeled.csv")
    summary = os.path.join(PATHS["congestion_dir"], "congestion_video_summary.csv")
    thresholds = os.path.join(PATHS["congestion_dir"], "congestion_thresholds.json")
    if not (os.path.exists(windows) and os.path.exists(summary)):
        return None
    with open(thresholds) as f:
        thr = json.load(f)
    return pd.read_csv(windows), pd.read_csv(summary), thr



@st.cache_data
def load_comparison(name):
    """Read one comparison table written by the src/compare_*.py scripts."""
    path = os.path.join("outputs", "comparison", name)
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_resource
def load_yolo_model():
    return YOLO(PATHS["yolo_weights"])


@st.cache_resource
def load_risk_classifiers():
    d = PATHS["risk_models_dir"]
    return (joblib.load(os.path.join(d, "random_forest.joblib")),
            joblib.load(os.path.join(d, "xgboost.joblib")),
            joblib.load(os.path.join(d, "label_encoder.joblib")))


@st.cache_resource
def load_pseudolabeler():
    """
    The Stage-1 K-means labeller (scaler + kmeans + cluster->label map).

    Needed so an uploaded clip can be given a real K-means label rather than a
    copy of the Random Forest prediction. Without this the "K-means label"
    option on the Analyze page silently showed RF's output under a different
    name, which would make the two look like they always agreed.
    """
    d = PATHS["risk_models_dir"]
    scaler_path = os.path.join(d, "scaler.joblib")
    kmeans_path = os.path.join(d, "kmeans.joblib")
    if not (os.path.exists(scaler_path) and os.path.exists(kmeans_path)):
        return None
    mapping = load_metrics()["pseudolabeling"]["cluster_to_label"]
    # JSON object keys are strings; cluster indices are ints.
    return (joblib.load(scaler_path), joblib.load(kmeans_path),
            {int(k): v for k, v in mapping.items()})


# `_df` is skipped by Streamlit's hasher, which would leave these functions with
# a CONSTANT cache key - they would never recompute even after the underlying
# data changed. `token` is hashed and carries a cheap fingerprint of the data.
@st.cache_data
def derive_congestion_threshold(_df, token):
    ws = (_df[["video_id", "window_id", "congestion_score"]]
          .drop_duplicates()["congestion_score"].to_numpy().reshape(-1, 1))
    if len(np.unique(ws)) < 2:
        return float(ws.mean()), float(ws.max())
    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(ws)
    return float(np.mean(sorted(km.cluster_centers_.flatten()))), float(ws.max())


@st.cache_data
def global_vehicle_counts(_df, token):
    """Real distinct-vehicle counts by class, aggregated across tracking CSVs."""
    totals = {}
    for vid in _df["video_id"].unique():
        t = load_tracking_csv(vid)
        if t is None or t.empty or "class_name" not in t.columns:
            continue
        dom = t.groupby("track_id")["class_name"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        for cls, cnt in dom.value_counts().items():
            totals[cls] = totals.get(cls, 0) + int(cnt)
    return totals


@st.cache_data
def feature_ranges(df):
    return {f: (float(df[f].min()), float(df[f].max())) for f in FEATURES}


@st.cache_data
def load_tracking_csv(video_id):
    """
    Looks in the corpus tracking directory first, then the uploads directory,
    so an analysed upload behaves like any other clip in Video Explorer instead
    of being invisible to it.
    """
    for base in (PATHS["tracking_dir"], PATHS["uploads_dir"]):
        path = os.path.join(base, f"{video_id}.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def data_token(df):
    """Cheap fingerprint used to key the caches above."""
    return f"{len(df)}-{df['video_id'].nunique()}"


# ===========================================================================
# Upload store (keeps the research corpus untouched)
# ===========================================================================
def load_uploads_store():
    p = PATHS["uploads_store"]
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_upload_to_store(feat_df, track_df, video_name):
    """
    Persists one analysed upload. risk_predictions.csv is NEVER touched.

    Stores the PER-VEHICLE feature rows, not the one-row-per-window summary the
    old version saved - counting classes off the summary counted windows and
    reported them as vehicles.
    """
    os.makedirs(os.path.dirname(PATHS["uploads_store"]), exist_ok=True)
    os.makedirs(PATHS["uploads_dir"], exist_ok=True)

    safe_id = "upload__" + safe_video_id(os.path.splitext(video_name)[0])

    dom = track_df.groupby("track_id")["class_name"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    out = feat_df.copy()
    out["video_id"] = safe_id
    out["class_name"] = out["track_id"].map(dom)

    existing = load_uploads_store()
    if not existing.empty and "video_id" in existing.columns:
        existing = existing[existing["video_id"] != safe_id]   # replace if re-analysed
        combined = pd.concat([existing, out], ignore_index=True)
    else:
        combined = out
    combined.to_csv(PATHS["uploads_store"], index=False)

    track_df.to_csv(os.path.join(PATHS["uploads_dir"], f"{safe_id}.csv"), index=False)
    return safe_id


def reset_uploads_store():
    removed = 0
    if os.path.exists(PATHS["uploads_store"]):
        os.remove(PATHS["uploads_store"])
        removed += 1
    if os.path.isdir(PATHS["uploads_dir"]):
        for f in os.listdir(PATHS["uploads_dir"]):
            if f.endswith(".csv"):
                os.remove(os.path.join(PATHS["uploads_dir"], f))
                removed += 1
    return removed


def uploads_summary(uploads_df, label_order, label_col):
    if uploads_df.empty:
        return {"videos": 0, "windows": 0, "high": 0, "vehicles": 0, "by_class": {}}

    col = label_col if label_col in uploads_df.columns else "risk_label"
    n_videos = uploads_df["video_id"].nunique()
    n_windows = uploads_df[["video_id", "window_id"]].drop_duplicates().shape[0]
    # distinct windows, not rows - this is shown beside n_windows as
    # "<label> windows", so both must be counted the same way
    n_high = (uploads_df.loc[uploads_df[col] == label_order[-1],
                             ["video_id", "window_id"]].drop_duplicates().shape[0]
              if col in uploads_df else 0)

    # Distinct vehicles, not rows: a vehicle spans several windows.
    by_class = {}
    if "class_name" in uploads_df.columns:
        vehicles = uploads_df.dropna(subset=["class_name"]).drop_duplicates(
            subset=["video_id", "track_id"])
        by_class = vehicles["class_name"].value_counts().to_dict()

    # Raw per-class detection counts straight from the saved tracking CSVs.
    # Without this, a class that was detected but never won a track's majority
    # vote disappears from the summary entirely, making it look as though the
    # detector never produced it.
    raw_by_class = {}
    for vid in uploads_df["video_id"].unique():
        t = load_tracking_csv(vid)
        if t is None or t.empty or "class_name" not in t.columns:
            continue
        for cls, n in t["class_name"].value_counts().items():
            raw_by_class[cls] = raw_by_class.get(cls, 0) + int(n)

    return {"videos": n_videos, "windows": n_windows, "high": n_high,
            "vehicles": int(sum(by_class.values())), "by_class": by_class,
            "raw_by_class": raw_by_class,
            "detected_only": [c for c in raw_by_class if c not in by_class]}


# ===========================================================================
# Small helpers
# ===========================================================================
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def hex_to_rgb_tuple(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgba(hex_color, alpha):
    """
    Translucent colour as an rgba() string.

    Appending alpha to a hex string ("#FB7185" + "55") produces 8-digit hex,
    which Plotly rejects outright:
        Invalid value of type 'builtins.str' received for the 'fillcolor'
        property of scatterpolar -- Received value: '#FB718555'
    That is what made the radar chart raise instead of render. Plotly accepts
    rgba(), so build that rather than relying on CSS-style hex alpha.
    """
    r, g, b = hex_to_rgb_tuple(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


def compute_risk_score_index(labels_series, label_order):
    rmap = {l: (i / (len(label_order) - 1) if len(label_order) > 1 else 0)
            for i, l in enumerate(label_order)}
    ranks = labels_series.map(rmap).dropna()
    return float(ranks.mean() * 100) if len(ranks) else 0.0


def dataset_averages(df, label_order, label_col):
    avg_cong = df.drop_duplicates(subset=["video_id", "window_id"])["congestion_score"].mean()
    return float(avg_cong), compute_risk_score_index(df[label_col], label_order)


def relative_framing(value, avg):
    if not avg:
        return ""
    d = (value - avg) / avg * 100
    return f"{abs(d):.0f}% {'higher' if d >= 0 else 'lower'} than a typical clip"


def normalize_feature_dict(raw, ranges):
    out = {}
    for f in FEATURES:
        lo, hi = ranges.get(f, (0, 1))
        out[f] = 0.5 if (hi - lo) < 1e-9 else float(np.clip((raw.get(f, lo) - lo) / (hi - lo), 0, 1))
    return out


def describe_incident(row, dataset_df):
    def pr(col, v):
        return (dataset_df[col] <= v).mean() * 100
    reasons = []
    if pr("proximity_interactions", row["proximity_interactions"]) > 85:
        reasons.append("multiple close encounters with nearby vehicles")
    if pr("speed_variance", row["speed_variance"]) > 85:
        reasons.append("highly erratic speed changes")
    if pr("deceleration_events", row["deceleration_events"]) > 85:
        reasons.append("repeated hard braking")
    if row["displacement_ratio"] < 0.3:
        reasons.append("indecisive, non-linear movement")
    if pr("congestion_score", row["congestion_score"]) > 85:
        reasons.append("heavy surrounding traffic")
    if not reasons:
        reasons.append("an elevated overall risk profile")
    return "Flagged for " + ", ".join(reasons) + "."


def icon_for_class(class_name):
    cl = str(class_name).lower()
    for key, icon in CLASS_ICONS:
        if key in cl:
            return icon
    return "🚗"


def filter_short_tracks(df, min_len):
    c = df.groupby("track_id").size()
    return df[df["track_id"].isin(c[c >= min_len].index)].copy()


def detection_breakdown(raw_df, kept_df):
    """
    A full accounting of what the detector saw — nothing dropped silently.

    Reporting only "distinct vehicles by dominant class" hides two things:

      * Classes that were detected but never won a track's majority vote. One
        clip had 20 truck detections on a track that was car in 401 frames; the
        dominant-class vote is right about the physical vehicle, but showing
        only that made it look like trucks were never detected at all.
      * Tracks removed by the min-track-length filter, which discards likely
        false positives but should be visible rather than invisible.

    So this returns per-class DETECTION counts and per-class VEHICLE counts
    side by side, plus what the filter removed.
    """
    raw_counts = raw_df["class_name"].value_counts().to_dict() if len(raw_df) else {}

    if len(kept_df):
        dom = kept_df.groupby("track_id")["class_name"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        vehicle_counts = dom.value_counts().to_dict()
        kept_counts = kept_df["class_name"].value_counts().to_dict()
    else:
        vehicle_counts, kept_counts = {}, {}

    classes = sorted(set(raw_counts) | set(vehicle_counts),
                     key=lambda c: -raw_counts.get(c, 0))
    rows = [{
        "class": c,
        "vehicles": int(vehicle_counts.get(c, 0)),
        "detections (kept)": int(kept_counts.get(c, 0)),
        "detections (raw)": int(raw_counts.get(c, 0)),
    } for c in classes]

    return {
        "table": pd.DataFrame(rows),
        "raw_tracks": int(raw_df["track_id"].nunique()) if len(raw_df) else 0,
        "kept_tracks": int(kept_df["track_id"].nunique()) if len(kept_df) else 0,
        "raw_detections": int(len(raw_df)),
        "kept_detections": int(len(kept_df)),
        # Classes detected that no vehicle ended up being labelled as.
        "classes_without_vehicles": [c for c in raw_counts if not vehicle_counts.get(c)],
    }


def summarize_window_risk(df, label_order, label_col):
    """One row per window: the riskiest vehicle in it."""
    rmap = {l: i for i, l in enumerate(label_order)}
    d = df.copy()
    d["_r"] = d[label_col].map(rmap)
    return d.loc[d.groupby("window_id")["_r"].idxmax()].drop(columns="_r").reset_index(drop=True)


# ===========================================================================
# Live tracking for uploads
# ===========================================================================
def run_tracking_on_video(video_path, model, conf, iou, imgsz, progress_callback=None):
    """
    Mirrors detect_and_track.py exactly - same tracker, same thresholds from
    pipeline_core.DETECTION - so an uploaded clip is measured the same way as
    the corpus the models were trained on. Also returns the frame WIDTH, which
    the risk features need to normalise distances.
    """
    # The tracker has its OWN confidence gates that silently override the conf
    # argument. botsort_traffic.yaml ships track_high_thresh=0.25 and
    # new_track_thresh=0.25, so a detection below 0.25 never becomes a track no
    # matter what conf is set to — measured: conf=0.25, 0.15 and 0.05 all
    # produced byte-identical results (92 vehicles), making the slider dead
    # below 0.25. Write a per-run copy with those gates tied to the requested
    # confidence so the control actually does what it says.
    tracker_cfg = DETECTION["tracker"]
    try:
        base = open(tracker_cfg).read()
        lowered = re.sub(r"^track_high_thresh:.*$", f"track_high_thresh: {conf}", base, flags=re.M)
        lowered = re.sub(r"^new_track_thresh:.*$", f"new_track_thresh: {conf}", lowered, flags=re.M)
        lowered = re.sub(r"^track_low_thresh:.*$", f"track_low_thresh: {max(conf/2, 0.01):.3f}",
                         lowered, flags=re.M)
        tmp_tracker = os.path.join(tempfile.mkdtemp(), "botsort_run.yaml")
        with open(tmp_tracker, "w") as f:
            f.write(lowered)
        tracker_cfg = tmp_tracker
    except Exception:
        pass  # fall back to the project config rather than failing the analysis

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or pipeline_core.DEFAULT_FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()

    rows = []
    # persist=False for the same reason as detect_and_track.py: the YOLO model is
    # cached across the whole session, so persisting the tracker would carry
    # BoT-SORT's motion-compensation buffer from one uploaded clip to the next.
    # Analysing a 1080p clip and then a 720p one would silently disable camera
    # motion compensation for the second - and every clip after it.
    for fi, r in enumerate(model.track(source=video_path, tracker=tracker_cfg,
                                       stream=True, conf=conf, iou=iou, imgsz=imgsz,
                                       persist=False, verbose=False)):
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xywh.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            names = r.names
            for (cx, cy, w, h), tid, cid, cf in zip(boxes, ids, classes, confs):
                rows.append({"frame": fi, "track_id": int(tid), "class_id": int(cid),
                             "class_name": names.get(int(cid), str(cid)),
                             "cx": float(cx), "cy": float(cy),
                             "width": float(w), "height": float(h),
                             "confidence": float(cf)})
        if progress_callback:
            progress_callback(min((fi + 1) / total, 1.0))

    return pd.DataFrame(rows), fps, frame_width, frame_height


@st.cache_data
def compute_all_speeds(video_id, fps, frame_width, window_seconds):
    t = load_tracking_csv(video_id)
    if t is None or t.empty:
        return np.array([])
    n = pipeline_core.normalize_positions(t, frame_width)
    n = pipeline_core.assign_windows(n, fps, window_seconds)
    out = []
    for _, tg in n.groupby(["window_id", "track_id"]):
        pos = tg.sort_values("frame")[["frame", "nx", "ny"]].to_numpy()
        out.extend(pipeline_core.compute_speeds(pos, fps).tolist())
    return np.array(out)


# ===========================================================================
# Charts
# ===========================================================================
def _lerp(c1, c2, t):
    r1, g1, b1 = hex_to_rgb_tuple(c1)
    r2, g2, b2 = hex_to_rgb_tuple(c2)
    return "#%02x%02x%02x" % tuple(
        int(max(0, min(255, a + (b - a) * t))) for a, b in
        ((r1, r2), (g1, g2), (b1, b2)))


def gradient_steps(max_value, high_cut, n=40):
    steps = []
    for i in range(n):
        a, b = max_value * i / n, max_value * (i + 1) / n
        mid = (a + b) / 2
        if mid <= high_cut:
            color = _lerp(PALETTE["emerald"], PALETTE["amber"],
                          min(max(mid / max(high_cut, 1e-6), 0), 1))
        else:
            color = _lerp(PALETTE["amber"], PALETTE["rose"],
                          min(max((mid - high_cut) / max(max_value - high_cut, 1e-6), 0), 1))
        steps.append({"range": [a, b], "color": color})
    return steps


def gauge_chart(value, max_value, title, high_cut):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": title, "font": {"size": 15, "color": FONT_COLOR, "family": "Sora"}},
        number={"font": {"size": 34, "color": FONT_COLOR, "family": "Sora"}},
        gauge={"axis": {"range": [0, max_value], "tickcolor": MUTED},
               "bar": {"color": "white", "thickness": 0.18},
               "bgcolor": PALETTE["card"], "borderwidth": 0,
               "steps": gradient_steps(max_value, high_cut)},
    ))
    fig.update_layout(height=260, margin=dict(t=48, b=8, l=24, r=24),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": FONT_COLOR})
    return fig


def radar_chart(vals_norm, title=""):
    keys = list(FEATURE_DISPLAY_NAMES.keys())
    labels = list(FEATURE_DISPLAY_NAMES.values())
    vals = [vals_norm.get(k, 0) for k in keys]
    vals += vals[:1]
    labels += labels[:1]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=labels, fill="toself",
                                    line_color=PALETTE["rose"],
                                    fillcolor=rgba(PALETTE["rose"], 0.33)))
    fig.update_layout(polar=dict(bgcolor=PALETTE["card"],
                                 radialaxis=dict(visible=True, showticklabels=False,
                                                 range=[0, 1], gridcolor=GRID),
                                 angularaxis=dict(gridcolor=GRID)),
                      showlegend=False, height=380,
                      title={"text": title, "font": {"color": FONT_COLOR}},
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": FONT_COLOR})
    return fig


def chart_vehicle_mix(track_df):
    dom = track_df.groupby("track_id")["class_name"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    c = dom.value_counts()
    fig = go.Figure(go.Pie(labels=c.index, values=c.values, hole=0.52,
                           marker=dict(colors=CYCLE[:len(c)],
                                       line=dict(color=PALETTE["bg"], width=2))))
    fig.update_layout(height=380, title={"text": "Vehicle mix (distinct vehicles)",
                                         "font": {"color": FONT_COLOR}},
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": FONT_COLOR},
                      legend=dict(font=dict(color=FONT_COLOR)))
    return fig


def chart_risk_congestion(clip_df, cong_thr, label_order, label_col, window_seconds):
    clip_df = clip_df.drop_duplicates(subset=["window_id"]).sort_values("window_id")
    lmap = {l: i for i, l in enumerate(label_order)}
    y = clip_df[label_col].map(lmap)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Risk classification", "Congestion"))
    fig.add_trace(go.Scatter(x=clip_df["window_id"], y=y, mode="lines+markers",
                             line=dict(color="#3d4c66", width=1),
                             marker=dict(size=11,
                                         color=[RISK_COLORS.get(l, MUTED) for l in clip_df[label_col]],
                                         line=dict(width=1.2, color="white")),
                             customdata=clip_df[label_col],
                             hovertemplate="Window %{x}<br>Risk: %{customdata}<extra></extra>",
                             showlegend=False), row=1, col=1)
    fig.update_yaxes(tickvals=list(range(len(label_order))), ticktext=label_order, row=1, col=1)
    ymax = clip_df["congestion_score"].max() * 1.3 + 1
    fig.add_hrect(y0=0, y1=cong_thr, fillcolor=PALETTE["emerald"], opacity=0.08,
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=cong_thr, y1=ymax, fillcolor=PALETTE["rose"], opacity=0.08,
                  line_width=0, row=2, col=1)
    fig.add_trace(go.Scatter(x=clip_df["window_id"], y=clip_df["congestion_score"],
                             mode="lines", line=dict(color=PALETTE["cyan"], width=2.4),
                             hovertemplate="Window %{x}<br>Vehicles: %{y}<extra></extra>",
                             showlegend=False), row=2, col=1)
    fig.update_xaxes(title_text=f"Window index ({window_seconds:g}s each)", row=2, col=1)
    fig.update_layout(height=540, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PALETTE["card"],
                      font={"color": FONT_COLOR}, margin=dict(t=56, b=36), showlegend=False)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def chart_trajectory_heatmap(track_df):
    fig = go.Figure(go.Histogram2dContour(x=track_df["cx"], y=track_df["cy"],
                                          colorscale="Plasma",
                                          contours=dict(coloring="fill"), line=dict(width=0)))
    fig.update_yaxes(autorange="reversed", title="y (px)", gridcolor=GRID)
    fig.update_xaxes(title="x (px)", gridcolor=GRID)
    fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PALETTE["card"],
                      font={"color": FONT_COLOR}, margin=dict(t=28),
                      title={"text": "Where vehicles travel", "font": {"color": FONT_COLOR}})
    return fig


def chart_speed_distribution(speeds):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=speeds, histnorm="probability density",
                               marker_color=PALETTE["cyan"], opacity=0.5, name="Speed"))
    if len(speeds) > 10 and np.std(speeds) > 0:
        kde = gaussian_kde(speeds)
        xs = np.linspace(speeds.min(), speeds.max(), 300)
        fig.add_trace(go.Scatter(x=xs, y=kde(xs), mode="lines",
                                 line=dict(color=PALETTE["amber"], width=3), name="Density"))
    fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PALETTE["card"],
                      font={"color": FONT_COLOR},
                      xaxis_title="Speed (frame-widths / sec)", yaxis_title="Density",
                      legend=dict(orientation="h", y=1.15, font=dict(color=FONT_COLOR)),
                      margin=dict(t=48))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def snapshot_frame(video_path, frame_number, boxes_df):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for _, row in boxes_df.iterrows():
        cx, cy, w, h = row["cx"], row["cy"], row["width"], row["height"]
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        # The frame is RGB at this point, so the colour must be RGB too. The old
        # code converted to RGB and then drew with a BGR tuple, which swapped
        # red and blue - "Safe" green survived, but High Risk drew as a blue-ish
        # tint instead of rose.
        color = hex_to_rgb_tuple(RISK_COLORS.get(row.get("risk_label", "Safe"),
                                                 PALETTE["emerald"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame


def build_annotated_video(source_path, track_df, window_map, window_frames, fps, tmp_dir):
    cap = cv2.VideoCapture(source_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = fps if fps and fps > 0 else 25
    raw = os.path.join(tmp_dir, "annotated_raw.mp4")
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    by_frame = {f: g for f, g in track_df.groupby("frame")}
    maxf = int(track_df["frame"].max())
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret or fi > maxf:
            break
        label = window_map.get(fi // window_frames, "Safe")
        color = hex_to_bgr(RISK_COLORS.get(label, PALETTE["emerald"]))  # BGR: frame is BGR here
        g = by_frame.get(fi)
        if g is not None:
            for _, row in g.iterrows():
                cx, cy, bw, bh = row["cx"], row["cy"], row["width"], row["height"]
                cv2.rectangle(frame, (int(cx - bw / 2), int(cy - bh / 2)),
                              (int(cx + bw / 2), int(cy + bh / 2)), color, 2)
        cv2.rectangle(frame, (0, 0), (300, 46), (18, 22, 30), -1)
        cv2.putText(frame, f"Window {fi // window_frames}  {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        writer.write(frame)
        fi += 1
    cap.release()
    writer.release()

    # mp4v rarely plays in browsers; re-encode to H.264 when ffmpeg is available.
    final = os.path.join(tmp_dir, "annotated_final.mp4")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", raw, "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", final],
                       check=True, capture_output=True, timeout=600)
        return final
    except Exception:
        return raw


def generate_report_card(name, cong, risk, counts, label_order, prof):
    fig = plt.figure(figsize=(8, 10), facecolor=PALETTE["bg"])
    gs = fig.add_gridspec(4, 2, height_ratios=[0.5, 1.2, 1.3, 1])
    at = fig.add_subplot(gs[0, :]); at.axis("off")
    at.text(0, 0.6, "Traffic Risk Report Card", fontsize=22, fontweight="bold", color="white")
    at.text(0, 0.15, name, fontsize=11, color=MUTED)

    def dg(ax, v, mx, t, c):
        frac = min(v / mx, 1.0) if mx else 0
        ax.pie([frac, 1 - frac], colors=[c, "#24303f"], startangle=90, counterclock=False,
               wedgeprops=dict(width=0.35, edgecolor=PALETTE["bg"]))
        ax.text(0, 0, f"{v:.0f}", ha="center", va="center", fontsize=20,
                color="white", fontweight="bold")
        ax.set_title(t, color="white", fontsize=12); ax.set_aspect("equal")

    dg(fig.add_subplot(gs[1, 0]), cong, max(cong * 1.5, 10), "Congestion", PALETTE["indigo"])
    dg(fig.add_subplot(gs[1, 1]), risk, 100, "Risk Score", PALETTE["rose"])

    axr = fig.add_subplot(gs[2, :], projection="polar")
    keys = list(FEATURE_DISPLAY_NAMES.keys())
    labels = list(FEATURE_DISPLAY_NAMES.values())
    vals = [prof.get(k, 0) for k in keys]
    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    vals += vals[:1]; ang += ang[:1]
    axr.set_facecolor(PALETTE["card"])
    axr.plot(ang, vals, color=PALETTE["rose"], linewidth=2)
    axr.fill(ang, vals, color=PALETTE["rose"], alpha=0.3)
    axr.set_xticks(ang[:-1]); axr.set_xticklabels(labels, color="white", fontsize=8)
    axr.set_yticklabels([]); axr.tick_params(colors="white")

    axb = fig.add_subplot(gs[3, :])
    axb.bar(label_order, [counts.get(l, 0) for l in label_order],
            color=[RISK_COLORS.get(l, "gray") for l in label_order])
    axb.set_facecolor(PALETTE["card"]); axb.tick_params(colors="white")
    for s in axb.spines.values():
        s.set_color("#3a4658")
    axb.set_title("Window classification", color="white", fontsize=12)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ===========================================================================
# Pages
# ===========================================================================
def page_command_center(df, metrics, label_order, label_col):
    avg_cong, avg_risk = dataset_averages(df, label_order, label_col)
    rf_acc = metrics["random_forest"]["test_accuracy"]
    n_videos = df["video_id"].nunique()
    n_windows = df[["video_id", "window_id"]].drop_duplicates().shape[0]

    per_video_high = (df[df[label_col] == label_order[-1]]
                      .groupby("video_id").size().sort_values(ascending=False).head(20).tolist())
    per_video_cong = (df.drop_duplicates(subset=["video_id", "window_id"])
                      .groupby("video_id")["congestion_score"].mean().head(20).tolist())

    theme.section_header("🛰️", "Command Center", "Live totals across the analysed corpus")

    vcounts = global_vehicle_counts(df, data_token(df))
    top_classes = sorted(vcounts.items(), key=lambda x: -x[1])[:4]

    cols = st.columns(4)
    spark_base = per_video_cong or [1, 2, 3, 2, 4]
    for i, (cls, cnt) in enumerate(top_classes):
        with cols[i]:
            theme.kpi_card(cls, cnt, icon=icon_for_class(cls), spark=spark_base,
                           accent=CYCLE[i % len(CYCLE)], key=f"vc{i}")
    for j in range(len(top_classes), 4):
        with cols[j]:
            theme.kpi_card("—", 0, icon="🚗", accent=PALETTE["indigo"], key=f"vcpad{j}")

    st.write("")
    c = st.columns(4)
    with c[0]:
        theme.kpi_card("Videos processed", n_videos, icon="🎞️",
                       spark=per_video_cong or [1, 2, 3], accent=PALETTE["indigo"], key="k1")
    with c[1]:
        # one row per vehicle per window - the 8,538 figure quoted in the
        # report. n_windows below is the distinct-window count (586) and
        # belongs in the header chip, not here.
        theme.kpi_card("Vehicle-windows", len(df), icon="🪟",
                       spark=per_video_high or [1, 2, 1, 3], accent=PALETTE["cyan"], key="k2")
    with c[2]:
        theme.kpi_card("Classifier agreement", rf_acc * 100, icon="🎯", decimals=1, suffix="%",
                       accent=PALETTE["emerald"], key="k3")
    with c[3]:
        theme.kpi_card("Avg risk score", avg_risk, icon="⚠️", suffix="/100",
                       spark=per_video_high or [10, 20, 15, 25], accent=PALETTE["rose"], key="k4")

    st.caption("‘Classifier agreement’ is how often Random Forest reproduces the K-means "
               "label it was trained on — a consistency check, not a measure of real-world "
               "crash risk.")

    # -------------------- Live uploads --------------------
    uploads_df = load_uploads_store()
    usum = uploads_summary(uploads_df, label_order, label_col)
    st.write("")
    theme.section_header("📥", "Live uploads",
                         f"Analysed clips accumulate here — the {n_videos}-video corpus stays frozen")

    if usum["videos"] == 0:
        st.caption("No uploads yet. Analyse a video on the **Analyze Video** tab "
                   "and its counts appear here.")
    else:
        ucols = st.columns(4)
        with ucols[0]:
            theme.kpi_card("Uploaded videos", usum["videos"], icon="📹",
                           accent=PALETTE["cyan"], key="up1")
        with ucols[1]:
            theme.kpi_card("Upload windows", usum["windows"], icon="🪟",
                           accent=PALETTE["violet"], key="up2")
        with ucols[2]:
            theme.kpi_card("Vehicles detected", usum["vehicles"], icon="🚗",
                           accent=PALETTE["indigo"], key="up3")
        with ucols[3]:
            theme.kpi_card(f"{label_order[-1]} windows", usum["high"], icon="⚠️",
                           accent=PALETTE["rose"], key="up4")

        if usum["by_class"]:
            st.caption("Upload vehicle mix: " +
                       ", ".join(f"{icon_for_class(k)} {k} × {v}"
                                 for k, v in sorted(usum["by_class"].items(), key=lambda x: -x[1])))
        if usum.get("raw_by_class"):
            st.caption("All detections by class (every frame-level box, nothing omitted): " +
                       ", ".join(f"{icon_for_class(k)} {k} × {v:,}"
                                 for k, v in sorted(usum["raw_by_class"].items(),
                                                    key=lambda x: -x[1])))
        if usum.get("detected_only"):
            st.caption(f"⚠ Detected but never a vehicle's majority class: "
                       f"{', '.join(usum['detected_only'])} — see Detection breakdown "
                       f"on the Analyze page.")

        if st.button("Reset uploaded data", icon=":material/delete:", key="reset_uploads"):
            n = reset_uploads_store()
            st.cache_data.clear()
            st.success(f"Cleared uploaded data ({n} file(s)). Corpus restored to {n_videos} videos.")
            st.rerun()

    st.write("")
    c1, c2 = st.columns([1, 1.15])
    with c1:
        counts = df[label_col].value_counts().reindex(label_order).fillna(0)
        fig = go.Figure(go.Pie(labels=label_order, values=counts.values, hole=0.55,
                               marker=dict(colors=[RISK_COLORS.get(l, MUTED) for l in label_order],
                                           line=dict(color=PALETTE["bg"], width=2))))
        fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", font={"color": FONT_COLOR},
                          title={"text": f"Risk distribution — {st.session_state.classifier}",
                                 "font": {"color": FONT_COLOR}})
        st.plotly_chart(fig, width="stretch")
    with c2:
        theme.section_header("✨", "What this system does")
        st.markdown(
            "- **Detects & tracks** every vehicle frame by frame\n"
            "- **Measures congestion** per 3-second window\n"
            "- **Extracts 5 risk features** per vehicle, in resolution-independent units\n"
            "- **Classifies risk** as Safe / Moderate / High Risk\n"
            "- **Upload any clip** on the Analyze page for live results\n\n"
            "Every figure here is computed from the pipeline — nothing is simulated.")


def page_model_lab(metrics, label_order):
    theme.section_header("🔬", "Model Lab", "How the classifier decides, and how well")

    if "interpretation_note" in metrics:
        st.info(metrics["interpretation_note"], icon=":material/info:")

    rf_imp = metrics["random_forest"]["feature_importances"]
    xgb_imp = metrics["xgboost"]["feature_importances"]
    fs = sorted(rf_imp.keys(), key=lambda f: (rf_imp[f] + xgb_imp[f]))
    labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in fs]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=[rf_imp[f] for f in fs], name="Random Forest",
                         orientation="h", marker_color=RF_COLOR))
    fig.add_trace(go.Bar(y=labels, x=[xgb_imp[f] for f in fs], name="XGBoost",
                         orientation="h", marker_color=XGB_COLOR))
    fig.update_layout(barmode="group", height=420, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor=PALETTE["card"], font={"color": FONT_COLOR},
                      title={"text": "Feature importance", "font": {"color": FONT_COLOR}},
                      legend=dict(orientation="h", y=1.12, font=dict(color=FONT_COLOR)))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    for col, key, title in [(c1, "random_forest", "Random Forest"), (c2, "xgboost", "XGBoost")]:
        cm = np.array(metrics[key]["confusion_matrix"])
        lab = metrics[key]["labels_order"]
        acc = metrics[key]["test_accuracy"]
        f = go.Figure(go.Heatmap(z=cm, x=lab, y=lab, colorscale="Purples", showscale=False,
                                 text=cm, texttemplate="%{text}", textfont=dict(size=14)))
        f.update_yaxes(autorange="reversed")
        f.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", font={"color": FONT_COLOR},
                        title={"text": f"{title} — {acc:.1%}", "font": {"color": FONT_COLOR}})
        col.plotly_chart(f, width="stretch")

    with st.expander("Cross-validation, clustering & split details"):
        rf = metrics["random_forest"]
        st.write(f"RF cross-validated accuracy: **{rf['cv_accuracy_mean']:.3f}** "
                 f"(± {rf['cv_accuracy_std']:.3f})")
        st.write(f"RF OOB score: **{rf['oob_score']:.3f}**")
        st.write(f"XGBoost stopped at boosting round: **{metrics['xgboost']['best_iteration']}**")
        ps = metrics["pseudolabeling"]
        st.write(f"Chosen k: **{ps['best_k']}** "
                 f"(silhouette by k: { {k: round(v,3) for k,v in ps['silhouette_scores_by_k'].items()} })")
        if "feature_units" in metrics:
            st.write(f"Feature units: `{metrics['feature_units']}`")
        st.json(metrics["split_sizes"])


def page_congestion():
    theme.section_header("🚥", "Congestion", "Output of the congestion_analysis.py stage")
    data = load_congestion()
    if data is None:
        st.warning("No congestion output found. Run `python src/congestion_analysis.py` "
                   "to generate it.", icon=":material/warning:")
        return
    windows, summary, thr = data

    c = st.columns(4)
    with c[0]:
        theme.kpi_card("Windows analysed", len(windows), icon="🪟",
                       accent=PALETTE["cyan"], key="cg1")
    with c[1]:
        theme.kpi_card("Videos", summary.shape[0], icon="🎞️",
                       accent=PALETTE["indigo"], key="cg2")
    with c[2]:
        theme.kpi_card("Clusters (k)", thr.get("k", 0), icon="🎯",
                       accent=PALETTE["violet"], key="cg3")
    with c[3]:
        theme.kpi_card("Window length", thr.get("window_seconds", 3), suffix="s",
                       icon="⏱️", accent=PALETTE["emerald"], key="cg4")

    st.caption(f"Thresholds derived from K-means centroids: "
               f"{[round(t, 2) for t in thr.get('thresholds', [])]} vehicles per window · "
               f"labels {thr.get('labels_low_to_high', [])}")

    c1, c2 = st.columns(2)
    with c1:
        vc = windows["congestion_label"].value_counts().reindex(
            thr.get("labels_low_to_high", [])).fillna(0)
        fig = go.Figure(go.Bar(x=vc.index, y=vc.values,
                               marker_color=CYCLE[:len(vc)]))
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor=PALETTE["card"], font={"color": FONT_COLOR},
                          title={"text": "Windows per congestion level",
                                 "font": {"color": FONT_COLOR}})
        fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure(go.Histogram(x=windows["raw_congestion"], nbinsx=40,
                                     marker_color=PALETTE["cyan"], opacity=0.75))
        for t in thr.get("thresholds", []):
            fig.add_vline(x=t, line_dash="dash", line_color=PALETTE["rose"])
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor=PALETTE["card"], font={"color": FONT_COLOR},
                          xaxis_title="Mean vehicles per window", yaxis_title="Windows",
                          title={"text": "Congestion distribution (dashed = threshold)",
                                 "font": {"color": FONT_COLOR}})
        fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig, width="stretch")

    st.dataframe(summary.sort_values("mean_congestion", ascending=False),
                 width="stretch", height=420)


def page_video_explorer(df, meta, cong_thr, ranges, label_order, label_col, window_seconds):
    theme.section_header("🎥", "Video Explorer", "Drill into any clip")

    uploads_df = load_uploads_store()
    corpus_ids = sorted(df["video_id"].unique().tolist())
    upload_ids = sorted(uploads_df["video_id"].unique().tolist()) if not uploads_df.empty else []

    source = st.segmented_control("Source", ["Corpus", "Uploads"], default="Corpus",
                                  key="ve_source")
    if source == "Uploads":
        if not upload_ids:
            st.info("No uploads yet — analyse a clip on the Analyze Video tab.")
            return
        pool, vids = uploads_df, upload_ids
    else:
        pool, vids = df, corpus_ids

    sel = st.selectbox("Select a video", vids, key="ve_video")

    col = label_col if label_col in pool.columns else "risk_label"
    clip = pool[pool["video_id"] == sel].drop_duplicates(subset=["window_id"])
    if clip.empty:
        st.warning("No windows for this video.")
        return

    nwin = clip["window_id"].nunique()
    # a window counts once however many of its vehicles are High Risk -
    # counting rows here made the numerator exceed nwin (e.g. 47 of 24)
    nhigh = clip.loc[clip[col] == label_order[-1], "window_id"].nunique()
    fps, _ = pipeline_core.resolve_fps(sel, meta)
    frame_width, _ = pipeline_core.resolve_frame_width(sel, meta, load_tracking_csv(sel))
    _, avg_risk = dataset_averages(df, label_order, label_col)
    vrisk = compute_risk_score_index(clip[col], label_order)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(gauge_chart(clip["congestion_score"].mean(),
                                    max(ranges["congestion_score"][1], 1),
                                    "Avg. congestion", cong_thr), width="stretch")
    with c2:
        st.plotly_chart(gauge_chart(vrisk, 100, "Risk score", 66), width="stretch")
    with c3:
        st.write("")
        theme.kpi_card(f"{label_order[-1]} windows", nhigh, icon="⚠️", suffix=f"/{nwin}",
                       accent=PALETTE["rose"], key="ve1")
        st.caption(relative_framing(vrisk, avg_risk))
        st.caption(f"{fps:.0f} fps · {frame_width:.0f}px wide")

    st.plotly_chart(chart_risk_congestion(clip, cong_thr, label_order, col, window_seconds),
                    width="stretch")

    track = load_tracking_csv(sel)
    cc1, cc2 = st.columns(2)
    with cc1:
        if track is not None and not track.empty:
            st.plotly_chart(chart_trajectory_heatmap(track), width="stretch")
    with cc2:
        sp = compute_all_speeds(sel, fps, frame_width, window_seconds)
        if len(sp) > 0:
            st.plotly_chart(chart_speed_distribution(sp), width="stretch")

    if track is not None and not track.empty:
        st.plotly_chart(chart_vehicle_mix(track), width="stretch")

    ranks = clip[col].map({l: i for i, l in enumerate(label_order)})
    if ranks.notna().any():
        worst = clip.loc[ranks.idxmax()]
        st.plotly_chart(radar_chart(normalize_feature_dict(worst.to_dict(), ranges),
                                    "Riskiest window profile"), width="stretch")


def page_data_explorer(df, label_order, label_col):
    theme.section_header("🗂️", "Data Explorer", "Filter and export the raw predictions")
    c1, c2, c3 = st.columns(3)
    with c1:
        vf = st.multiselect("Video", sorted(df["video_id"].unique().tolist()))
    with c2:
        rf_ = st.multiselect("Risk label", label_order, default=label_order)
    with c3:
        mx = int(df["congestion_score"].max())
        cr = st.slider("Congestion score range", 0, mx, (0, mx))

    fdf = df.copy()
    if vf:
        fdf = fdf[fdf["video_id"].isin(vf)]
    if rf_:
        fdf = fdf[fdf[label_col].isin(rf_)]
    fdf = fdf[(fdf["congestion_score"] >= cr[0]) & (fdf["congestion_score"] <= cr[1])]

    st.caption(f"Showing {len(fdf):,} of {len(df):,} rows · risk column: `{label_col}`")
    st.dataframe(fdf, width="stretch", height=480)
    st.download_button("Download filtered data (CSV)",
                       fdf.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_risk_predictions.csv", mime="text/csv",
                       icon=":material/download:")


# ---------------------------------------------------------------------------
# Analyze page - results are held in session_state so every control below
# keeps working instead of wiping the analysis (see module docstring, item 1)
# ---------------------------------------------------------------------------
def run_analysis(uploaded, settings, thresholds, label_order):
    """Runs the full pipeline on an uploaded clip and returns a result dict."""
    model = load_yolo_model()
    rf, xgb, encoder = load_risk_classifiers()

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, uploaded.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    pb = st.progress(0.0, text=STATUS_MESSAGES[0])
    track, fps, frame_width, frame_height = run_tracking_on_video(
        tmp_path, model, settings["conf"], settings["iou"], settings["imgsz"],
        progress_callback=lambda p: pb.progress(
            p, text=STATUS_MESSAGES[min(int(p * len(STATUS_MESSAGES)),
                                        len(STATUS_MESSAGES) - 1)]))
    pb.empty()

    if track.empty:
        return {"error": "No vehicles detected in this video."}

    raw_track = track.copy()
    track = filter_short_tracks(track, settings.get("min_track_len", DETECTION["min_track_len"]))
    if track.empty:
        return {"error": "No tracks survived filtering — try a longer or steadier clip."}

    breakdown = detection_breakdown(raw_track, track)

    feat = pipeline_core.compute_features(
        track, fps, frame_width,
        thresholds["proximity_threshold"], thresholds["deceleration_threshold"],
        window_seconds=settings["window_seconds"])

    if feat.empty:
        return {"error": "Not enough motion to compute features."}

    feat["rf_prediction"] = rf.predict(feat[FEATURES])
    feat["xgb_prediction"] = encoder.inverse_transform(xgb.predict(feat[FEATURES]))

    # Run the SAME Stage-1 K-means labeller the corpus was labelled with, so
    # "K-means label" on this page means what it says. Assigning the RF output
    # here instead (as the previous version did) made the reference label and
    # the classifier trivially identical, hiding every disagreement.
    pseudo = load_pseudolabeler()
    if pseudo is not None:
        scaler, kmeans, cluster_to_label = pseudo
        scaled = scaler.transform(pipeline_core.clustering_frame(feat)[pipeline_core.CLUSTER_FEATURES])
        feat["risk_label"] = [cluster_to_label.get(int(c), "Safe") for c in kmeans.predict(scaled)]
    else:
        feat["risk_label"] = feat["rf_prediction"]

    return {
        "name": uploaded.name, "track": track, "feat": feat,
        "breakdown": breakdown,
        "fps": fps, "frame_width": frame_width, "frame_height": frame_height,
        "tmp_path": tmp_path, "tmp_dir": tmp_dir,
        "window_seconds": settings["window_seconds"],
        "window_frames": pipeline_core.frames_per_window(fps, settings["window_seconds"]),
        "settings": settings,
    }


def discard_analysis():
    """
    Drop the stored analysis and remove its temp directory.

    Each run copies the upload into a fresh tempfile.mkdtemp() and may render an
    annotated replay beside it. Without this, every re-analysis left the
    previous video and its re-encoded copy behind for the lifetime of the
    machine - easily hundreds of MB over a session.
    """
    old = st.session_state.get("analysis")
    if isinstance(old, dict) and old.get("tmp_dir"):
        shutil.rmtree(old["tmp_dir"], ignore_errors=True)
    st.session_state.analysis = None


def page_analyze(df, cong_thr, ranges, label_order, label_col):
    theme.section_header("🎬", "Analyze new video",
                         "Upload footage — congestion and risk computed live, end to end")

    thresholds = load_risk_thresholds()

    # ---- Controls. In a form so adjusting a slider does not trigger anything
    # expensive; only Run Analysis does.
    with st.form("analyze_form"):
        uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        with st.expander("Detection settings", expanded=False):
            c1, c2, c3 = st.columns(3)
            conf = c1.slider("Confidence threshold", 0.05, 0.90, DETECTION["conf"], 0.05,
                             help="Lower finds more small/distant vehicles but adds false positives.")
            iou = c2.slider("NMS IoU", 0.10, 0.90, DETECTION["iou"], 0.05,
                            help="Higher keeps more overlapping boxes in dense traffic.")
            imgsz = c3.select_slider("Inference size", [416, 512, 640, 960, 1280],
                                     value=DETECTION["imgsz"],
                                     help="Larger detects smaller objects; slower.")
            c4, c5 = st.columns(2)
            window_seconds = c4.slider("Window length (seconds)", 1.0, 10.0,
                                       float(thresholds.get("window_seconds",
                                                            pipeline_core.DEFAULT_WINDOW_SECONDS)),
                                       0.5,
                                       help="Congestion and risk are computed per window of "
                                            "this much real time.")
            # Exposed because it DISCARDS VEHICLES. A vehicle visible for fewer
            # than this many frames is thrown away as a probable false positive,
            # but a human counting the video still counts it. Set it to 1 to
            # keep everything the detector found.
            min_track_len = c5.slider("Minimum track length (frames)", 1, 15,
                                      DETECTION["min_track_len"], 1,
                                      help="Vehicles tracked for fewer frames than this are "
                                           "discarded as likely false positives. Lower it to "
                                           "keep briefly-visible vehicles — set 1 to discard "
                                           "nothing.")
            st.caption(f"Corpus defaults: conf={DETECTION['conf']}, iou={DETECTION['iou']}, "
                       f"imgsz={DETECTION['imgsz']}. Changing these makes results less "
                       f"comparable to the corpus.")
        submitted = st.form_submit_button("Run analysis", type="primary",
                                          icon=":material/play_arrow:")

    if submitted:
        if uploaded is None:
            st.warning("Choose a video file first.", icon=":material/warning:")
        else:
            discard_analysis()   # free the previous run's temp dir
            result = run_analysis(uploaded, {"conf": conf, "iou": iou, "imgsz": imgsz,
                                             "window_seconds": window_seconds,
                                             "min_track_len": min_track_len},
                                  thresholds, label_order)
            st.session_state.analysis = result
            if "error" not in result:
                try:
                    saved_id = save_upload_to_store(result["feat"], result["track"],
                                                    result["name"])
                    st.session_state.analysis["saved_id"] = saved_id
                    st.cache_data.clear()
                except Exception as e:
                    st.warning(f"Analysis complete, but saving to the uploads store failed: {e}")

    analysis = st.session_state.analysis
    if analysis is None:
        st.info("Upload a video and press **Run analysis** to begin.", icon=":material/upload:")
        return
    if "error" in analysis:
        st.warning(analysis["error"], icon=":material/warning:")
        if st.button("Clear", key="clear_err"):
            discard_analysis()
            st.rerun()
        return

    # ---- Everything below reads from session_state, so it survives reruns ----
    track = analysis["track"]
    feat = analysis["feat"]
    fps = analysis["fps"]
    window_seconds = analysis["window_seconds"]
    col = label_col if label_col in feat.columns else "risk_label"

    top = st.columns([3, 1])
    top[0].success(f"Showing results for **{analysis['name']}** · "
                   f"{analysis['frame_width']}×{analysis['frame_height']} · {fps:.0f} fps · "
                   f"{window_seconds:g}s windows · risk from **{st.session_state.classifier}**")
    if top[1].button("Clear results", icon=":material/close:", key="clear_analysis"):
        discard_analysis()
        st.rerun()

    # ---- Vehicle census first: the number most people check against the video ----
    bd = analysis.get("breakdown")
    if bd is not None:
        theme.section_header("🚗", "Vehicle census",
                             "Total vehicles found and how many of each class — "
                             "compare this against a manual count")
        tbl = bd["table"]
        total_vehicles = int(tbl["vehicles"].sum())

        cells = st.columns(len(tbl) + 1)
        with cells[0]:
            theme.kpi_card("Total vehicles", total_vehicles, icon="🚦",
                           accent=PALETTE["cyan"], key="census_total")
        for i, row in enumerate(tbl.itertuples(index=False), start=1):
            if i >= len(cells):
                break
            with cells[i]:
                theme.kpi_card(row._0 if hasattr(row, "_0") else row[0],
                               int(row[1]), icon=icon_for_class(row[0]),
                               accent=CYCLE[(i - 1) % len(CYCLE)], key=f"census{i}")

        s = analysis["settings"]
        st.caption(
            f"Settings used — confidence **{s['conf']}**, IoU **{s['iou']}**, "
            f"inference size **{s['imgsz']}**, minimum track length "
            f"**{s.get('min_track_len', DETECTION['min_track_len'])} frames**, "
            f"window **{s['window_seconds']:g}s**. "
            f"{bd['raw_tracks'] - bd['kept_tracks']} short track(s) were discarded by the "
            f"minimum-track-length filter — lower it to 1 to keep every vehicle the "
            f"detector found."
        )
        if total_vehicles < bd["raw_tracks"]:
            st.info(f"The detector produced **{bd['raw_tracks']}** tracks; "
                    f"**{total_vehicles}** survived filtering. If your manual count is "
                    f"higher than this, lower the confidence threshold and the minimum "
                    f"track length, then re-run.", icon=":material/info:")

    ws_sum = summarize_window_risk(feat, label_order, col)
    nwin = ws_sum["window_id"].nunique()
    nhigh = int((ws_sum[col] == label_order[-1]).sum())
    cong_avg = float(ws_sum["congestion_score"].mean())
    risk = compute_risk_score_index(ws_sum[col], label_order)
    avg_cong, avg_risk = dataset_averages(df, label_order, label_col)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(gauge_chart(cong_avg,
                                    max(ranges["congestion_score"][1], cong_avg * 1.3),
                                    "Congestion", cong_thr), width="stretch")
        st.caption(relative_framing(cong_avg, avg_cong))
    with c2:
        st.plotly_chart(gauge_chart(risk, 100, "Risk score", 66), width="stretch")
        st.caption(relative_framing(risk, avg_risk))
    with c3:
        st.write("")
        theme.kpi_card("Windows analysed", nwin, icon="🪟", accent=PALETTE["cyan"], key="la1")
        theme.kpi_card(f"{label_order[-1]} windows", nhigh, icon="⚠️",
                       accent=PALETTE["rose"], key="la2")

    st.plotly_chart(chart_risk_congestion(ws_sum, cong_thr, label_order, col, window_seconds),
                    width="stretch")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.plotly_chart(chart_trajectory_heatmap(track), width="stretch")
    with cc2:
        st.plotly_chart(chart_vehicle_mix(track), width="stretch")

    # ---- Full detection accounting: every class, nothing dropped silently ----
    bd = analysis.get("breakdown")
    if bd:
        theme.section_header("🔎", "Detection breakdown",
                             "Every class the detector produced — including ones no vehicle "
                             "was finally labelled as")
        b1, b2 = st.columns([2, 1])
        with b1:
            st.dataframe(bd["table"], width="stretch", hide_index=True)
        with b2:
            st.metric("Vehicles kept", bd["kept_tracks"],
                      delta=f"{bd['kept_tracks'] - bd['raw_tracks']} short tracks removed"
                            if bd["raw_tracks"] != bd["kept_tracks"] else None,
                      delta_color="off")
            st.metric("Detections kept", f"{bd['kept_detections']:,}",
                      delta=f"of {bd['raw_detections']:,} raw", delta_color="off")
        st.caption(
            "**vehicles** counts distinct tracked vehicles by their majority class — one "
            "physical vehicle, one class. **detections** counts every frame-level box. "
            "A class can have detections but zero vehicles when it only ever appeared on "
            "frames of a vehicle that was something else for most of its life."
        )
        if bd["classes_without_vehicles"]:
            st.warning(
                "Detected but not the majority class of any vehicle: "
                + ", ".join(f"**{c}** ({int(bd['table'].loc[bd['table']['class']==c, 'detections (raw)'].iloc[0])} detections)"
                            for c in bd["classes_without_vehicles"])
                + ". On footage far from the training distribution these are usually "
                  "class flicker on a correctly-located vehicle rather than a missed one — "
                  "check the annotated replay to see which.",
                icon=":material/info:")

    ranks = ws_sum[col].map({l: i for i, l in enumerate(label_order)})
    worst = ws_sum.loc[ranks.idxmax()]
    worst_profile = normalize_feature_dict(worst.to_dict(), ranges)
    st.plotly_chart(radar_chart(worst_profile, "Riskiest window profile"), width="stretch")

    # ---- Incidents ----
    if nhigh > 0:
        theme.section_header("🚨", "Incident timeline")
        inc = (ws_sum[ws_sum[col] == label_order[-1]]
               .sort_values("congestion_score", ascending=False).head(5))
        wf = analysis["window_frames"]
        for _, row in inc.iterrows():
            midf = int(row["window_id"] * wf + wf // 2)
            # Merge on BOTH window_id and track_id. Merging on track_id alone
            # duplicated every row for any vehicle that appears in more than one
            # window, so a single frame could draw the same box many times with
            # labels borrowed from unrelated windows.
            frame_tracks = track[track["frame"] == midf].copy()
            frame_tracks["window_id"] = midf // wf
            fd = frame_tracks.merge(feat[["window_id", "track_id", col]],
                                    on=["window_id", "track_id"], how="left")
            fd = fd.rename(columns={col: "risk_label"})
            snap = snapshot_frame(analysis["tmp_path"], midf, fd)
            with st.container(border=True):
                ci, ct = st.columns([1, 2])
                if snap is not None:
                    ci.image(snap, width="stretch")
                ct.markdown(f"**Window {int(row['window_id'])}** · ~{midf / fps:.1f}s into the clip")
                ct.write(describe_incident(row, df))

    # ---- Annotated replay: works now, because the analysis is in session state ----
    theme.section_header("🎞️", "Annotated replay")
    if st.button("Generate annotated replay", icon=":material/movie:", key="gen_replay"):
        with st.spinner("Rendering annotated video…"):
            wmap = ws_sum.set_index("window_id")[col].to_dict()
            st.session_state.analysis["replay"] = build_annotated_video(
                analysis["tmp_path"], track, wmap, analysis["window_frames"], fps,
                analysis["tmp_dir"])
    if analysis.get("replay") and os.path.exists(analysis["replay"]):
        st.video(analysis["replay"])

    # ---- Report card ----
    theme.section_header("📄", "Report card")
    counts = ws_sum[col].value_counts().to_dict()
    buf = generate_report_card(analysis["name"], cong_avg, risk, counts, label_order,
                               worst_profile)
    st.download_button("Download report card (PNG)", buf,
                       file_name=f"report_{analysis['name']}.png", mime="image/png",
                       icon=":material/download:")

    with st.expander("Raw predictions for this video"):
        st.dataframe(feat, width="stretch", height=340)


# ===========================================================================

def page_method_comparison():
    """
    Every method choice in the pipeline, compared against its alternatives.

    Written by src/compare_importance.py, compare_classifiers.py,
    compare_clustering.py, compare_thresholds.py and compare_weighting.py.
    The page reads their CSV output rather than recomputing, so what is shown
    here is exactly what those scripts produced.
    """
    theme.section_header("⚖️", "Method Comparison",
                         "Each choice tested against its alternatives, "
                         "not asserted")

    charts = os.path.join("outputs", "charts", "master")

    def chart(fname, caption=None):
        p = os.path.join(charts, fname)
        if os.path.exists(p):
            st.image(p, width="stretch")
            if caption:
                st.caption(caption)
        else:
            st.info(f"{fname} not generated yet - run the matching "
                    f"src/compare_*.py script.", icon=":material/info:")

    tabs = st.tabs(["Classifiers", "Clustering", "Thresholds",
                    "Feature importance", "Ablation & weighting"])

    # ------------------------------------------------------ classifiers
    with tabs[0]:
        df = load_comparison("classifier_comparison.csv")
        if df is None:
            st.info("Run `python src/compare_classifiers.py`", icon=":material/info:")
        else:
            best = df.sort_values("mcc", ascending=False).iloc[0]
            rf = df[df.model == "Random Forest"]
            c = st.columns(3)
            with c[0]:
                theme.kpi_card("Models compared", len(df), icon="🧮",
                               accent=PALETTE["indigo"], key="mc1")
            with c[1]:
                theme.kpi_card("Best by MCC", best.mcc, icon="🏆", decimals=4,
                               accent=PALETTE["emerald"], key="mc2")
            with c[2]:
                if len(rf):
                    rank = int((df.mcc > rf.mcc.iloc[0]).sum()) + 1
                    theme.kpi_card("Random Forest rank", rank, icon="📉",
                                   accent=PALETTE["rose"], key="mc3")
            st.caption(f"Best model: **{best.model}**. Protocol: GroupKFold by "
                       f"video, 5 folds - no clip appears in train and test of "
                       f"the same fold.")
            chart("12_classifier_comparison.png")
            show = df[["model", "accuracy", "balanced_accuracy", "macro_f1",
                       "mcc", "high_risk_recall", "fit_seconds"]].copy()
            st.dataframe(show.style.format({c: "{:.4f}" for c in show.columns[1:]}),
                         width="stretch", hide_index=True)
            st.markdown(
                "**Why MCC is the headline.** With 61/35/4 class shares, accuracy "
                "has a floor of 0.61 before any learning happens - the dummy rows "
                "show it. MCC is the metric that cannot be inflated by imbalance "
                "(Chicco & Jurman, 2020, *BMC Genomics* 21:6).")

    # ------------------------------------------------------ clustering
    with tabs[1]:
        cl = load_comparison("clustering_comparison.csv")
        stab = load_comparison("clustering_stability.csv")
        if cl is None:
            st.info("Run `python src/compare_clustering.py`", icon=":material/info:")
        else:
            chart("13_clustering_comparison.png",
                  "Six algorithms across k = 2 to 8, three internal validity indices.")
            chart("14_clustering_stability.png",
                  "Bootstrap stability and cluster composition.")
            if stab is not None:
                st.markdown("**Bootstrap stability** (Adjusted Rand Index over 30 "
                            "resamples; Hennig, 2007, *CSDA* 52:258-271)")
                st.dataframe(stab.style.format({"ari_mean": "{:.4f}",
                                                "ari_sd": "{:.4f}",
                                                "ari_min": "{:.4f}"}),
                             width="stretch", hide_index=True)
                k2 = stab[stab.k == 2]
                if len(k2):
                    st.warning(
                        f"k=2 maximises the silhouette but is the **least stable** "
                        f"partition: ARI {k2.ari_mean.iloc[0]:.3f} "
                        f"± {k2.ari_sd.iloc[0]:.3f}, minimum "
                        f"{k2.ari_min.iloc[0]:.3f}. Under resampling it sometimes "
                        f"collapses entirely. This is the evidence for rejecting it.",
                        icon=":material/warning:")
            st.dataframe(cl, width="stretch", hide_index=True)

    # ------------------------------------------------------ thresholds
    with tabs[2]:
        th = load_comparison("threshold_methods.csv")
        if th is None:
            st.info("Run `python src/compare_thresholds.py`", icon=":material/info:")
        else:
            chart("15_threshold_methods.png",
                  "Six derivation methods on each distribution, with Hartigan's "
                  "dip test of unimodality.")
            for q in th.quantity.unique():
                sub = th[th.quantity == q]
                p = sub.dip_p.iloc[0]
                verdict = ("multimodal - a two-regime reading is supported"
                           if p < 0.05 else
                           "unimodal - no boundary exists, so a tail cut-off "
                           "is the honest description")
                st.markdown(f"**{q.capitalize()}** — dip p = {p:.3f} ({verdict})")
                st.dataframe(sub[["method", "threshold", "pct_below"]]
                             .style.format({"threshold": "{:.4f}",
                                            "pct_below": "{:.1f}"}),
                             width="stretch", hide_index=True)

    # ------------------------------------------------------ importance
    with tabs[3]:
        imp = load_comparison("importance_methods.csv")
        if imp is None:
            st.info("Run `python src/compare_importance.py`", icon=":material/info:")
        else:
            chart("11_importance_methods.png")
            st.dataframe(imp.style.format({c: "{:.4f}" for c in imp.columns[1:]}),
                         width="stretch", hide_index=True)
            st.markdown(
                "**Why this matters.** The impurity importance reported in the "
                "thesis is biased towards features with more potential split "
                "points (Strobl et al., 2007, *BMC Bioinformatics* 8:25). "
                "Permutation importance, computed on held-out data, is unbiased. "
                "Displacement ranks first and congestion last under every method - "
                "both headline claims survive - but proximity moves from 2nd "
                "under impurity to 4th under permutation.")

    # ------------------------------------------------------ ablation
    with tabs[4]:
        abl = load_comparison("feature_ablation.csv")
        w = load_comparison("weighting_schemes.csv")
        if abl is None:
            st.info("Run `python src/compare_weighting.py`", icon=":material/info:")
        else:
            chart("16_ablation_weighting.png")
            st.markdown("**Ablation** — drop one descriptor, re-run the whole "
                        "labelling chain. Lower ARI means the labelling changed more.")
            st.dataframe(abl.style.format({c: "{:.4f}" for c in abl.columns[1:]}),
                         width="stretch", hide_index=True)
            if w is not None:
                st.markdown("**Weighting schemes**")
                st.dataframe(w, width="stretch", hide_index=True)
                st.warning(
                    "Importance weighting is **circular**: the importance was "
                    "derived from labels that were themselves derived from these "
                    "features. It is shown for comparison only and must not be "
                    "used to regenerate labels. PCA weighting is free of that "
                    "feedback but produces a degenerate partition, so the "
                    "unweighted mean remains the defensible choice.",
                    icon=":material/warning:")


def main():
    df = load_predictions()
    metrics = load_metrics()
    meta = load_video_meta()
    label_order = metrics["pseudolabeling"]["label_order"]
    thresholds = load_risk_thresholds()
    window_seconds = float(thresholds.get("window_seconds", pipeline_core.DEFAULT_WINDOW_SECONDS))

    cong_thr, _ = derive_congestion_threshold(df, data_token(df))
    ranges = feature_ranges(df)

    theme.inject_css(st.session_state.accent)
    theme.render_header(df["video_id"].nunique(),
                        df[["video_id", "window_id"]].drop_duplicates().shape[0])

    page = option_menu(
        menu_title=None,
        options=["Command Center", "Model Lab", "Method Comparison",
                 "Congestion", "Video Explorer",
                 "Data Explorer", "Analyze Video"],
        icons=["grid-1x2", "cpu", "sliders", "stoplights", "camera-video",
               "table", "cloud-upload"],
        orientation="horizontal", default_index=0, key="main_nav",
        styles={
            "container": {"padding": "6px", "background-color": "rgba(255,255,255,0.03)",
                          "border-radius": "16px", "border": "1px solid rgba(255,255,255,0.07)",
                          "margin-bottom": "10px"},
            "icon": {"color": MUTED, "font-size": "15px"},
            "nav-link": {"font-size": "13.5px", "text-align": "center", "margin": "0 3px",
                         "border-radius": "10px", "color": "#C2CCDB", "padding": "10px 14px",
                         "--hover-color": "rgba(255,255,255,0.06)", "font-family": "Inter"},
            "nav-link-selected": {"background": f"linear-gradient(100deg,{PALETTE['indigo']},{PALETTE['cyan']})",
                                  "color": "#04121A", "font-weight": "600"},
        },
    )

    # Global classifier toggle - drives which prediction column every page reads.
    ctrl = st.columns([2, 3])
    with ctrl[0]:
        # No `default=` here on purpose. The value is seeded once via
        # session_state.setdefault at the top of the module; passing both a
        # default and a session_state-backed key is the documented Streamlit
        # anti-pattern ("mixing value parameter and session state") and emits a
        # warning on every rerun.
        st.segmented_control(
            "Risk shown by", list(CLASSIFIER_COLUMNS.keys()), key="classifier",
            help="Random Forest and XGBoost are trained to reproduce the K-means "
                 "label; switching between them shows where they disagree.")
    label_col = CLASSIFIER_COLUMNS.get(st.session_state.classifier, "rf_prediction")
    if label_col not in df.columns:
        label_col = "risk_label"

    if page == "Command Center":
        page_command_center(df, metrics, label_order, label_col)
    elif page == "Model Lab":
        page_model_lab(metrics, label_order)
    elif page == "Method Comparison":
        page_method_comparison()
    elif page == "Congestion":
        page_congestion()
    elif page == "Video Explorer":
        page_video_explorer(df, meta, cong_thr, ranges, label_order, label_col, window_seconds)
    elif page == "Data Explorer":
        page_data_explorer(df, label_order, label_col)
    elif page == "Analyze Video":
        page_analyze(df, cong_thr, ranges, label_order, label_col)


if __name__ == "__main__":
    main()

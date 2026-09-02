"""
risk_model.py

Stage 1 - K-means pseudo-labeling:
    The 5 features are transformed by pipeline_core.clustering_frame (displacement
    ratio flipped so every column points the same way, speed variance log1p'd),
    scaled, then clustered. k in {2,3} is chosen by silhouette score. The cluster
    with the highest mean scaled profile becomes "High Risk", the lowest "Safe".
    Fit ONLY on the training-split videos.

Stage 2 - Random Forest + XGBoost:
    Trained on the RAW (untransformed) 5 features to predict the Stage 1
    pseudo-label. Video-level 60/20/20 split prevents windows from the same video
    leaking across splits.

WHAT CHANGED AND WHY
--------------------
1. SPEED VARIANCE IS LOG-TRANSFORMED BEFORE CLUSTERING.
   It spans several orders of magnitude and is extremely heavy-tailed - the old
   corpus contained a value of 11,498,238, which was a tracker ID-switch
   teleporting a box across the frame, not a vehicle. StandardScaler does not
   tame a tail like that, so the scaled column dominated the composite used to
   rank clusters and "High Risk" collapsed into "largest speed variance", which
   before normalisation mostly meant "highest resolution footage".
   pipeline_core also now discards physically impossible steps outright.

2. THE ACCURACY NUMBER IS LABELLED FOR WHAT IT IS.
   RF and XGBoost are trained to predict labels that K-means derived from the
   same five features, so a high score means the classifiers recovered the
   clustering - it is NOT evidence that the clustering matches real-world risk.
   That caveat is written into the metrics JSON so it travels with the number
   instead of being remembered separately.

Output:
    outputs/csv/risk_predictions.csv     (every vehicle-window: features,
                                           pseudo-label, RF + XGB predictions)
    outputs/risk_model_metrics.json      (silhouette scores, CV scores,
                                           test accuracy, classification
                                           reports, feature importances)
    models/risk_model/*.joblib           (scaler, kmeans, rf, xgb, label
                                           encoder - reusable without retraining)
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import pipeline_core
from pipeline_core import PATHS, FEATURES, CLUSTER_FEATURES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONFIG = {
    "risk_features_csv": PATHS["risk_features"],
    "predictions_csv": PATHS["risk_predictions"],
    "metrics_json": PATHS["risk_metrics"],
    "models_dir": PATHS["risk_models_dir"],
    "random_state": 42,
    "train_frac": 0.6,
    "val_frac": 0.2,   # test_frac is the remainder (0.2)
    "k_candidates": [2, 3],
    # Set to an integer to skip silhouette-based selection, mirroring
    # congestion_analysis.py's FORCE_K. None lets the silhouette decide.
    #
    # Why this defaults to 3:
    #   Silhouette on the training split alone prefers k=2 (0.550 vs 0.305), but
    #   that k=2 solution just shaves off a 4% outlier tail and collapses
    #   everything else into one "Safe" bucket - no middle ground at all. Scored
    #   on the full dataset the preference reverses (k=3: 0.313, k=2: 0.280), so
    #   the k=2 win is an artefact of which videos landed in the training split,
    #   not a stable property of the data.
    #   k=3 also produces the Safe / Moderate / High Risk scheme the rest of the
    #   project is built around - the dashboard's signal-light colours, the
    #   report card and the incident feed all assume three levels - and gives a
    #   usable 61/34/4.5 split rather than 96/4.
    #   Both silhouette scores are still recorded in the metrics JSON, so the
    #   trade-off stays visible instead of being hidden by this choice.
    "force_k": 3,
}

LABEL_NAMES_BY_K = {
    2: ["Safe", "High Risk"],
    3: ["Safe", "Moderate", "High Risk"],
}


# ----------------------------------------------------------------------
# Video-level split - prevents windows from the same video leaking
# across train/val/test
# ----------------------------------------------------------------------
def split_by_video(df, train_frac, val_frac, random_state):
    # rng.permutation() returns a NEW array and is safe for any dtype;
    # RandomState.shuffle() on a pandas string array can silently duplicate.
    video_ids = np.array(df["video_id"].unique(), dtype=object)
    rng = np.random.RandomState(random_state)
    video_ids = rng.permutation(video_ids)

    n = len(video_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = set(video_ids[:n_train])
    val_ids = set(video_ids[n_train:n_train + n_val])
    test_ids = set(video_ids[n_train + n_val:])

    train_df = df[df["video_id"].isin(train_ids)].copy()
    val_df = df[df["video_id"].isin(val_ids)].copy()
    test_df = df[df["video_id"].isin(test_ids)].copy()

    print(f"video split -> train: {len(train_ids)} videos / {len(train_df)} rows, "
          f"val: {len(val_ids)} videos / {len(val_df)} rows, "
          f"test: {len(test_ids)} videos / {len(test_df)} rows")

    return train_df, val_df, test_df


# ----------------------------------------------------------------------
# Stage 1 - fit K-means pseudo-labeler on TRAIN ONLY
# ----------------------------------------------------------------------
def fit_pseudolabeler(train_df, k_candidates, random_state, force_k=None):
    cluster_df = pipeline_core.clustering_frame(train_df)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_df[CLUSTER_FEATURES])

    best_k, best_sil, sil_scores, models = None, -1, {}, {}

    for k in k_candidates:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(scaled)
        sample_size = 5000 if len(scaled) > 5000 else None
        sil = silhouette_score(scaled, km.labels_, sample_size=sample_size,
                               random_state=random_state)
        sil_scores[k] = sil
        models[k] = km
        print(f"  k={k} -> silhouette={sil:.3f}")
        if sil > best_sil:
            best_k, best_sil = k, sil

    if force_k is not None and force_k in models:
        print(f"FORCE_K override: using k={force_k} (silhouette {sil_scores[force_k]:.3f}) "
              f"instead of the silhouette winner k={best_k} ({best_sil:.3f})")
        print("  see CONFIG['force_k'] for why — the k=2 win is split-dependent and "
              "leaves no middle risk band")
        best_k = force_k

    best_model = models[best_k]
    print(f"using k={best_k} (silhouette={sil_scores[best_k]:.3f})")

    # Rank clusters by mean scaled profile. clustering_frame has already made
    # every column point the same way (larger == riskier), so an unweighted mean
    # is a meaningful ordering rather than a mix of opposing signals.
    centers = best_model.cluster_centers_
    composite = centers.mean(axis=1)
    order = np.argsort(composite)  # ascending: least risky -> most risky

    label_names = LABEL_NAMES_BY_K.get(best_k, [f"Risk_Level_{i}" for i in range(best_k)])
    cluster_to_label = {int(cluster_idx): label_names[rank]
                        for rank, cluster_idx in enumerate(order)}

    print(f"cluster -> label mapping: {cluster_to_label}")
    print("cluster centres (scaled, larger == riskier):")
    for idx in order:
        prof = ", ".join(f"{f}={centers[idx][i]:+.2f}"
                         for i, f in enumerate(CLUSTER_FEATURES))
        print(f"  {cluster_to_label[int(idx)]:10s} {prof}")

    return {
        "scaler": scaler,
        "kmeans": best_model,
        "cluster_features": CLUSTER_FEATURES,
        "cluster_to_label": cluster_to_label,
        "label_order": label_names,
        "best_k": best_k,
        "silhouette_scores": sil_scores,
        "cluster_centers": centers.tolist(),
    }


def assign_pseudolabels(df, pseudolabeler):
    d = df.copy()
    cluster_df = pipeline_core.clustering_frame(d)
    scaled = pseudolabeler["scaler"].transform(cluster_df[CLUSTER_FEATURES])

    cluster_idx = pseudolabeler["kmeans"].predict(scaled)
    d["cluster_idx"] = cluster_idx
    d["risk_label"] = [pseudolabeler["cluster_to_label"][c] for c in cluster_idx]
    d["risk_composite_score"] = scaled.mean(axis=1)  # for charts, not used by classifiers

    return d


# ----------------------------------------------------------------------
# Stage 2 - RF + XGBoost, trained on RAW features to predict risk_label
# ----------------------------------------------------------------------
def train_random_forest(X_train, y_train, random_state):
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features="sqrt",
        oob_score=True,
        random_state=random_state,
        n_jobs=-1,
    )

    min_class_count = pd.Series(y_train).value_counts().min()
    n_splits = min(5, max(2, int(min_class_count)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring="accuracy")
    print(f"RF {n_splits}-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    rf.fit(X_train, y_train)
    print(f"RF OOB score: {rf.oob_score_:.3f}")

    return rf, cv_scores


def train_xgboost(X_train, y_train_enc, X_val, y_val_enc, random_state, n_classes):
    # eval_metric must match the objective XGBoost picks from the label count.
    # With 2 classes it trains a BINARY model whose internal num_class is 1, and
    # a hard-coded "mlogloss" then fails with
    #   "MultiClassEvaluation: label must be in [0, num_class), num_class=1".
    # The old code got away with it only because k=3 happened to win the
    # silhouette comparison; the moment k=2 wins, training dies.
    eval_metric = "mlogloss" if n_classes > 2 else "logloss"

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.5,
        eval_metric=eval_metric,
        early_stopping_rounds=20,
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train_enc, eval_set=[(X_val, y_val_enc)], verbose=False)
    print(f"XGBoost ({eval_metric}) stopped at {xgb.best_iteration} boosting rounds (of 500 max)")
    return xgb


def evaluate(model, X_test, y_test, label_order, model_name, encoder=None):
    preds = model.predict(X_test)
    if encoder is not None:
        preds = encoder.inverse_transform(preds)
        y_true = encoder.inverse_transform(y_test)
    else:
        y_true = y_test

    acc = accuracy_score(y_true, preds)
    report = classification_report(y_true, preds, labels=label_order,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=label_order)

    print(f"\n{model_name} test accuracy: {acc:.3f}")
    print(classification_report(y_true, preds, labels=label_order, zero_division=0))

    return {
        "test_accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels_order": label_order,
    }


def main():
    df = pd.read_csv(CONFIG["risk_features_csv"])
    print(f"loaded {len(df)} vehicle-window rows from {CONFIG['risk_features_csv']}")

    train_df, val_df, test_df = split_by_video(
        df, CONFIG["train_frac"], CONFIG["val_frac"], CONFIG["random_state"]
    )

    # ---------------- Stage 1: pseudo-labels ----------------
    print("\n--- Stage 1: K-means pseudo-labeling (fit on train only) ---")
    pseudolabeler = fit_pseudolabeler(train_df, CONFIG["k_candidates"],
                                      CONFIG["random_state"], CONFIG["force_k"])

    train_df = assign_pseudolabels(train_df, pseudolabeler)
    val_df = assign_pseudolabels(val_df, pseudolabeler)
    test_df = assign_pseudolabels(test_df, pseudolabeler)

    label_order = pseudolabeler["label_order"]

    # Print the per-split label balance. A split that is missing a class
    # entirely breaks training in ways whose error messages point somewhere
    # else, so it is worth seeing directly rather than inferring it later.
    print("\npseudo-label distribution by split:")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = d["risk_label"].value_counts().reindex(label_order).fillna(0).astype(int)
        print(f"  {name:5s} " + "  ".join(f"{l}={counts[l]}" for l in label_order))
        missing = [l for l in label_order if counts[l] == 0]
        if missing:
            print(f"        ⚠ {name} split has NO examples of: {missing}")

    # ---------------- Stage 2: RF + XGBoost ----------------
    print("\n--- Stage 2: Random Forest ---")
    X_train, y_train = train_df[FEATURES], train_df["risk_label"]
    X_val, y_val = val_df[FEATURES], val_df["risk_label"]
    X_test, y_test = test_df[FEATURES], test_df["risk_label"]

    rf, rf_cv_scores = train_random_forest(X_train, y_train, CONFIG["random_state"])
    rf_metrics = evaluate(rf, X_test, y_test, label_order, "Random Forest")

    print("\n--- Stage 2: XGBoost ---")
    encoder = LabelEncoder().fit(label_order)  # fixed class order, not fit on data
    y_train_enc = encoder.transform(y_train)
    y_val_enc = encoder.transform(y_val)
    y_test_enc = encoder.transform(y_test)

    xgb = train_xgboost(X_train, y_train_enc, X_val, y_val_enc, CONFIG["random_state"],
                        n_classes=len(label_order))
    xgb_metrics = evaluate(xgb, X_test, y_test_enc, label_order, "XGBoost", encoder=encoder)

    # ---------------- Save predictions for ALL rows ----------------
    full_df = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ], ignore_index=True)

    full_df["rf_prediction"] = rf.predict(full_df[FEATURES])
    full_df["rf_confidence"] = rf.predict_proba(full_df[FEATURES]).max(axis=1)

    xgb_preds_enc = xgb.predict(full_df[FEATURES])
    full_df["xgb_prediction"] = encoder.inverse_transform(xgb_preds_enc)
    full_df["xgb_confidence"] = xgb.predict_proba(full_df[FEATURES]).max(axis=1)

    os.makedirs(os.path.dirname(CONFIG["predictions_csv"]), exist_ok=True)
    full_df.to_csv(CONFIG["predictions_csv"], index=False)
    print(f"\nwrote {len(full_df)} rows to {CONFIG['predictions_csv']}")
    print("\nlabel distribution:")
    print(full_df["risk_label"].value_counts())

    # ---------------- Save models ----------------
    os.makedirs(CONFIG["models_dir"], exist_ok=True)
    joblib.dump(pseudolabeler["scaler"], os.path.join(CONFIG["models_dir"], "scaler.joblib"))
    joblib.dump(pseudolabeler["kmeans"], os.path.join(CONFIG["models_dir"], "kmeans.joblib"))
    joblib.dump(rf, os.path.join(CONFIG["models_dir"], "random_forest.joblib"))
    joblib.dump(xgb, os.path.join(CONFIG["models_dir"], "xgboost.joblib"))
    joblib.dump(encoder, os.path.join(CONFIG["models_dir"], "label_encoder.joblib"))
    print(f"saved model artifacts to {CONFIG['models_dir']}")

    # ---------------- Save metrics ----------------
    metrics = {
        "interpretation_note": (
            "test_accuracy measures how well RF/XGBoost reproduce the K-means "
            "pseudo-labels, which were derived from these same 5 features. A high "
            "score means the classifiers recovered the clustering; it is NOT "
            "evidence that the clustering corresponds to real-world crash risk. "
            "Validating that would require labelled incident data."
        ),
        "feature_units": "frame_widths (resolution-invariant); see pipeline_core.py",
        "pseudolabeling": {
            "best_k": pseudolabeler["best_k"],
            "force_k": CONFIG["force_k"],
            "silhouette_scores_by_k": pseudolabeler["silhouette_scores"],
            "cluster_to_label": pseudolabeler["cluster_to_label"],
            "cluster_features": CLUSTER_FEATURES,
            "cluster_centers_scaled": pseudolabeler["cluster_centers"],
            "label_order": label_order,
        },
        "random_forest": {
            "cv_accuracy_mean": float(rf_cv_scores.mean()),
            "cv_accuracy_std": float(rf_cv_scores.std()),
            "oob_score": float(rf.oob_score_),
            **rf_metrics,
            "feature_importances": dict(zip(FEATURES, rf.feature_importances_.tolist())),
        },
        "xgboost": {
            "best_iteration": int(xgb.best_iteration),
            **xgb_metrics,
            "feature_importances": dict(zip(FEATURES, xgb.feature_importances_.tolist())),
        },
        "split_sizes": {
            "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df),
        },
        "label_distribution": full_df["risk_label"].value_counts().to_dict(),
    }

    os.makedirs(os.path.dirname(CONFIG["metrics_json"]), exist_ok=True)
    with open(CONFIG["metrics_json"], "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved metrics to {CONFIG['metrics_json']}")


if __name__ == "__main__":
    main()

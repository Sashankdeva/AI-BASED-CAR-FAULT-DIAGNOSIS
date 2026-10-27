"""Train and evaluate the fault-diagnosis models.

1. Primary classifier — Random Forest vs. Gradient Boosting compared with
   stratified 5-fold CV on the training split; the better one (macro-F1) is
   kept and its probabilities are calibrated (isotonic) so "confidence" means
   what it says.
2. Multi-fault detector — multi-label Random Forest trained on single- and
   two-fault data, so simultaneous faults can be reported.
3. Novelty detector — Local Outlier Factor vs. Isolation Forest; flags readings
   unlike anything in training (faults the classifier was never taught).

Everything is saved as one bundle used by the API.

Usage:  python -m car_fault.train        (then: python -m car_fault.evaluate)
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__, real_data
from .config import (
    DATASET_PATH,
    FAULT_NAMES,
    METRICS_PATH,
    MODEL_PATH,
    MULTI_DATASET_PATH,
    RANDOM_SEED,
    REPORT_DIR,
    SENSOR_NAMES,
)
from .data_generator import FAULT_CLASSES, UNSEEN_FAULTS, fault_indicators, generate, generate_multi
from .preprocessing import ENGINEERED, OBDFeatureEngineer

NOVELTY_QUANTILE = 0.01
NOVELTY_MAX_ROWS = 20_000  # ~1% of known-pattern training data is allowed to look "novel"


def candidates() -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline([
            ("features", OBDFeatureEngineer()),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                n_jobs=-1, random_state=RANDOM_SEED)),
        ]),
        "gradient_boosting": Pipeline([
            ("features", OBDFeatureEngineer()),
            ("model", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                class_weight="balanced", random_state=RANDOM_SEED)),
        ]),
    }


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        print(f"No dataset at {DATASET_PATH} — generating synthetic telemetry...")
        generate().to_csv(DATASET_PATH, index=False)
    if not MULTI_DATASET_PATH.exists():
        generate_multi().to_csv(MULTI_DATASET_PATH, index=False)
    return pd.read_csv(DATASET_PATH), pd.read_csv(MULTI_DATASET_PATH)


def split(df: pd.DataFrame, stratify: bool = True):
    """Deterministic 80/20 split shared by train.py and evaluate.py."""
    return train_test_split(df, test_size=0.2, random_state=RANDOM_SEED,
                            stratify=df["fault"] if stratify else None)


def calibration_stats(proba: np.ndarray, y_true: np.ndarray, classes: list[str], bins: int = 10) -> dict:
    """Expected calibration error, multiclass Brier score and reliability-diagram bins."""
    conf = proba.max(axis=1)
    correct = np.array(classes)[proba.argmax(axis=1)] == y_true
    onehot = (np.array(classes)[None, :] == y_true[:, None]).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    table, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
            table.append({"bin": [round(lo, 2), round(hi, 2)], "count": int(m.sum()),
                          "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return {"ece": float(ece), "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()), "bins": table}


def save_confusion_png(cm: np.ndarray, labels: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for r in range(len(labels)):
        for c in range(len(labels)):
            ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=7,
                    color="white" if norm[r, c] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Car fault diagnosis — test confusion matrix")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=130)
    plt.close(fig)


def train_classifier(train: pd.DataFrame, test: pd.DataFrame):
    X_train, y_train = train[SENSOR_NAMES], train["fault"]
    X_test, y_test = test[SENSOR_NAMES], test["fault"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = {}
    for name, pipe in candidates().items():
        t0 = time.time()
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
        cv_scores[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
        print(f"  {name:<18} CV macro-F1 = {scores.mean():.4f} ± {scores.std():.4f}  ({time.time() - t0:.1f}s)")
    best_name = max(cv_scores, key=lambda k: cv_scores[k]["mean"])
    print(f"Selected: {best_name} -> calibrating (isotonic, 5-fold)")

    # ensemble=False: one model fit on all training data + calibrators learned
    # from out-of-fold predictions, so the bundle holds a single forest.
    clf = CalibratedClassifierCV(candidates()[best_name], method="isotonic", cv=5, ensemble=False)
    clf.fit(X_train, y_train)
    raw_model = clf.calibrated_classifiers_[0].estimator

    classes = list(clf.classes_)
    proba = clf.predict_proba(X_test)
    pred = np.array(classes)[proba.argmax(axis=1)]
    yt = y_test.to_numpy()
    calibration = {
        "uncalibrated": calibration_stats(raw_model.predict_proba(X_test), yt, classes),
        "calibrated": calibration_stats(proba, yt, classes),
    }
    print(f"Calibration ECE {calibration['uncalibrated']['ece']:.4f} -> {calibration['calibrated']['ece']:.4f} | "
          f"Brier {calibration['uncalibrated']['brier']:.4f} -> {calibration['calibrated']['brier']:.4f}")

    labels = [f for f in FAULT_NAMES if f in set(y_test)]
    report = classification_report(yt, pred, labels=labels, output_dict=True, zero_division=0)
    print(classification_report(yt, pred, labels=labels, zero_division=0))
    metrics = {
        "accuracy": float(accuracy_score(yt, pred)),
        "macro_f1": float(f1_score(yt, pred, average="macro")),
        "weighted_f1": float(f1_score(yt, pred, average="weighted")),
        # Per single reading, before alert persistence is applied.
        "missed_fault_rate": float(((yt != "normal") & (pred == "normal")).sum() / (yt != "normal").sum()),
        "false_alarm_rate": float(((yt == "normal") & (pred != "normal")).sum() / (yt == "normal").sum()),
    }

    sub = X_test.sample(min(2000, len(X_test)), random_state=RANDOM_SEED)
    perm = permutation_importance(clf, sub, y_test.loc[sub.index], n_repeats=5,
                                  random_state=RANDOM_SEED, scoring="f1_macro")
    importance = dict(sorted(
        {f: round(float(v), 5) for f, v in zip(SENSOR_NAMES, perm.importances_mean)}.items(),
        key=lambda kv: -kv[1]))

    cm = confusion_matrix(yt, pred, labels=labels)
    return clf, best_name, cv_scores, metrics, report, labels, cm, calibration, importance


def train_multilabel(train: pd.DataFrame, multi_train: pd.DataFrame,
                     test: pd.DataFrame, multi_test: pd.DataFrame):
    data = pd.concat([train, multi_train], ignore_index=True)
    model = Pipeline([
        ("features", OBDFeatureEngineer()),
        ("model", RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                                         n_jobs=-1, random_state=RANDOM_SEED)),
    ])
    model.fit(data[SENSOR_NAMES], fault_indicators(data["fault"]))

    def scores(df):
        Y = fault_indicators(df["fault"])
        P = model.predict(df[SENSOR_NAMES])
        return {
            "exact_match": float((P == Y).all(axis=1).mean()),
            "micro_precision": float(precision_score(Y, P, average="micro", zero_division=0)),
            "micro_recall": float(recall_score(Y, P, average="micro", zero_division=0)),
            "per_fault_recall": {f: float(recall_score(Y[:, i], P[:, i], zero_division=0))
                                 for i, f in enumerate(FAULT_CLASSES)},
        }

    metrics = {"two_fault_test": scores(multi_test), "single_fault_test": scores(test)}
    m = metrics["two_fault_test"]
    print(f"Multi-fault detector: two-fault exact match {m['exact_match']:.3f}, "
          f"micro recall {m['micro_recall']:.3f}, micro precision {m['micro_precision']:.3f}")
    return model, metrics


def novelty_candidates() -> dict[str, Pipeline]:
    return {
        "local_outlier_factor": Pipeline([
            ("features", OBDFeatureEngineer()),
            ("scale", StandardScaler()),
            ("model", LocalOutlierFactor(n_neighbors=20, novelty=True)),
        ]),
        "isolation_forest": Pipeline([
            ("features", OBDFeatureEngineer()),
            ("scale", StandardScaler()),
            ("model", IsolationForest(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
    }


def train_novelty(train: pd.DataFrame, multi_train: pd.DataFrame, test: pd.DataFrame, multi_test: pd.DataFrame):
    """Fit both detectors, keep the one that catches the most unseen faults.

    Isolation Forest isolates on one random feature at a time, so a reading that
    is extreme on a single sensor (e.g. coolant at -40 °C) barely shortens its
    path — it misses most unseen faults here. LOF compares local density in the
    full feature space and catches them.
    """
    both = pd.concat([train, multi_train], ignore_index=True)
    # LOF stores its training rows; 20k keep the density estimate and the file small.
    both = both.sample(min(len(both), NOVELTY_MAX_ROWS), random_state=RANDOM_SEED)
    data, labels = both[SENSOR_NAMES], both["fault"]  # labels only steer healthy-expectation fitting
    unseen = generate(rows=1000, seed=RANDOM_SEED + 7, normal_share=0, faults=UNSEEN_FAULTS)

    results, fitted = {}, {}
    for name, model in novelty_candidates().items():
        model.fit(data, labels)
        threshold = float(np.quantile(model.score_samples(data), NOVELTY_QUANTILE))
        flagged = lambda X: float((model.score_samples(X[SENSOR_NAMES]) < threshold).mean())
        results[name] = {
            "threshold": threshold,
            "known_test_flagged_rate": flagged(test),
            "two_fault_test_flagged_rate": flagged(multi_test),
            "unseen_fault_detection_rate": {f: flagged(g) for f, g in unseen.groupby("fault")},
        }
        fitted[name] = (model, threshold)
        r = results[name]
        print(f"Novelty [{name}]: known flagged {r['known_test_flagged_rate']:.3f}, "
              f"unseen detected {r['unseen_fault_detection_rate']}")

    best = max(results, key=lambda k: np.mean(list(results[k]["unseen_fault_detection_rate"].values())))
    print(f"Selected novelty detector: {best}")
    model, threshold = fitted[best]
    return model, threshold, {"selected": best, "candidates": results}


def main() -> None:
    df, multi = load_datasets()
    train, test = split(df)
    multi_train, multi_test = split(multi)
    real_rows = 0
    train = real_data.blank_unlogged(train)  # many cars do not report all 15 sensors
    if real_data.available():
        # Real healthy driving (training trips only) + faults injected onto real readings.
        real_train_trips, _ = real_data.split_trips(real_data.load_carobd())
        real = real_data.semi_synthetic(real_train_trips)
        real_rows = len(real)
        train = pd.concat([train, real[SENSOR_NAMES + ["fault"]]], ignore_index=True)
        print(f"Added {real_rows:,} semi-synthetic rows from {real_train_trips['vehicle_id'].nunique()} "
              f"real carOBD trips")
    else:
        print("No real data in data/real/carOBD — training on synthetic data only "
              "(python -m car_fault.real_data --download to add it)")
    print(f"Train {len(train):,} | Test {len(test):,} (synthetic) | Multi-fault train {len(multi_train):,} "
          f"| {df['fault'].nunique()} classes")

    clf, best_name, cv_scores, metrics, report, labels, cm, calibration, importance = train_classifier(train, test)
    multilabel, multi_metrics = train_multilabel(train, multi_train, test, multi_test)
    novelty, novelty_threshold, novelty_metrics = train_novelty(train, multi_train, test, multi_test)

    healthy = train.loc[train["fault"] == "normal", SENSOR_NAMES]
    baseline = {c: {"mean": float(healthy[c].mean()), "std": float(healthy[c].std() or 1.0),
                    "p05": float(healthy[c].quantile(0.05)), "p95": float(healthy[c].quantile(0.95))}
                for c in SENSOR_NAMES}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "classifier": clf,
        "model_name": best_name,
        "classes": list(clf.classes_),
        "multilabel": multilabel,
        "multilabel_classes": FAULT_CLASSES,
        "novelty": novelty,
        "novelty_threshold": novelty_threshold,
        "sensors": SENSOR_NAMES,
        "engineered_features": ENGINEERED,
        "baseline": baseline,
        "feature_importance": importance,
        "metrics": metrics,
        "version": __version__,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_rows": len(train),
        "real_train_rows": real_rows,
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)

    METRICS_PATH.write_text(json.dumps({
        "model": best_name,
        "cv_macro_f1": cv_scores,
        "test": metrics,
        "per_class": {k: report[k] for k in labels},
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "calibration": calibration,
        "multi_fault": multi_metrics,
        "novelty": novelty_metrics,
        "feature_importance": importance,
    }, indent=2))
    save_confusion_png(cm, labels)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model -> {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()

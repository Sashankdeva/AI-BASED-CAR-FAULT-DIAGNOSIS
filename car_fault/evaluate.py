"""Evaluation report: EDA, ROC/PR, calibration, SHAP, robustness, alert persistence.

Run after training:  python -m car_fault.evaluate
Writes figures to reports/figures/ and numbers to reports/evaluation.json.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .alerting import AlertTracker
from .config import FAULTS, METRICS_PATH, RANDOM_SEED, REPORT_DIR, SENSOR_NAMES
from .data_generator import FAULT_CLASSES, generate
from .diagnose import DiagnosisEngine, get_engine
from .preprocessing import OBDFeatureEngineer
from .train import load_datasets, split

FIG_DIR = REPORT_DIR / "figures"
EVAL_PATH = REPORT_DIR / "evaluation.json"
PALETTE = ["#2f6fed", "#e06c1b", "#1f9d55", "#d62b3a", "#8a5cd1", "#d4a106", "#1aa3b8", "#7a8594"]


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130)
    plt.close(fig)
    print(f"  figure -> reports/figures/{name}")


# ------------------------------------------------------------------ EDA
def plot_eda(df: pd.DataFrame) -> None:
    order = list(FAULTS)
    counts = df["fault"].value_counts().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([FAULTS[f]["label"] for f in order][::-1], counts.values[::-1], color=PALETTE[0])
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    ax.set_title("Class distribution")
    ax.set_xlabel("Rows")
    _save(fig, "eda_class_distribution.png")

    feats = pd.DataFrame(OBDFeatureEngineer().fit(df[SENSOR_NAMES], df["fault"]).transform(df[SENSOR_NAMES]),
                         columns=OBDFeatureEngineer().get_feature_names_out())
    feats["fault"] = df["fault"].values
    keys = ["coolant_temp", "oil_vs_expected", "battery_voltage", "misfire_count",
            "total_fuel_trim", "fuel_pressure", "o2_voltage", "maf_vs_expected"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, k in zip(axes.flat, keys):
        data = [feats.loc[feats["fault"] == f, k] for f in order]
        ax.boxplot(data, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "#cfdcfb", "edgecolor": "#2f6fed"}, medianprops={"color": "#d62b3a"})
        ax.set_xticks(range(1, len(order) + 1), [f.replace("_", "\n", 1)[:18] for f in order], fontsize=6, rotation=60)
        ax.set_title(k, fontsize=10)
    fig.suptitle("Key signals by fault class — each fault has a distinct signature")
    _save(fig, "eda_sensor_boxplots.png")

    corr = feats.drop(columns="fault").corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature correlation (raw + engineered)")
    _save(fig, "eda_correlation.png")


# ------------------------------------------------------- ROC / PR / calibration
def plot_curves(engine: DiagnosisEngine, test: pd.DataFrame) -> dict:
    proba = engine.classifier.predict_proba(test[SENSOR_NAMES])
    y = test["fault"].to_numpy()
    out = {}
    fig_r, ax_r = plt.subplots(figsize=(7, 6))
    fig_p, ax_p = plt.subplots(figsize=(7, 6))
    for i, (c, color) in enumerate(zip(engine.classes, PALETTE * 2)):
        yt = (y == c).astype(int)
        fpr, tpr, _ = roc_curve(yt, proba[:, i])
        prec, rec, _ = precision_recall_curve(yt, proba[:, i])
        auc, ap = roc_auc_score(yt, proba[:, i]), average_precision_score(yt, proba[:, i])
        out[c] = {"roc_auc": float(auc), "pr_auc": float(ap)}
        ax_r.plot(fpr, tpr, color=color, lw=1.6, label=f"{c} ({auc:.3f})")
        ax_p.plot(rec, prec, color=color, lw=1.6, label=f"{c} ({ap:.3f})")
    ax_r.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax_r.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (one-vs-rest)")
    ax_p.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curves (one-vs-rest)")
    for ax in (ax_r, ax_p):
        ax.legend(fontsize=7, loc="lower right" if ax is ax_r else "lower left")
    _save(fig_r, "roc_curves.png")
    _save(fig_p, "pr_curves.png")
    return out


def plot_calibration(metrics: dict) -> None:
    cal = metrics["calibration"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="Perfect calibration")
    for key, color in (("uncalibrated", PALETTE[1]), ("calibrated", PALETTE[0])):
        b = cal[key]["bins"]
        ax.plot([x["confidence"] for x in b], [x["accuracy"] for x in b], "o-", color=color,
                label=f"{key} (ECE {cal[key]['ece']:.3f})")
    ax.set(xlabel="Predicted confidence", ylabel="Observed accuracy", title="Reliability diagram",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    _save(fig, "calibration_reliability.png")


# --------------------------------------------------------------------- SHAP
def plot_shap(engine: DiagnosisEngine, test: pd.DataFrame) -> dict:
    sample = test.sample(400, random_state=RANDOM_SEED)
    X = engine._features.transform(sample[SENSOR_NAMES])
    values = engine.explainer.shap_values(X, check_additivity=False)
    values = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    mean_abs = np.abs(values).mean(axis=0)  # (features, classes)
    order = np.argsort(mean_abs.sum(axis=1))[::-1][:14]
    names = [engine.feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 6))
    left = np.zeros(len(order))
    for k, (c, color) in enumerate(zip(engine.classes, PALETTE)):
        ax.barh(names[::-1], mean_abs[order, k][::-1], left=left[::-1], color=color, label=c)
        left += mean_abs[order, k]
    ax.set(xlabel="mean |SHAP value|", title="Global feature impact by fault class (SHAP)")
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, "shap_importance.png")
    return {str(engine.feature_names[i]): round(float(mean_abs[i].sum()), 4) for i in order}


# --------------------------------------------------------------- robustness
def _scores(engine: DiagnosisEngine, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = np.array(engine.classes)[engine.classifier.predict_proba(X).argmax(axis=1)]
    y = y.to_numpy()
    faulty, healthy = y != "normal", y == "normal"
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "missed_fault_rate": float((faulty & (pred == "normal")).sum() / max(faulty.sum(), 1)),
        "false_alarm_rate": float((healthy & (pred != "normal")).sum() / healthy.sum()) if healthy.any() else None,
    }


def robustness(engine: DiagnosisEngine, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Does performance hold when the data stops looking exactly like training data?"""
    rng = np.random.default_rng(RANDOM_SEED)
    X, y = test[SENSOR_NAMES].copy(), test["fault"]
    std = train.loc[train["fault"] == "normal", SENSOR_NAMES].std()

    def noisy(scale):
        return X + rng.normal(0, 1, X.shape) * (std.to_numpy() * scale)

    drift = X.copy()  # ageing sensors: small systematic offsets
    drift["battery_voltage"] += 0.3
    drift["coolant_temp"] += 3
    drift["fuel_pressure"] -= 15
    drift["o2_voltage"] += 0.05

    missing = X.mask(rng.random(X.shape) < 0.10)
    no_oil = X.assign(oil_pressure=np.nan)

    fresh = generate(rows=4800, seed=2024)
    mild = generate(rows=4800, seed=2025, intensity=(0.25, 0.5))

    scenarios = {
        "Clean test set": (X, y),
        "Unseen vehicles (new seed)": (fresh[SENSOR_NAMES], fresh["fault"]),
        "Sensor noise +10% of sd": (noisy(0.10), y),
        "Sensor noise +25% of sd": (noisy(0.25), y),
        "Sensor drift (ageing)": (drift, y),
        "10% readings missing": (missing, y),
        "Oil pressure not reported": (no_oil, y),
        "Mild faults only": (mild[SENSOR_NAMES], mild["fault"]),
    }
    table = {name: _scores(engine, Xs, ys) for name, (Xs, ys) in scenarios.items()}

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(table)
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [table[n]["accuracy"] for n in names], 0.4, label="Accuracy", color=PALETTE[0])
    ax.bar(xs + 0.2, [table[n]["macro_f1"] for n in names], 0.4, label="Macro F1", color=PALETTE[2])
    ax.set_xticks(xs, names, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.axhline(table["Clean test set"]["accuracy"], ls="--", color="#999", lw=1)
    ax.set_title("Robustness under distribution shift")
    ax.legend()
    _save(fig, "robustness.png")
    return table


# -------------------------------------------------------------- persistence
def persistence(engine: DiagnosisEngine, readings_per_vehicle: int = 20, vehicles: int = 150) -> dict:
    """Simulate vehicles streaming readings; compare raw vs. persistent alerting."""
    rng = np.random.default_rng(RANDOM_SEED)

    # Healthy fleet: how many false alerts reach drivers?
    healthy = generate(rows=vehicles * readings_per_vehicle, seed=3030, normal_share=1.0, faults=["normal"])
    healthy = healthy[healthy["fault"] == "normal"].head(vehicles * readings_per_vehicle)
    ids = [f"h{i // readings_per_vehicle}" for i in range(len(healthy))]
    res = engine.diagnose_many(healthy[SENSOR_NAMES], vehicle_ids=ids, tracker=AlertTracker(), explain=False)
    raw = np.array([r["raw_alert"] for r in res])
    conf = np.array([r["alert"] for r in res])
    healthy_stats = {
        "raw_false_alert_per_reading": float(raw.mean()),
        "confirmed_false_alert_per_reading": float(conf.mean()),
        "vehicles_with_false_alert_raw": float(raw.reshape(-1, readings_per_vehicle).any(axis=1).mean()),
        "vehicles_with_false_alert_confirmed": float(conf.reshape(-1, readings_per_vehicle).any(axis=1).mean()),
        "readings_per_vehicle": readings_per_vehicle,
    }

    # Faulty fleet: fixed fault intensity per vehicle, how fast is it confirmed?
    detection = {}
    n_fault_readings = 10
    for fault in FAULT_CLASSES:
        rows, vids = [], []
        for v in range(40):
            level = rng.uniform(0.25, 1.0)
            d = generate(rows=n_fault_readings, seed=int(rng.integers(1e9)), normal_share=0,
                         faults=[fault], intensity=(level, level))
            rows.append(d[SENSOR_NAMES])
            vids += [f"{fault}-{v}"] * len(d)
        frame = pd.concat(rows, ignore_index=True)
        res = engine.diagnose_many(frame, vehicle_ids=vids, tracker=AlertTracker(), explain=False)
        conf = np.array([r["alert"] for r in res]).reshape(40, -1)
        raw = np.array([r["raw_alert"] for r in res]).reshape(40, -1)
        first = [int(np.argmax(row)) + 1 for row in conf if row.any()]
        detection[fault] = {
            "detected_within_10": float(conf.any(axis=1).mean()),
            "mean_readings_to_alert": float(np.mean(first)) if first else None,
            "raw_reading_hit_rate": float(raw.mean()),
        }

    result = {"healthy": healthy_stats, "faulty": detection}
    h = healthy_stats
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(["Raw (every reading)", "Persistent (3 of 5)"],
                [h["raw_false_alert_per_reading"] * 100, h["confirmed_false_alert_per_reading"] * 100],
                color=[PALETTE[1], PALETTE[0]])
    axes[0].set(ylabel="% of healthy readings", title="False alerts shown to drivers")
    names = list(detection)
    axes[1].barh(names, [detection[f]["mean_readings_to_alert"] or 0 for f in names], color=PALETTE[2])
    axes[1].set(xlabel="Readings until alert (mean)", title="Detection delay per fault")
    _save(fig, "alert_persistence.png")
    return result


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    df, _ = load_datasets()
    train, test = split(df)

    print("EDA...")
    plot_eda(df)
    print("ROC / PR / calibration...")
    curves = plot_curves(engine, test)
    plot_calibration(metrics)
    print("SHAP...")
    shap_global = plot_shap(engine, test)
    print("Robustness...")
    robust = robustness(engine, train, test)
    print("Alert persistence...")
    persist = persistence(engine)

    report = {"per_class_auc": curves, "shap_global": shap_global, "robustness": robust,
              "persistence": persist}
    EVAL_PATH.write_text(json.dumps(report, indent=2))
    print(pd.DataFrame(robust).T.round(3).to_string())
    print(json.dumps(persist["healthy"], indent=2))
    print(f"Saved -> {EVAL_PATH}")


if __name__ == "__main__":
    main()

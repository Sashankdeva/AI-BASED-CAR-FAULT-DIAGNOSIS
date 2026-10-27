"""Real OBD-II data (carOBD): loading, semi-synthetic training data, evaluation.

Source: https://github.com/eron93br/carOBD — 2014 Toyota Etios (1.5 L), 1 Hz
logs of everyday driving. Please cite the authors' paper ("Detecting Anomalies
in the Engine Coolant Sensor Using One-Class Classifiers", IEEE) when using it.

The logs have no fault labels, so they are used two ways:
* **Healthy reference** — trips are split into train / held-out. Held-out trips
  measure how often the model raises alarms on a real, healthy car.
* **Semi-synthetic faults** — fault signatures are injected onto real readings,
  so the model learns faults on top of *real* driving instead of only on top of
  the simulator. The 5 sensors the car does not log are simulated for training.

Usage:  python -m car_fault.real_data --download        (35 training/held-out trips, ~33 MB, once)
        python -m car_fault.real_data                   (evaluate on the 11 held-out trips)
        python -m car_fault.real_data --download-fresh  (57 never-seen trips, ~36 MB, once)
        python -m car_fault.real_data --fresh           (final test on the never-seen trips)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .alerting import AlertTracker
from .config import DATA_DIR, FAULTS, MODEL_PATH, RANDOM_SEED, REPORT_DIR, SENSOR_NAMES
from .data_generator import FAULT_CLASSES, _inject

REAL_DIR = DATA_DIR / "real" / "carOBD"
REAL_EVAL_PATH = REPORT_DIR / "real_data_evaluation.json"
# Final test set: trips never used for training, tuning or rule design.
FRESH_DIR = DATA_DIR / "real" / "carOBD_fresh"
FRESH_EVAL_PATH = REPORT_DIR / "real_data_fresh_test.json"

# carOBD column -> our sensor name. The other 5 sensors (MAF, O2, fuel pressure,
# misfire count, oil pressure) are not logged by this car.
COLUMN_MAP = {
    "ENGINE_RPM": "engine_rpm",
    "VEHICLE_SPEED": "vehicle_speed",
    "THROTTLE": "throttle_pos",
    "ENGINE_LOAD": "engine_load",
    "COOLANT_TEMPERATURE": "coolant_temp",
    "INTAKE_AIR_TEMP": "intake_air_temp",
    "SHORT_TERM_FUEL_TRIM_BANK_1": "short_fuel_trim",
    "LONG_TERM_FUEL_TRIM_BANK_1": "long_fuel_trim",
    "TIMING_ADVANCE": "timing_advance",
    "CONTROL_MODULE_VOLTAGE": "battery_voltage",
}
LOGGED = list(COLUMN_MAP.values())
MISSING = [s for s in SENSOR_NAMES if s not in LOGGED]
# Faults whose signature is fully visible in the logged sensors: for these, some
# training rows keep the unlogged sensors blank, exactly as the real car reports.
VISIBLE_WITHOUT_MISSING = {"normal", "engine_overheating", "battery_charging_fault",
                           "fuel_system_lean", "maf_sensor_fault", "o2_sensor_fault"}  # trims visible

RAW_URL = "https://raw.githubusercontent.com/eron93br/carOBD/master/obdiidata/{}.csv"
SUBSET = ([f"drive{i}" for i in range(1, 14)] + [f"long{i}" for i in range(1, 13)]
          + [f"idle{i}" for i in range(1, 11)])


FRESH_SUBSET = [f"live{i}" for i in range(1, 40)] + [f"ufpe{i}" for i in range(1, 19)]


def download(folder: Path = REAL_DIR, names: list[str] = SUBSET) -> None:
    """Fetch the evaluation subset. The data is not redistributed in this repo."""
    import urllib.request

    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = folder / f"{name}.csv"
        if not target.exists():
            urllib.request.urlretrieve(RAW_URL.format(name), target)
    print(f"{len(names)} files in {folder}")


def available(folder: Path = REAL_DIR) -> bool:
    return folder.exists() and any(folder.glob("*.csv"))


def load_carobd(folder: Path = REAL_DIR) -> pd.DataFrame:
    """Read every log, map columns, keep only engine-running rows."""
    frames = []
    for path in sorted(folder.glob("*.csv")):
        # Data rows end with a trailing comma; without index_col=False pandas
        # silently shifts every column by one.
        d = pd.read_csv(path, skipinitialspace=True, index_col=False)
        d.columns = [c.replace("()", "").strip() for c in d.columns]
        out = d[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
        out["timestamp"] = d.iloc[:, 0]  # ENGINE_RUN_TIME, seconds since start
        out["vehicle_id"] = path.stem    # one trip = one tracked sequence
        out["trip_type"] = path.stem.rstrip("0123456789")
        frames.append(out)
    if not frames:
        raise FileNotFoundError(f"No CSV files in {folder}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["engine_rpm"] >= 500].dropna(subset=LOGGED)
    for s in MISSING:
        df[s] = np.nan
    return df.reset_index(drop=True)


def split_trips(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out every 3rd trip of each type. Splitting by trip (not by row) keeps
    near-identical consecutive seconds of one drive out of both sides."""
    test_ids = []
    for _, ids in df.groupby("trip_type")["vehicle_id"]:
        trips = sorted(ids.unique(), key=lambda s: int(s.lstrip("abcdefghijklmnopqrstuvwxyz")))
        test_ids += trips[2::3]
    is_test = df["vehicle_id"].isin(test_ids)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def simulate_missing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Fill the 5 unlogged sensors with healthy values for the current RPM/load.
    These follow the simulator's physics — the only option without real data."""
    df = df.copy()
    n = len(df)
    rpm, load = df["engine_rpm"].to_numpy(), df["engine_load"].to_numpy()
    cold = df["coolant_temp"].to_numpy() < 70
    df["maf_rate"] = (0.016 * rpm * load / 100 * rng.normal(1.0, 0.08, n)).clip(0.5)
    df["oil_pressure"] = ((150 + 0.05 * rpm) * np.where(cold, 1.25, 1.0) + rng.normal(0, 25, n)).clip(0)
    df["fuel_pressure"] = rng.normal(380, 20, n)
    df["o2_voltage"] = rng.uniform(0.1, 0.9, n)
    df["misfire_count"] = rng.poisson(0.6, n).astype(float)
    return df


def blank_unlogged(df: pd.DataFrame, share: float = 0.3, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Blank the commonly-unlogged sensors on a share of rows whose fault stays
    visible without them, so the model learns to judge cars that do not report
    MAF / O2 / fuel pressure / misfires / oil pressure."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    rows = df["fault"].isin(VISIBLE_WITHOUT_MISSING).to_numpy() & (rng.random(len(df)) < share)
    df.loc[rows, MISSING] = np.nan
    return df


def semi_synthetic(real: pd.DataFrame, rows: int = 20000, normal_share: float = 0.4,
                   seed: int = RANDOM_SEED, blank_share: float = 0.5,
                   faults: list[str] | None = None) -> pd.DataFrame:
    """Real driving + simulated unlogged sensors + injected fault signatures."""
    rng = np.random.default_rng(seed)
    faults = faults or FAULT_CLASSES
    n_normal = int(rows * normal_share)
    per_fault = (rows - n_normal) // len(faults)
    parts = []
    for fault, n in [("normal", n_normal)] + [(f, per_fault) for f in faults]:
        if n == 0:
            continue
        base = real.iloc[rng.integers(0, len(real), n)][SENSOR_NAMES].reset_index(drop=True)
        part = simulate_missing(base, rng)
        if fault != "normal":
            part = _inject(part, fault, rng)
        if fault in VISIBLE_WITHOUT_MISSING:
            blank = rng.random(n) < blank_share
            part.loc[blank, MISSING] = np.nan
        part["fault"] = fault
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df["misfire_count"] = df["misfire_count"].clip(lower=0)
    df["oil_pressure"] = df["oil_pressure"].clip(lower=0)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ------------------------------------------------------------------ evaluation
def evaluate_engine(engine, test_trips: pd.DataFrame) -> dict:
    """Healthy false alarms + detection of faults injected onto held-out real trips."""
    res = engine.diagnose_many(test_trips[LOGGED], vehicle_ids=list(test_trips["vehicle_id"]),
                               tracker=AlertTracker(), explain=False,
                               timestamps=list(test_trips["timestamp"].astype(float)),
                               run_times=list(test_trips["timestamp"].astype(float)))  # = ENGINE_RUN_TIME
    raw = np.array([r["raw_alert"] for r in res])
    conf = np.array([r["alert"] for r in res])
    trip_alert = pd.Series(conf).groupby(test_trips["vehicle_id"].to_numpy()).any()
    preds = pd.Series([r["fault"] for r in res]).value_counts(normalize=True)
    healthy = {
        "readings": len(res),
        "trips": int(test_trips["vehicle_id"].nunique()),
        "raw_alert_rate": float(raw.mean()),
        "confirmed_alert_rate": float(conf.mean()),
        "trips_with_confirmed_alert": float(trip_alert.mean()),
        "novel_rate": float(np.mean([r["novelty"]["is_novel"] for r in res])),
        "predicted": {k: float(v) for k, v in preds.items()},
    }

    # Faults injected onto held-out real trips. "as logged" leaves the 5 unlogged
    # sensors blank (what this car can actually report); "all sensors" simulates them.
    detection = {}
    for mode, blank in (("as_logged", 1.0), ("all_sensors", 0.0)):
        faults_df = semi_synthetic(test_trips, rows=3500, normal_share=0, seed=RANDOM_SEED + 99,
                                   blank_share=blank)
        if blank:  # as the car logs it: unlogged sensors are always missing
            faults_df.loc[:, MISSING] = np.nan
        out = engine.diagnose_many(faults_df[SENSOR_NAMES], explain=False)
        pred = np.array([r["fault"] for r in out])
        flagged = np.array([r["raw_alert"] for r in out])
        detection[mode] = {
            f: {"recall": float((pred[m] == f).mean()), "flagged_as_any_fault": float(flagged[m].mean())}
            for f, m in ((f, (faults_df["fault"] == f).to_numpy()) for f in FAULT_CLASSES)
        }
    return {"healthy_heldout": healthy, "injected_fault_detection": detection}


def _engine_from(path: Path):
    from .diagnose import DiagnosisEngine
    return DiagnosisEngine.load(path)


def main(model_path: Path = MODEL_PATH, out_path: Path = REAL_EVAL_PATH, label: str = "current") -> dict:
    real = load_carobd()
    train_trips, test_trips = split_trips(real)
    print(f"Real carOBD: {len(real):,} running rows | {train_trips['vehicle_id'].nunique()} train trips, "
          f"{test_trips['vehicle_id'].nunique()} held-out trips ({len(test_trips):,} rows)")
    report = {
        "model": label,
        "source": "https://github.com/eron93br/carOBD (2014 Toyota Etios, 1 Hz, unlabelled normal driving)",
        "held_out_trips": sorted(test_trips["vehicle_id"].unique()),
        "missing_sensors": MISSING,
        **evaluate_engine(_engine_from(model_path), test_trips),
        "sensor_range_comparison": _ranges(real),
    }
    out_path.write_text(json.dumps(report, indent=2))
    _print(report)
    print(f"Saved -> {out_path}")
    return report


def _ranges(real: pd.DataFrame) -> dict:
    from .data_generator import generate
    synth = generate(rows=5000, seed=RANDOM_SEED + 3, normal_share=1.0, faults=["normal"])
    synth = synth[synth["fault"] == "normal"]
    return {s: {"synthetic": [float(synth[s].quantile(q)) for q in (.05, .5, .95)],
                "real": [float(real[s].quantile(q)) for q in (.05, .5, .95)]} for s in LOGGED}


def _print(report: dict) -> None:
    h = report["healthy_heldout"]
    print(f"[{report['model']}] held-out healthy trips: raw alerts {h['raw_alert_rate']:.1%}, "
          f"confirmed {h['confirmed_alert_rate']:.1%}, trips alerted {h['trips_with_confirmed_alert']:.0%}, "
          f"novel {h['novel_rate']:.1%}")
    for mode, d in report["injected_fault_detection"].items():
        print(f"  injected faults ({mode}): " + ", ".join(f"{f.split('_')[0]} {v['recall']:.0%}" for f, v in d.items()))


def plot(before: dict, after: dict) -> None:
    fig_dir = REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    gap = after["sensor_range_comparison"]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for ax, (s, g) in zip(axes.flat, gap.items()):
        for x, (key, color) in enumerate((("synthetic", "#2f6fed"), ("real", "#e06c1b"))):
            lo, mid, hi = g[key]
            ax.plot([x, x], [lo, hi], color=color, lw=6, alpha=0.35, solid_capstyle="butt")
            ax.plot(x, mid, "o", color=color, label=key)
        ax.set_xticks([0, 1], ["synthetic", "real"], fontsize=8)
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(s, fontsize=9)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Healthy driving: simulator vs. real Toyota Etios (5th–95th percentile, median)")
    fig.tight_layout()
    fig.savefig(fig_dir / "real_vs_synthetic_ranges.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    keys = ["raw_alert_rate", "confirmed_alert_rate", "trips_with_confirmed_alert", "novel_rate"]
    names = ["Raw alerts", "Confirmed alerts", "Trips alerted", "Flagged novel"]
    xs = np.arange(len(keys))
    for off, (rep, color, lab) in zip((-0.2, 0.2), ((before, "#e06c1b", "v1 synthetic only"),
                                                     (after, "#2f6fed", "v2 + real data"))):
        axes[0].bar(xs + off, [rep["healthy_heldout"][k] * 100 for k in keys], 0.4, color=color, label=lab)
    axes[0].set_xticks(xs, names)
    axes[0].set(ylabel="% of held-out healthy readings / trips", title="False alarms on real healthy trips")
    axes[0].legend()

    faults = FAULT_CLASSES
    ys = np.arange(len(faults))
    for off, (rep, color, lab) in zip((-0.2, 0.2), ((before, "#e06c1b", "v1"), (after, "#2f6fed", "v2"))):
        axes[1].barh(ys + off, [rep["injected_fault_detection"]["as_logged"][f]["recall"] * 100 for f in faults],
                     0.4, color=color, label=lab)
    axes[1].set_yticks(ys, faults, fontsize=8)
    axes[1].set(xlabel="% correctly diagnosed", title="Faults injected onto real trips (10 logged sensors)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "real_data_before_after.png", dpi=130)
    plt.close(fig)
    print("  figures -> reports/figures/real_vs_synthetic_ranges.png, real_data_before_after.png")


def fresh_test(model_path: Path = MODEL_PATH) -> dict:
    """One-shot final test on trips that played no part in training or tuning."""
    fresh = load_carobd(FRESH_DIR)
    print(f"Fresh carOBD trips: {fresh['vehicle_id'].nunique()} trips, {len(fresh):,} running rows "
          f"({len(fresh) / 3600:.1f} h)")
    report = {"model": "current", "note": "final test: never used for training, tuning or rule design",
              "trips": sorted(fresh["vehicle_id"].unique()), "hours": len(fresh) / 3600,
              **evaluate_engine(_engine_from(model_path), fresh)}
    FRESH_EVAL_PATH.write_text(json.dumps(report, indent=2))
    _print(report)
    print(f"Saved -> {FRESH_EVAL_PATH}")
    return report


if __name__ == "__main__":
    import sys

    if "--download" in sys.argv:
        download()
    if "--download-fresh" in sys.argv:
        download(FRESH_DIR, FRESH_SUBSET)
    if "--fresh" in sys.argv:
        fresh_test()
        sys.exit()
    after = main()
    before_path = REPORT_DIR / "real_data_evaluation_v1.json"
    if before_path.exists():
        plot(json.loads(before_path.read_text()), after)

"""Synthetic OBD-II telemetry generator.

Public OBD-II datasets rarely come with labelled faults, so we simulate
physically-plausible readings for a petrol car across four driving modes and
inject the signature of each fault at a random *intensity*. Low-intensity
faults deliberately overlap with healthy readings so the classifier has to
learn real decision boundaries instead of trivial thresholds.

Usage:  python -m car_fault.data_generator --rows 24000
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import DATASET_PATH, FAULT_NAMES, MULTI_DATASET_PATH, RANDOM_SEED, SENSOR_NAMES

DRIVING_MODES = {
    #            rpm mean, sd   speed range   load mean, sd  throttle mean, sd  timing
    "idle":     (800, 60,      (0, 0),       22, 4,         4, 1.5,            10),
    "city":     (1900, 350,    (15, 60),     38, 10,        18, 6,             24),
    "highway":  (2600, 300,    (80, 130),    45, 8,         25, 6,             30),
    "accel":    (4000, 700,    (30, 110),    78, 10,        65, 15,            16),
}
MODE_PROBS = [0.2, 0.35, 0.3, 0.15]


def _healthy(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample healthy readings across driving modes."""
    modes = rng.choice(list(DRIVING_MODES), size=n, p=MODE_PROBS)
    cols = {k: np.zeros(n) for k in SENSOR_NAMES}
    for mode, (rpm_m, rpm_s, spd, load_m, load_s, thr_m, thr_s, timing) in DRIVING_MODES.items():
        idx = modes == mode
        k = idx.sum()
        cols["engine_rpm"][idx] = rng.normal(rpm_m, rpm_s, k)
        cols["vehicle_speed"][idx] = rng.uniform(*spd, k)
        cols["engine_load"][idx] = rng.normal(load_m, load_s, k)
        cols["throttle_pos"][idx] = rng.normal(thr_m, thr_s, k)
        cols["timing_advance"][idx] = rng.normal(timing, 3, k)

    df = pd.DataFrame(cols)
    df["engine_rpm"] = df["engine_rpm"].clip(600, 6800)
    df["engine_load"] = df["engine_load"].clip(8, 100)
    df["throttle_pos"] = df["throttle_pos"].clip(0, 100)

    displacement = rng.uniform(0.8, 1.4, n)  # vehicle-to-vehicle variation
    df["maf_rate"] = 0.016 * df["engine_rpm"] * df["engine_load"] / 100 * displacement * rng.normal(1, 0.05, n)

    ambient = rng.uniform(5, 40, n)
    df["intake_air_temp"] = ambient + rng.normal(8, 3, n)
    df["coolant_temp"] = rng.normal(90, 3, n)
    warmup = rng.random(n) < 0.08  # cold-start readings are healthy too
    df.loc[warmup, "coolant_temp"] = rng.uniform(30, 80, warmup.sum())

    df["short_fuel_trim"] = rng.normal(0, 3, n)
    df["long_fuel_trim"] = rng.normal(1, 3, n) + np.where(warmup, 2, 0)
    df["o2_voltage"] = rng.uniform(0.1, 0.9, n)  # switching sensor
    df["battery_voltage"] = rng.normal(14.1, 0.25, n)
    df["fuel_pressure"] = rng.normal(380, 20, n)
    df["misfire_count"] = rng.poisson(0.6, n).astype(float)
    oil_viscosity = np.where(warmup, 1.25, 1.0)
    df["oil_pressure"] = (150 + 0.05 * df["engine_rpm"]) * oil_viscosity + rng.normal(0, 25, n)
    df["driving_mode"] = modes
    return df


def _inject(df: pd.DataFrame, fault: str, rng: np.random.Generator,
            intensity: tuple[float, float] = (0.25, 1.0)) -> pd.DataFrame:
    """Apply a fault signature with random intensity in the given range."""
    n = len(df)
    i = rng.uniform(*intensity, n)

    if fault == "engine_overheating":
        df["coolant_temp"] = 104 + 18 * i + rng.normal(0, 3, n)
        df["timing_advance"] -= 6 * i
        df["oil_pressure"] -= 35 * i
        df["intake_air_temp"] += 6 * i
    elif fault == "low_oil_pressure":
        df["oil_pressure"] *= 1 - 0.75 * i
    elif fault == "ignition_misfire":
        df["misfire_count"] = rng.poisson(4 + 55 * i).astype(float)
        df["engine_rpm"] += rng.normal(0, 120, n)
        df["short_fuel_trim"] += 5 * i
        df["long_fuel_trim"] += 3 * i
        lean_read = rng.random(n) < 0.6
        df.loc[lean_read, "o2_voltage"] = rng.uniform(0.05, 0.35, lean_read.sum())
    elif fault == "fuel_system_lean":
        df["fuel_pressure"] *= 1 - 0.4 * i
        df["short_fuel_trim"] += 9 * i
        df["long_fuel_trim"] += 12 * i
        df["o2_voltage"] = rng.uniform(0.05, 0.3, n)
        df["misfire_count"] += rng.poisson(2 * i)
    elif fault == "battery_charging_fault":
        under = rng.random(n) < 0.7
        df["battery_voltage"] = np.where(
            under,
            rng.normal(13.2 - 1.3 * i, 0.2, n),   # alternator not charging
            rng.normal(14.9 + 1.1 * i, 0.2, n),   # regulator overcharging
        )
    elif fault == "maf_sensor_fault":
        df["maf_rate"] *= 1 - 0.45 * i  # under-reporting air
        df["short_fuel_trim"] += 4 * i
        df["long_fuel_trim"] += 10 * i
    elif fault == "o2_sensor_fault":
        df["o2_voltage"] = rng.uniform(0.8, 1.0, n)  # stuck rich
        df["short_fuel_trim"] -= 11 * i
        df["long_fuel_trim"] -= 9 * i
    # --- faults the classifier is never trained on (novelty-detection tests) ---
    elif fault == "unseen_rich_regulator":
        df["fuel_pressure"] = rng.uniform(540, 660, n)  # pressure regulator stuck closed
        df["short_fuel_trim"] -= 10 * i
        df["long_fuel_trim"] -= 14 * i
    elif fault == "unseen_coolant_sensor_open":
        df["coolant_temp"] = rng.uniform(-40, -30, n)  # open-circuit sensor reads minimum
    return df


FAULT_CLASSES = [f for f in FAULT_NAMES if f != "normal"]
UNSEEN_FAULTS = ["unseen_rich_regulator", "unseen_coolant_sensor_open"]
# Fault pairs that can physically co-occur without contradicting each other's signature.
FAULT_PAIRS = [
    ("engine_overheating", "low_oil_pressure"),
    ("engine_overheating", "battery_charging_fault"),
    ("engine_overheating", "ignition_misfire"),
    ("low_oil_pressure", "battery_charging_fault"),
    ("low_oil_pressure", "ignition_misfire"),
    ("battery_charging_fault", "ignition_misfire"),
    ("battery_charging_fault", "fuel_system_lean"),
    ("battery_charging_fault", "o2_sensor_fault"),
    ("battery_charging_fault", "maf_sensor_fault"),
    ("ignition_misfire", "maf_sensor_fault"),
]


def _finish(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["misfire_count"] = df["misfire_count"].clip(lower=0)
    df["oil_pressure"] = df["oil_pressure"].clip(lower=0)
    df["maf_rate"] = df["maf_rate"].clip(lower=0.5)
    df["o2_voltage"] = df["o2_voltage"].clip(0, 1.275)
    df[SENSOR_NAMES] = df[SENSOR_NAMES].round(3)
    return df


def generate(rows: int = 24000, seed: int = RANDOM_SEED, normal_share: float = 0.4,
             intensity: tuple[float, float] = (0.25, 1.0), faults: list[str] | None = None) -> pd.DataFrame:
    """Single-fault dataset: column `fault` holds exactly one label per row."""
    rng = np.random.default_rng(seed)
    faults = faults or FAULT_CLASSES
    n_normal = int(rows * normal_share)
    per_fault = (rows - n_normal) // len(faults)

    parts = []
    if n_normal:
        normal = _healthy(n_normal, rng)
        normal["fault"] = "normal"
        parts.append(normal)
    for fault in faults:
        part = _inject(_healthy(per_fault, rng), fault, rng, intensity)
        part["fault"] = fault
        parts.append(part)
    return _finish(pd.concat(parts, ignore_index=True), seed)


def generate_multi(rows: int = 3000, seed: int = RANDOM_SEED + 1) -> pd.DataFrame:
    """Two simultaneous faults per row; `fault` holds both labels joined by '+'."""
    rng = np.random.default_rng(seed)
    per_pair = rows // len(FAULT_PAIRS)
    parts = []
    for a, b in FAULT_PAIRS:
        part = _inject(_inject(_healthy(per_pair, rng), a, rng, (0.4, 1.0)), b, rng, (0.4, 1.0))
        part["fault"] = f"{a}+{b}"
        parts.append(part)
    return _finish(pd.concat(parts, ignore_index=True), seed)


def fault_indicators(labels: pd.Series) -> np.ndarray:
    """'a+b' style labels -> (n, len(FAULT_CLASSES)) 0/1 matrix; 'normal' is all zeros."""
    sets = labels.str.split("+")
    return np.array([[int(f in s) for f in FAULT_CLASSES] for s in sets])


def sample_reading(fault: str = "normal", seed: int | None = None) -> dict:
    """One synthetic reading — used by the live-feed simulator."""
    rng = np.random.default_rng(seed)
    df = _healthy(1, rng)
    for f in fault.split("+"):
        if f != "normal":
            df = _inject(df, f, rng)
    row = df.iloc[0]
    out = {k: round(float(row[k]), 3) for k in SENSOR_NAMES}
    out["misfire_count"] = max(0.0, out["misfire_count"])
    out["oil_pressure"] = max(0.0, out["oil_pressure"])
    out["maf_rate"] = max(0.5, out["maf_rate"])
    out["o2_voltage"] = min(max(out["o2_voltage"], 0.0), 1.275)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()
    df = generate(args.rows, args.seed)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {DATASET_PATH}")
    print(df["fault"].value_counts().to_string())

    multi = generate_multi(seed=args.seed + 1)
    multi.to_csv(MULTI_DATASET_PATH, index=False)
    print(f"Wrote {len(multi):,} multi-fault rows to {MULTI_DATASET_PATH}")


if __name__ == "__main__":
    main()

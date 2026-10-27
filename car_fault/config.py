"""Central configuration: paths, sensor schema and fault catalog."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("CARFAULT_DATA_DIR", ROOT_DIR / "data"))
MODEL_DIR = Path(os.getenv("CARFAULT_MODEL_DIR", ROOT_DIR / "models"))
REPORT_DIR = Path(os.getenv("CARFAULT_REPORT_DIR", ROOT_DIR / "reports"))
FRONTEND_DIR = ROOT_DIR / "frontend"

DATASET_PATH = DATA_DIR / "obd_telemetry.csv"
MULTI_DATASET_PATH = DATA_DIR / "obd_multi_fault.csv"
MODEL_PATH = MODEL_DIR / "fault_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

RANDOM_SEED = 42

# Raw OBD-II / telematics signals accepted by the API.
# name -> (unit, description, plausible min, plausible max, typical healthy value)
SENSORS: dict[str, dict] = {
    "engine_rpm":      {"unit": "rpm",  "label": "Engine RPM",               "min": 0,    "max": 8000, "default": 2200},
    "vehicle_speed":   {"unit": "km/h", "label": "Vehicle speed",            "min": 0,    "max": 250,  "default": 60},
    "engine_load":     {"unit": "%",    "label": "Calculated engine load",   "min": 0,    "max": 100,  "default": 40},
    "throttle_pos":    {"unit": "%",    "label": "Throttle position",        "min": 0,    "max": 100,  "default": 22},
    "coolant_temp":    {"unit": "°C",   "label": "Coolant temperature",      "min": -40,  "max": 150,  "default": 90},
    "intake_air_temp": {"unit": "°C",   "label": "Intake air temperature",   "min": -40,  "max": 100,  "default": 30},
    "maf_rate":        {"unit": "g/s",  "label": "Mass air flow",            "min": 0,    "max": 300,  "default": 14},
    "short_fuel_trim": {"unit": "%",    "label": "Short-term fuel trim",     "min": -40,  "max": 40,   "default": 0},
    "long_fuel_trim":  {"unit": "%",    "label": "Long-term fuel trim",      "min": -40,  "max": 40,   "default": 1},
    "o2_voltage":      {"unit": "V",    "label": "O2 sensor voltage (B1S1)", "min": 0,    "max": 1.275, "default": 0.45},
    "battery_voltage": {"unit": "V",    "label": "Battery / system voltage", "min": 8,    "max": 18,   "default": 14.1},
    "fuel_pressure":   {"unit": "kPa",  "label": "Fuel rail pressure",       "min": 0,    "max": 700,  "default": 380},
    "timing_advance":  {"unit": "°",    "label": "Ignition timing advance",  "min": -20,  "max": 50,   "default": 14},
    "misfire_count":   {"unit": "/1k rev", "label": "Misfires per 1000 rev", "min": 0,    "max": 200,  "default": 1},
    "oil_pressure":    {"unit": "kPa",  "label": "Engine oil pressure",      "min": 0,    "max": 700,  "default": 300},
}
SENSOR_NAMES = list(SENSORS)

# Signals that describe *how the car is being driven*, not its health. They are
# model inputs (context) but are never reported as the reason for a fault.
OPERATING_CONTEXT = {"engine_rpm", "vehicle_speed", "engine_load", "throttle_pos", "maf_rate", "intake_air_temp"}

# Fault catalog: what the model predicts and how the app reacts to it.
# severity (0..1) drives the risk score; actions are shown to the driver.
FAULTS: dict[str, dict] = {
    "normal": {
        "label": "No Fault Detected",
        "system": "—",
        "severity": 0.0,
        "description": "All monitored signals are within healthy operating ranges.",
        "actions": ["Continue normal operation.", "Keep following the scheduled service interval."],
    },
    "engine_overheating": {
        "label": "Engine Overheating",
        "system": "Cooling",
        "severity": 0.95,
        "description": "Coolant temperature is abnormally high; the ECU is retarding timing to protect the engine.",
        "actions": [
            "Pull over safely and switch off the engine.",
            "Do not open the radiator cap while hot.",
            "Check coolant level, radiator fan, thermostat and water pump.",
        ],
    },
    "low_oil_pressure": {
        "label": "Low Oil Pressure",
        "system": "Lubrication",
        "severity": 1.0,
        "description": "Oil pressure is below the safe threshold for the current RPM — risk of severe engine damage.",
        "actions": [
            "Stop the engine immediately.",
            "Check the oil level and look for leaks.",
            "Have the oil pump and pressure sender inspected before driving again.",
        ],
    },
    "ignition_misfire": {
        "label": "Ignition Misfire",
        "system": "Ignition",
        "severity": 0.75,
        "description": "Elevated misfire count — unburnt fuel can damage the catalytic converter.",
        "actions": [
            "Avoid hard acceleration and heavy loads.",
            "Inspect spark plugs, ignition coils and plug wires.",
            "Read the cylinder-specific DTCs (P0301–P0308).",
        ],
    },
    "fuel_system_lean": {
        "label": "Fuel System Running Lean",
        "system": "Fuel",
        "severity": 0.65,
        "description": "Low fuel pressure with strongly positive fuel trims — the engine is starved of fuel.",
        "actions": [
            "Check the fuel pump, fuel filter and pressure regulator.",
            "Inspect for vacuum leaks around the intake.",
            "Schedule a service within the next few days.",
        ],
    },
    "battery_charging_fault": {
        "label": "Battery / Charging Fault",
        "system": "Electrical",
        "severity": 0.6,
        "description": "System voltage is outside the 13.5–14.8 V charging window while the engine is running.",
        "actions": [
            "Test the battery and alternator output.",
            "Check the alternator belt and battery terminals for corrosion.",
            "Avoid switching the engine off far from help — it may not restart.",
        ],
    },
    "maf_sensor_fault": {
        "label": "MAF Sensor Fault",
        "system": "Air intake",
        "severity": 0.5,
        "description": "Measured air flow is inconsistent with RPM and load; fuel metering is being corrected.",
        "actions": [
            "Clean the MAF sensor with dedicated MAF cleaner.",
            "Check the air filter and intake ducting for leaks.",
            "Replace the MAF sensor if readings remain implausible.",
        ],
    },
    "o2_sensor_fault": {
        "label": "O2 Sensor Fault",
        "system": "Emissions",
        "severity": 0.4,
        "description": "Upstream O2 sensor reads stuck-rich, forcing large negative fuel corrections.",
        "actions": [
            "Inspect O2 sensor wiring and connector.",
            "Replace the upstream O2 sensor if it does not switch.",
            "Expect higher fuel consumption until repaired.",
        ],
    },
}
FAULT_NAMES = list(FAULTS)

# Not a model class: returned when the novelty detector sees a pattern unlike
# anything in training (a fault the classifier was never taught).
UNKNOWN_FAULT = {
    "label": "Unrecognised Abnormal Pattern",
    "system": "Unknown",
    "severity": 0.6,
    "description": "Sensor readings do not match healthy driving or any known fault signature.",
    "actions": [
        "Have the vehicle scanned with a full diagnostic tool.",
        "Check the flagged sensors and their wiring for damage.",
        "Drive cautiously until the cause is identified.",
    ],
}

# Labels for engineered features (shown in SHAP explanations).
ENGINEERED_LABELS = {
    "total_fuel_trim": "Total fuel trim (short + long)",
    "maf_vs_expected": "Air flow vs. healthy expectation",
    "oil_vs_expected": "Oil pressure vs. healthy expectation",
    "coolant_minus_intake": "Coolant − intake temperature",
    "charging_deviation": "Deviation from typical charging voltage",
}

# Alert persistence: raise an alert only when the same fault is seen in
# PERSIST_MIN_HITS of the last PERSIST_WINDOW readings of a vehicle, unless it
# is safety-critical (severity >= IMMEDIATE_SEVERITY and confident).
PERSIST_WINDOW = 5
PERSIST_MIN_HITS = 3
IMMEDIATE_SEVERITY = 0.95
IMMEDIATE_CONFIDENCE = 0.8
# The window is also bounded in time: readings older than PERSIST_MAX_AGE_S are
# forgotten (3 hits spread over a week are not a pattern), and the confirming
# hits must span at least PERSIST_MIN_SPAN_S (3 hits in 0.3 s are one glitch).
PERSIST_MAX_AGE_S = 300.0
PERSIST_MIN_SPAN_S = 2.0
# Some faults must persist longer. An unrecognised pattern is weak evidence (a
# broken sensor stays broken; an unusual driving moment passes). A failing
# alternator stays low, while a healthy one dips briefly at idle under electrical
# load (seen on the real carOBD car) — and undercharging is not urgent.
# Fuel-trim faults (lean, MAF, O2) develop slowly; brief trim swings are normal.
PERSIST_MIN_SPAN_OVERRIDES_S = {"unknown_anomaly": 30.0, "battery_charging_fault": 60.0,
                                "fuel_system_lean": 60.0, "maf_sensor_fault": 60.0, "o2_sensor_fault": 60.0}
MIN_SENSORS_PRESENT = 8  # readings with fewer known sensors are rejected

# Enabling conditions, as in real OBD-II monitors: a fault is only judged once
# the engine has run long enough for that system to settle. Right after start the
# alternator has not begun charging, so a 12.2 V reading is normal, not a fault.
# Uses the optional `engine_run_time` input (standard PID 0x1F, seconds since start).
# Fuel trims are only judged once closed-loop fuel control has settled after warm-up.
ENABLING_RUN_TIME_S = {"battery_charging_fault": 60, "fuel_system_lean": 180,
                       "maf_sensor_fault": 180, "o2_sensor_fault": 180}

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # CSV uploads and request bodies

# Risk-level bands on the 0–100 risk score.
RISK_LEVELS = [(30, "LOW"), (55, "MEDIUM"), (75, "HIGH"), (101, "CRITICAL")]

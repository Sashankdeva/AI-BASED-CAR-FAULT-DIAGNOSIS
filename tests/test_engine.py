import numpy as np
import pytest

from car_fault.alerting import AlertTracker
from car_fault.config import FAULT_NAMES, SENSOR_NAMES
from car_fault.data_generator import generate, generate_multi, sample_reading
from car_fault.diagnose import get_engine, risk_level


@pytest.fixture(scope="module")
def engine():
    return get_engine()


def test_generator_shape():
    df = generate(rows=800, seed=1)
    assert set(SENSOR_NAMES) <= set(df.columns)
    assert set(df["fault"]) == set(FAULT_NAMES)
    assert not df[SENSOR_NAMES].isna().any().any()
    assert generate_multi(rows=100)["fault"].str.contains(r"\+").all()


def test_risk_levels():
    assert risk_level(0) == "LOW"
    assert risk_level(40) == "MEDIUM"
    assert risk_level(60) == "HIGH"
    assert risk_level(99) == "CRITICAL"


def test_result_contract(engine):
    r = engine.diagnose(sample_reading("normal", seed=3))
    for key in ("fault", "confidence", "risk_score", "risk_level", "alert", "raw_alert", "probabilities",
                "detected_faults", "novelty", "shap_contributions", "explanations", "recommended_actions"):
        assert key in r
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-3
    assert 0 <= r["risk_score"] <= 100


@pytest.mark.parametrize("fault", ["engine_overheating", "battery_charging_fault", "ignition_misfire"])
def test_detects_clear_faults(engine, fault):
    hits = sum(engine.diagnose(sample_reading(fault, seed=s), explain=False)["fault"] == fault for s in range(20))
    assert hits >= 17


def test_overheating_is_high_risk_and_explained(engine):
    reading = sample_reading("normal", seed=5) | {"coolant_temp": 122, "timing_advance": 5}
    r = engine.diagnose(reading)
    assert r["fault"] == "engine_overheating"
    assert r["alert"] and r["risk_level"] in {"HIGH", "CRITICAL"}
    assert r["explanations"][0]["sensor"] == "coolant_temp"
    assert r["shap_contributions"][0]["feature"] == "coolant_temp"
    assert r["shap_contributions"][0]["contribution"] > 0


def test_missing_sensors_are_imputed(engine):
    reading = sample_reading("normal", seed=1)
    reading.pop("oil_pressure")
    r = engine.diagnose(reading)
    assert any("oil_pressure" in w for w in r["warnings"])


def test_too_few_sensors_rejected(engine):
    with pytest.raises(ValueError):
        engine.diagnose({"engine_rpm": 900, "coolant_temp": 90})


def test_persistence_requires_repeats():
    tracker = AlertTracker(window=5, min_hits=3)
    base = {"fault": "maf_sensor_fault", "raw_alert": True, "confidence": 0.9}
    statuses = [tracker.update("car", dict(base), severity=0.5)["persistence"]["status"] for _ in range(3)]
    assert statuses == ["pending", "pending", "confirmed"]
    # A safety-critical fault on another vehicle alerts on the first reading.
    critical = tracker.update("car2", {"fault": "low_oil_pressure", "raw_alert": True, "confidence": 0.95}, severity=1.0)
    assert critical["alert"] and critical["persistence"]["immediate"]


def test_critical_fault_alerts_immediately_even_with_second_fault(engine):
    # Two faults split the classifier's probability; overheating must still alert at once.
    alerted = 0
    for s in range(10):
        r = engine.diagnose(sample_reading("engine_overheating+battery_charging_fault", seed=s),
                            tracker=AlertTracker(), explain=False)
        alerted += r["fault"] == "engine_overheating" and r["alert"]
    assert alerted >= 8


def test_charging_not_judged_right_after_start(engine):
    reading = sample_reading("normal", seed=2) | {"battery_voltage": 12.2}
    early = engine.diagnose_many([reading], run_times=[5.0], explain=False)[0]
    late = engine.diagnose_many([reading], run_times=[300.0], explain=False)[0]
    assert not early["raw_alert"] and any("paused" in w for w in early["warnings"])
    assert late["fault"] == "battery_charging_fault" and late["raw_alert"]


def test_missing_sensors_imputed_as_healthy(engine):
    # A car that does not log MAF, O2, fuel pressure, misfires or oil pressure
    # must not look faulty just because those sensors are absent.
    hidden = ["maf_rate", "o2_voltage", "fuel_pressure", "misfire_count", "oil_pressure"]
    healthy = generate(rows=800, seed=11, normal_share=1.0, faults=["normal"])
    healthy = healthy[healthy["fault"] == "normal"][SENSOR_NAMES].drop(columns=hidden)
    flagged = np.mean([r["raw_alert"] for r in engine.diagnose_many(healthy, explain=False)])
    assert flagged < 0.12  # v1 (no context-aware imputation) raised a fault on ~54%


def test_weak_evidence_needs_longer_persistence():
    tracker = AlertTracker()
    base = {"fault": "unknown_anomaly", "raw_alert": True, "confidence": 0.9}
    early = [tracker.update("v", dict(base), severity=0.6, at=t) for t in (0, 2, 4, 6)]
    assert not any(r["alert"] for r in early)  # 6 s < 30 s
    later = [tracker.update("v", dict(base), severity=0.6, at=t) for t in range(8, 34, 2)]
    assert later[-1]["alert"]


def test_multiple_faults_reported(engine):
    hits = 0
    for s in range(10):
        r = engine.diagnose(sample_reading("engine_overheating+battery_charging_fault", seed=s), explain=False)
        found = {d["fault"] for d in r["detected_faults"]}
        hits += {"engine_overheating", "battery_charging_fault"} <= found
    assert hits >= 8


@pytest.mark.parametrize("unseen", ["unseen_rich_regulator", "unseen_coolant_sensor_open"])
def test_unseen_fault_never_passes_silently(engine, unseen):
    # The label may be "unknown" or the closest known fault (a stuck-rich regulator
    # looks like an O2 fault: both drive fuel trims negative), but it must be flagged
    # as novel and must raise an alert — never reported as healthy.
    results = [engine.diagnose(sample_reading(unseen, seed=s), explain=False) for s in range(10)]
    assert sum(r["novelty"]["is_novel"] for r in results) >= 9
    assert sum(r["raw_alert"] for r in results) >= 9
    assert sum(r["fault"] == "unknown_anomaly" for r in results) >= 5


def test_healthy_not_flagged_novel(engine):
    novel = sum(engine.diagnose(sample_reading("normal", seed=s), explain=False)["novelty"]["is_novel"]
                for s in range(50))
    assert novel <= 3


def test_probabilities_are_calibrated(engine):
    assert type(engine.classifier).__name__ == "CalibratedClassifierCV"

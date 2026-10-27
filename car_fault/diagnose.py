"""Inference engine: ML Detection -> Fault Check -> Confidence Score -> Alert.

This module is framework-free so any host (FastAPI, Flask, Streamlit, a
mobile backend, a batch job) can import it directly:

    from car_fault.diagnose import DiagnosisEngine
    engine = DiagnosisEngine.load()
    result = engine.diagnose({"engine_rpm": 2100, ...})
"""

from __future__ import annotations

import copy
import time
import uuid
from functools import lru_cache

import joblib
import numpy as np

from .alerting import AlertTracker, untracked
from .config import (
    ENABLING_RUN_TIME_S,
    ENGINEERED_LABELS,
    FAULTS,
    MODEL_PATH,
    OPERATING_CONTEXT,
    RISK_LEVELS,
    SENSORS,
    SENSOR_NAMES,
    UNKNOWN_FAULT,
)
from .preprocessing import to_frame

ALERT_CONFIDENCE = 0.5     # minimum calibrated confidence before a fault counts as detected
MULTI_FAULT_THRESHOLD = 0.5
UNKNOWN_MAX_CONFIDENCE = 0.6  # novel reading + unsure classifier -> "unknown fault"
UNKNOWN_MIN_RISK = 60.0
SHAP_TREES = 60  # trees sampled for per-reading SHAP (see _tree_subset)


def fault_info(name: str) -> dict:
    return UNKNOWN_FAULT if name == "unknown_anomaly" else FAULTS[name]


def risk_level(score: float) -> str:
    for upper, name in RISK_LEVELS:
        if score < upper:
            return name
    return "CRITICAL"


def _tree_subset(forest, n: int):
    """A forest predicts the mean of its trees, so SHAP values of a random tree
    subset are an unbiased estimate of the full forest's SHAP values, at a
    fraction of the cost. Gradient-boosting models are returned unchanged."""
    if not hasattr(forest, "estimators_") or len(forest.estimators_) <= n:
        return forest
    sub = copy.copy(forest)
    idx = np.random.default_rng(0).choice(len(forest.estimators_), n, replace=False)
    sub.estimators_ = [forest.estimators_[i] for i in sorted(idx)]
    sub.n_estimators = n
    return sub


def _single_thread(estimator) -> None:
    """Thread start-up dominates latency when scoring one reading at a time."""
    model = estimator[-1] if hasattr(estimator, "steps") else estimator
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1


class DiagnosisEngine:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.classifier = bundle["classifier"]
        self.classes: list[str] = bundle["classes"]
        self.multilabel = bundle["multilabel"]
        self.multilabel_classes: list[str] = bundle["multilabel_classes"]
        self.novelty = bundle["novelty"]
        self.novelty_threshold: float = bundle["novelty_threshold"]
        self.baseline: dict = bundle["baseline"]
        self.severity = np.array([FAULTS[c]["severity"] for c in self.classes])
        self.ml_severity = np.array([FAULTS[c]["severity"] for c in self.multilabel_classes])

        for cc in self.classifier.calibrated_classifiers_:
            _single_thread(cc.estimator)
        _single_thread(self.multilabel)
        _single_thread(self.novelty)

        # The calibrated model wraps one fitted pipeline; SHAP explains its trees.
        raw = self.classifier.calibrated_classifiers_[0].estimator
        self._features, self._tree = raw[0], raw[-1]
        self.feature_names = list(self._features.get_feature_names_out())
        self._explainer = None

    @classmethod
    def load(cls, path=MODEL_PATH) -> "DiagnosisEngine":
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Train it first: python -m car_fault.train")
        return cls(joblib.load(path))

    @property
    def info(self) -> dict:
        b = self.bundle
        return {
            "model": b["model_name"],
            "calibration": "isotonic",
            "multi_fault_detector": "random_forest (multi-label)",
            "novelty_detector": type(self.novelty[-1]).__name__,
            "version": b["version"],
            "trained_at": b["trained_at"],
            "train_rows": b["train_rows"],
            "classes": self.classes,
            "metrics": b["metrics"],
            "feature_importance": b["feature_importance"],
        }

    # ---------------------------------------------------------------- explain
    @property
    def explainer(self):
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(_tree_subset(self._tree, SHAP_TREES))
        return self._explainer

    def _shap(self, X_feat: np.ndarray, class_idx: np.ndarray) -> list[list[dict]]:
        """Top SHAP contributions toward each row's predicted class."""
        try:
            values = self.explainer.shap_values(X_feat, check_additivity=False)
        except Exception:  # SHAP is an add-on; never fail a diagnosis because of it
            return [[] for _ in range(len(X_feat))]
        values = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)

        out = []
        for row, x, k in zip(values, X_feat, class_idx):
            contrib = row[:, k]
            order = np.argsort(-np.abs(contrib))[:6]
            items = []
            for j in order:
                name = str(self.feature_names[j])
                spec = SENSORS.get(name)
                items.append({
                    "feature": name,
                    "label": spec["label"] if spec else ENGINEERED_LABELS.get(name, name),
                    "value": round(float(x[j]), 3),
                    "unit": spec["unit"] if spec else "",
                    "contribution": round(float(contrib[j]), 4),
                    "engineered": spec is None,
                })
            out.append(items)
        return out

    def _abnormal_readings(self, reading: dict) -> list[dict]:
        """Diagnostic sensors that sit outside the healthy baseline (rule-based view)."""
        factors = []
        for name in SENSOR_NAMES:
            value = reading.get(name)
            if name in OPERATING_CONTEXT or value is None or np.isnan(value):
                continue
            base = self.baseline[name]
            if value > base["p95"]:
                direction = "above"
            elif value < base["p05"]:
                direction = "below"
            else:
                continue
            spec = SENSORS[name]
            factors.append({
                "sensor": name,
                "label": spec["label"],
                "value": round(float(value), 3),
                "unit": spec["unit"],
                "healthy_range": [round(base["p05"], 2), round(base["p95"], 2)],
                "direction": direction,
                "z_score": round(float((value - base["mean"]) / base["std"]), 2),
                "message": f"{spec['label']} {value:g} {spec['unit']} is {direction} the healthy range "
                           f"({base['p05']:.4g}–{base['p95']:.4g} {spec['unit']})",
            })
        factors.sort(key=lambda f: -abs(f["z_score"]))
        return factors[:5]

    @staticmethod
    def _warnings(reading: dict) -> list[str]:
        out = []
        for name, spec in SENSORS.items():
            v = reading.get(name)
            if v is None or np.isnan(v):
                out.append(f"{name} not reported — imputed from training data")
            elif v < spec["min"] or v > spec["max"]:
                out.append(f"{name}={v} is outside the plausible range [{spec['min']}, {spec['max']}] and was clipped")
        return out

    # --------------------------------------------------------------- diagnose
    def diagnose_many(self, readings, vehicle_ids: list[str | None] | None = None,
                      tracker: AlertTracker | None = None, explain: bool = True,
                      timestamps: list[float | None] | None = None,
                      default_vehicle_id: str | None = None,
                      run_times: list[float | None] | None = None) -> list[dict]:
        """Diagnose readings in order. With a tracker, alerts need to persist per vehicle.

        Readings without a vehicle id fall back to `default_vehicle_id`; if that is
        also None they are never pooled together — each is judged alone.
        """
        t0 = time.perf_counter()
        frame = to_frame(readings)
        proba = self.classifier.predict_proba(frame)
        ml_proba = np.column_stack([p[:, 1] for p in self.multilabel.predict_proba(frame)])
        novelty_score = self.novelty.score_samples(frame)
        X_feat = self._features.transform(frame)
        top_idx = proba.argmax(axis=1)
        shap_rows = self._shap(X_feat, top_idx) if explain else [[] for _ in range(len(frame))]
        latency = (time.perf_counter() - t0) * 1000 / max(len(frame), 1)

        results = []
        rows = frame.to_dict("records")
        for i, (row, p, mp) in enumerate(zip(rows, proba, ml_proba)):
            order = np.argsort(p)[::-1]
            top = self.classes[order[0]]
            confidence = float(p[order[0]])
            class_risk = float(np.dot(p, self.severity) * 100)

            detected = [
                {"fault": f, "label": FAULTS[f]["label"], "probability": round(float(q), 4)}
                for f, q in sorted(zip(self.multilabel_classes, mp), key=lambda t: -t[1])
                if q >= MULTI_FAULT_THRESHOLD
            ]
            # Independent faults: P(no harm) multiplies across faults.
            multi_risk = float((1 - np.prod(1 - mp * self.ml_severity)) * 100)
            risk = max(class_risk, multi_risk)

            is_novel = bool(novelty_score[i] < self.novelty_threshold)
            fault = top
            if is_novel and (top == "normal" or confidence < UNKNOWN_MAX_CONFIDENCE):
                fault = "unknown_anomaly"
                risk = max(risk, UNKNOWN_MIN_RISK)
            info = fault_info(fault)
            fault_names = {d["fault"] for d in detected} | ({fault} if fault not in ("normal", "unknown_anomaly") else set())

            # Two simultaneous faults split the classifier's probability (~0.5 each), so
            # the evidence for a fault is the stronger of the classifier and the
            # multi-label detector, which scores each fault independently.
            evidence = max(confidence, float(dict(zip(self.multilabel_classes, mp)).get(fault, 0.0)))
            raw_alert = fault == "unknown_anomaly" or (fault != "normal" and evidence >= ALERT_CONFIDENCE)
            run_time = run_times[i] if run_times else None
            extra_warnings = []
            if raw_alert and run_time is not None and run_time < ENABLING_RUN_TIME_S.get(fault, 0):
                raw_alert = False
                extra_warnings.append(f"{info['label']} check paused: engine started {run_time:.0f} s ago "
                                      f"(needs {ENABLING_RUN_TIME_S[fault]} s to settle)")
            reading_time = timestamps[i] if timestamps else None
            result = {
                "diagnosis_id": uuid.uuid4().hex[:12],
                "timestamp": reading_time if reading_time is not None else time.time(),
                "vehicle_id": vehicle_ids[i] if vehicle_ids else None,
                "fault": fault,
                "fault_label": info["label"],
                "system": info["system"],
                "description": info["description"],
                "model_prediction": top,
                "confidence": round(confidence, 4),
                "uncertain": confidence < ALERT_CONFIDENCE,
                "risk_score": round(risk, 1),
                "risk_level": risk_level(risk),
                "raw_alert": raw_alert,
                "alert": raw_alert,
                "persistence": None,
                "probabilities": {self.classes[j]: round(float(p[j]), 4) for j in order},
                "detected_faults": detected,
                "multiple_faults": len(fault_names) >= 2,
                "novelty": {"is_novel": is_novel, "score": round(float(novelty_score[i]), 4),
                            "threshold": round(self.novelty_threshold, 4)},
                "shap_contributions": shap_rows[i],
                "explanations": self._abnormal_readings(row),
                "recommended_actions": info["actions"],
                "warnings": self._warnings(row) + extra_warnings,
                "latency_ms": round(latency, 2),
            }
            track_id = result["vehicle_id"] or default_vehicle_id
            if tracker is not None and track_id:
                tracker.update(track_id, result, info["severity"], probability=evidence, at=reading_time)
            elif tracker is not None:
                untracked(result)
            results.append(result)
        return results

    def diagnose(self, reading: dict, **kwargs) -> dict:
        return self.diagnose_many([reading], **kwargs)[0]


@lru_cache(maxsize=1)
def get_engine() -> DiagnosisEngine:
    return DiagnosisEngine.load()

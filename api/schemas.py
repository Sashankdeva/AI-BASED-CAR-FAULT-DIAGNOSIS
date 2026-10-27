"""Pydantic request/response contracts — the integration surface of this module."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """One OBD-II telemetry snapshot. Any sensor may be omitted (imputed), but at
    least 8 of the 15 must be present."""

    engine_rpm: Optional[float] = Field(None, description="Engine RPM", examples=[2200])
    vehicle_speed: Optional[float] = Field(None, description="km/h", examples=[60])
    engine_load: Optional[float] = Field(None, description="Calculated engine load, %", examples=[40])
    throttle_pos: Optional[float] = Field(None, description="Throttle position, %", examples=[22])
    coolant_temp: Optional[float] = Field(None, description="Coolant temperature, °C", examples=[90])
    intake_air_temp: Optional[float] = Field(None, description="Intake air temperature, °C", examples=[30])
    maf_rate: Optional[float] = Field(None, description="Mass air flow, g/s", examples=[14])
    short_fuel_trim: Optional[float] = Field(None, description="Short-term fuel trim, %", examples=[0])
    long_fuel_trim: Optional[float] = Field(None, description="Long-term fuel trim, %", examples=[1])
    o2_voltage: Optional[float] = Field(None, description="Upstream O2 sensor voltage, V", examples=[0.45])
    battery_voltage: Optional[float] = Field(None, description="System voltage, V", examples=[14.1])
    fuel_pressure: Optional[float] = Field(None, description="Fuel rail pressure, kPa", examples=[380])
    timing_advance: Optional[float] = Field(None, description="Ignition timing advance, °", examples=[14])
    misfire_count: Optional[float] = Field(None, description="Misfires per 1000 revolutions", examples=[1])
    oil_pressure: Optional[float] = Field(None, description="Engine oil pressure, kPa", examples=[300])
    vehicle_id: Optional[str] = Field(None, max_length=64,
                                      description="Vehicle identifier; alert persistence is tracked per vehicle. "
                                                  "Without it the reading is judged alone.")
    engine_run_time: Optional[float] = Field(None, ge=0, description="Seconds since engine start (PID 0x1F). "
                                             "Enables start-up rules, e.g. no charging verdict in the first 60 s.")
    timestamp: Optional[float] = Field(None, description="Unix seconds when the reading was taken. Live calls "
                                                         "use the server clock; batch logs use this if given.")


class BatchRequest(BaseModel):
    readings: List[SensorReading] = Field(..., min_length=1, max_length=5000)


class Explanation(BaseModel):
    sensor: str
    label: str
    value: float
    unit: str
    healthy_range: List[float]
    direction: str
    z_score: float
    message: str


class ShapContribution(BaseModel):
    feature: str
    label: str
    value: float
    unit: str
    contribution: float = Field(..., description="SHAP value toward the predicted class (+ supports it)")
    engineered: bool


class DetectedFault(BaseModel):
    fault: str
    label: str
    probability: float


class Persistence(BaseModel):
    window: int
    required: int
    hits: int
    immediate: bool
    status: str = Field(..., description="clear | pending | confirmed | untracked (no vehicle_id)")
    reason: str


class Novelty(BaseModel):
    is_novel: bool
    score: float
    threshold: float


class DiagnosisResult(BaseModel):
    diagnosis_id: str
    timestamp: float
    vehicle_id: Optional[str] = None
    fault: str = Field(..., description="Primary fault, or 'unknown_anomaly' for unrecognised patterns")
    fault_label: str
    system: str
    description: str
    model_prediction: str
    confidence: float = Field(..., description="Calibrated probability of the predicted class")
    uncertain: bool
    risk_score: float = Field(..., description="0–100, severity-weighted probability")
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    raw_alert: bool = Field(..., description="This single reading indicates a fault")
    alert: bool = Field(..., description="Confirmed alert after persistence check")
    persistence: Optional[Persistence]
    probabilities: Dict[str, float]
    detected_faults: List[DetectedFault]
    multiple_faults: bool
    novelty: Novelty
    shap_contributions: List[ShapContribution]
    explanations: List[Explanation]
    recommended_actions: List[str]
    warnings: List[str]
    latency_ms: float


class BatchSummary(BaseModel):
    total: int
    raw_alerts: int
    alerts: int
    by_fault: Dict[str, int]
    by_risk_level: Dict[str, int]


class BatchResponse(BaseModel):
    summary: BatchSummary
    results: List[DiagnosisResult]


class HealthResponse(BaseModel):
    status: str
    module: str
    version: str
    model_loaded: bool
    timestamp: float

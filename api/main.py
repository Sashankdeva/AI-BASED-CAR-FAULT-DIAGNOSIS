"""FastAPI service for AI-Based Car Fault Diagnosis.

Run standalone:   uvicorn api.main:app --reload --port 8005
Mount in a host:  host_app.mount("/car-fault", create_app())
"""

from __future__ import annotations

import io
import json
import os
import random
import secrets
import time
from collections import Counter, deque

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from car_fault import MODULE_ID, __version__
from car_fault.alerting import AlertTracker
from car_fault.config import (
    FAULT_NAMES,
    FAULTS,
    FRONTEND_DIR,
    MAX_UPLOAD_BYTES,
    METRICS_PATH,
    ROOT_DIR,
    SENSORS,
    UNKNOWN_FAULT,
)
from car_fault.data_generator import FAULT_PAIRS, sample_reading
from car_fault.diagnose import get_engine

from .schemas import (
    BatchRequest,
    BatchResponse,
    DiagnosisResult,
    HealthResponse,
    SensorReading,
)

MANIFEST_PATH = ROOT_DIR / "module.json"
_history: deque = deque(maxlen=200)  # recent alerts, in-memory
_tracker = AlertTracker()             # live alert persistence per vehicle_id


def _engine():
    try:
        return get_engine()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _reading_times(readings: list[dict], live: bool) -> list[float | None]:
    """Live calls are timed by the server clock (a client may report an earlier
    time, never a future one). Offline logs use their own timestamps, if any."""
    now = time.time()
    times = []
    for r in readings:
        ts = r.get("timestamp")
        ts = float(ts) if ts is not None and not pd.isna(ts) else None
        times.append(min(ts, now) if ts is not None and live else (now if live else ts))
    return times


def _run(readings: list[dict], tracker: AlertTracker | None, explain: bool = True,
         live: bool = True, default_vehicle_id: str | None = None) -> list[dict]:
    ids = [r.get("vehicle_id") for r in readings]
    try:
        results = _engine().diagnose_many(
            readings, vehicle_ids=ids, tracker=tracker, explain=explain,
            timestamps=_reading_times(readings, live), default_vehicle_id=default_vehicle_id,
            run_times=[r.get("engine_run_time") for r in readings])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    for res in results:
        if res["alert"]:
            _history.appendleft({k: res[k] for k in (
                "diagnosis_id", "timestamp", "vehicle_id", "fault", "fault_label",
                "confidence", "risk_score", "risk_level")})
    return results


def _batch(results: list[dict]) -> dict:
    return {
        "summary": {
            "total": len(results),
            "raw_alerts": sum(r["raw_alert"] for r in results),
            "alerts": sum(r["alert"] for r in results),
            "by_fault": dict(Counter(r["fault"] for r in results)),
            "by_risk_level": dict(Counter(r["risk_level"] for r in results)),
        },
        "results": results,
    }


router = APIRouter(prefix="/api/v1", tags=["car-fault-diagnosis"])


@router.get("/health", response_model=HealthResponse)
def health():
    try:
        get_engine()
        loaded = True
    except FileNotFoundError:
        loaded = False
    return HealthResponse(status="HEALTHY" if loaded else "DEGRADED", module=MODULE_ID,
                          version=__version__, model_loaded=loaded, timestamp=time.time())


@router.get("/manifest")
def manifest():
    """Integration manifest the host website reads to register this module."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@router.get("/model/info")
def model_info():
    return _engine().info


@router.get("/model/metrics")
def model_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "Metrics not found — train the model first.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@router.get("/sensors")
def sensors():
    """Input schema with units and plausible ranges — use it to build forms."""
    return SENSORS


@router.get("/faults")
def faults():
    return FAULTS | {"unknown_anomaly": UNKNOWN_FAULT}


@router.post("/diagnose", response_model=DiagnosisResult)
def diagnose(reading: SensorReading,
             stateless: bool = Query(False, description="Skip alert persistence (alert = this reading only)")):
    """Diagnose one reading. Alerts are confirmed per vehicle_id across consecutive calls;
    without a vehicle_id the reading is judged alone (status "untracked")."""
    return _run([reading.model_dump()], None if stateless else _tracker)[0]


def require_admin(x_admin_key: str | None = Header(None)) -> None:
    expected = os.getenv("CARFAULT_ADMIN_KEY")
    if not expected:
        raise HTTPException(403, "Reset is disabled. Set CARFAULT_ADMIN_KEY on the server to enable it.")
    if not x_admin_key or not secrets.compare_digest(x_admin_key, expected):
        raise HTTPException(403, "Missing or invalid X-Admin-Key header.")


@router.post("/vehicles/{vehicle_id}/reset", dependencies=[Depends(require_admin)])
def reset_vehicle(vehicle_id: str):
    """Clear a vehicle's alert-persistence history (e.g. after a repair). Requires X-Admin-Key."""
    _tracker.reset(vehicle_id)
    return {"vehicle_id": vehicle_id, "reset": True}


@router.post("/diagnose/batch", response_model=BatchResponse)
def diagnose_batch(req: BatchRequest, explain: bool = Query(False, description="Include SHAP (slower)")):
    """Readings are a time-ordered log: persistence runs per vehicle_id within this batch.
    Readings without vehicle_id are treated as one vehicle's log."""
    return _batch(_run([r.model_dump() for r in req.readings], AlertTracker(), explain,
                       live=False, default_vehicle_id="log"))


@router.post("/diagnose/csv", response_model=BatchResponse)
async def diagnose_csv(file: UploadFile = File(...),
                       explain: bool = Query(False, description="Include SHAP (slower)")):
    """Score an OBD-II CSV log (max 5 MB / 5000 rows). Optional columns: vehicle_id, timestamp."""
    raw = bytearray()
    while chunk := await file.read(1 << 20):
        raw += chunk
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"CSV larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        df = pd.read_csv(io.BytesIO(raw), nrows=5001)
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}") from exc
    if len(df) > 5000:
        raise HTTPException(413, "CSV limited to 5000 rows per request.")
    if "vehicle_id" in df.columns:
        df["vehicle_id"] = df["vehicle_id"].astype(str)
    df = df.astype(object).where(df.notna(), None)  # NaN cells -> missing sensors
    return _batch(_run(df.to_dict("records"), AlertTracker(), explain, live=False, default_vehicle_id="log"))


SCENARIOS = {
    **{f: FAULTS[f]["label"] for f in FAULT_NAMES},
    **{f"{a}+{b}": f"{FAULTS[a]['label']} + {FAULTS[b]['label']}" for a, b in FAULT_PAIRS[:4]},
    "unseen_rich_regulator": "Unseen: fuel pressure regulator stuck (not in training)",
    "unseen_coolant_sensor_open": "Unseen: coolant sensor open circuit (not in training)",
}


@router.get("/simulate/scenarios")
def simulate_scenarios():
    return SCENARIOS


@router.get("/simulate")
def simulate(scenario: str = Query("random", description="'random' or a key from /simulate/scenarios"),
             fault_rate: float = Query(0.35, ge=0, le=1)):
    """Synthetic live-feed reading for demos (no real car needed)."""
    if scenario == "random":
        scenario = random.choice(FAULT_NAMES[1:]) if random.random() < fault_rate else "normal"
    elif scenario not in SCENARIOS:
        raise HTTPException(400, f"Unknown scenario. Use 'random' or one of: {', '.join(SCENARIOS)}")
    return {"scenario": scenario, "reading": sample_reading(scenario)}


@router.get("/alerts")
def alerts(limit: int = Query(50, ge=1, le=200)):
    return list(_history)[:limit]


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI-Based Car Fault Diagnosis",
        description="OBD-II telemetry → ML fault classification → confidence & risk score → dashboard alert",
        version=__version__,
        root_path=os.getenv("CARFAULT_ROOT_PATH", ""),
    )
    origins = [o.strip() for o in os.getenv("CARFAULT_CORS_ORIGINS", "*").split(",")]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        # Reject oversized uploads before the multipart body is read and spooled.
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 64 * 1024:
            return JSONResponse({"detail": f"Request body larger than "
                                           f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."}, status_code=413)
        return await call_next(request)

    @app.middleware("http")
    async def revalidate_frontend(request: Request, call_next):
        # Host sites embed /widget.js; without this, browsers keep serving a stale
        # widget after an update. "no-cache" still allows cheap 304s via ETag.
        response = await call_next(request)
        if "/api/" not in request.url.path:
            response.headers["Cache-Control"] = "no-cache"
        return response

    if FRONTEND_DIR.exists():
        app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        def index():
            return RedirectResponse(url="ui/")

        @app.get("/widget.js", include_in_schema=False)
        def widget():
            return FileResponse(FRONTEND_DIR / "widget" / "car-fault-widget.js",
                                media_type="application/javascript")
    return app


app = create_app()

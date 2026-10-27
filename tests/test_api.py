import io
import time

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from car_fault.config import MAX_UPLOAD_BYTES
from car_fault.data_generator import generate, sample_reading

client = TestClient(app)


def test_health_and_manifest():
    assert client.get("/api/v1/health").json()["model_loaded"] is True
    m = client.get("/api/v1/manifest").json()
    assert m["id"] == "car-fault-diagnosis"
    assert m["ui"]["widget_tag"] == "car-fault-diagnosis"


def test_diagnose():
    body = sample_reading("low_oil_pressure", seed=7) | {"vehicle_id": "AP-39-1234"}
    r = client.post("/api/v1/diagnose", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["vehicle_id"] == "AP-39-1234"
    assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_diagnose_validation_error():
    assert client.post("/api/v1/diagnose", json={"engine_rpm": 900}).status_code == 422
    assert client.post("/api/v1/diagnose", json={"engine_rpm": "fast"}).status_code == 422


def _misfires(vehicle_id, n, start, step):
    return [client.post("/api/v1/diagnose", json=sample_reading("ignition_misfire", seed=50 + s)
                        | {"vehicle_id": vehicle_id, "timestamp": start + s * step}).json()
            for s in range(n)]


def test_persistence_per_vehicle():
    statuses = [r["persistence"]["status"] for r in _misfires("car-P", 4, time.time() - 30, 2.0)]
    assert statuses[0] == "pending" and "confirmed" in statuses
    stateless = client.post("/api/v1/diagnose", params={"stateless": True},
                            json=sample_reading("ignition_misfire", seed=1)).json()
    assert stateless["persistence"] is None and stateless["alert"]


def test_no_vehicle_id_is_not_pooled():
    # Two anonymous cars must never confirm each other's faults.
    results = [client.post("/api/v1/diagnose", json=sample_reading("ignition_misfire", seed=s)).json()
               for s in range(4)]
    assert all(r["persistence"]["status"] == "untracked" for r in results)
    assert all(r["persistence"]["hits"] == 0 for r in results)


def test_hits_too_close_together_do_not_confirm():
    statuses = [r["persistence"]["status"] for r in _misfires("car-fast", 4, time.time() - 5, 0.1)]
    assert "confirmed" not in statuses


def test_old_hits_expire():
    now = time.time()
    _misfires("car-old", 2, now - 3600, 2.0)            # an hour ago
    r = _misfires("car-old", 1, now - 1, 1.0)[0]        # one fresh hit
    assert r["persistence"]["hits"] == 1 and not r["alert"]


def test_reset_requires_admin_key(monkeypatch):
    monkeypatch.delenv("CARFAULT_ADMIN_KEY", raising=False)
    assert client.post("/api/v1/vehicles/car-P/reset").status_code == 403
    monkeypatch.setenv("CARFAULT_ADMIN_KEY", "s3cret-test-key")
    assert client.post("/api/v1/vehicles/car-P/reset").status_code == 403
    assert client.post("/api/v1/vehicles/car-P/reset", headers={"X-Admin-Key": "wrong"}).status_code == 403
    ok = client.post("/api/v1/vehicles/car-P/reset", headers={"X-Admin-Key": "s3cret-test-key"})
    assert ok.status_code == 200 and ok.json()["reset"]


def test_upload_size_limit():
    big = b"engine_rpm\n" + b"1000\n" * (MAX_UPLOAD_BYTES // 5 + 10)
    r = client.post("/api/v1/diagnose/csv", files={"file": ("big.csv", io.BytesIO(big), "text/csv")})
    assert r.status_code == 413


def test_batch_and_csv():
    readings = [sample_reading(f, seed=i) for i, f in enumerate(["normal", "o2_sensor_fault", "engine_overheating"])]
    r = client.post("/api/v1/diagnose/batch", json={"readings": readings}).json()
    assert r["summary"]["total"] == 3

    csv = generate(rows=80, seed=9).drop(columns=["fault", "driving_mode"]).to_csv(index=False)
    r = client.post("/api/v1/diagnose/csv", files={"file": ("fleet.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 200
    assert r.json()["summary"]["total"] == len(pd.read_csv(io.StringIO(csv)))


def test_simulate_and_alerts():
    s = client.get("/api/v1/simulate", params={"scenario": "engine_overheating"}).json()
    client.post("/api/v1/diagnose", json=s["reading"])
    assert any(a["fault"] == "engine_overheating" for a in client.get("/api/v1/alerts").json())
    assert client.get("/api/v1/simulate", params={"scenario": "nope"}).status_code == 400
    scenarios = client.get("/api/v1/simulate/scenarios").json()
    assert "unseen_rich_regulator" in scenarios and any("+" in k for k in scenarios)
    assert "unknown_anomaly" in client.get("/api/v1/faults").json()


def test_ui_served():
    assert client.get("/ui/").status_code == 200
    widget = client.get("/widget.js")
    assert "customElements" in widget.text


def test_frontend_not_cached_stale():
    for path in ("/widget.js", "/ui/", "/ui/widget/car-fault-widget.js"):
        assert client.get(path).headers["cache-control"] == "no-cache"
    # ETag revalidation still returns a cheap 304.
    etag = client.get("/ui/widget/car-fault-widget.js").headers["etag"]
    assert client.get("/ui/widget/car-fault-widget.js", headers={"If-None-Match": etag}).status_code == 304
    assert "cache-control" not in client.get("/api/v1/health").headers

# AI-Based Car Fault Diagnosis

**Project 5 · Sri Sashank Devarakonda (24B11AI085)**
AI-powered fault detection · calibrated confidence · explainable, preventive alerts

The model classifies OBD-II vehicle telemetry into 8 fault classes. Each diagnosis comes with:
- a **calibrated** confidence
- a 0–100 risk score
- **SHAP** explanations of why the model decided
- detection of **multiple simultaneous faults** and **unknown faults**
- **persistent alerts**, which repeat-check a fault before warning the driver to avoid false alarms

The module runs as a self-contained microservice with a versioned REST API, its own dashboard, and a drop-in web component. The team website can embed it without changing anything here.

```
Sensor Feed → Preprocessing → ML Detection → Fault Check → Confidence Score → Dashboard Alert
```

## Quick start

```bash
pip install -r requirements.txt
python -m car_fault.data_generator      # synthetic OBD-II data → data/
python -m car_fault.real_data --download  # real Toyota logs (~33 MB); training uses them if present
python -m car_fault.train               # trains all models → models/, reports/metrics.json
python -m car_fault.evaluate            # charts + robustness → reports/figures/, reports/evaluation.json
python -m car_fault.real_data           # false alarms + fault detection on held-out real trips
python -m uvicorn api.main:app --port 8005
```

On Windows you can run `start_api.bat` instead. Once the service is up:

- Dashboard: http://localhost:8005/ui/
- Swagger API docs: http://localhost:8005/docs
- Analysis notebook (with saved outputs): [notebooks/analysis.ipynb](notebooks/analysis.ipynb)
- Tests: `python -m pytest -q` (32 tests)

## Architecture

```
car_fault/                  framework-free ML package (importable by any host)
  config.py                 sensor schema, fault catalog (severity, actions), alert settings
  data_generator.py         synthetic OBD-II telemetry: single, two-fault and unseen-fault scenarios
  preprocessing.py          range clipping, imputation of missing sensors, 5 physics features
  train.py                  classifier + calibration, multi-fault detector, novelty detector
  diagnose.py               DiagnosisEngine: probabilities → risk → SHAP → unknown/multi-fault
  alerting.py               AlertTracker: per-vehicle alert persistence
  evaluate.py               EDA, ROC/PR, calibration, SHAP, robustness, persistence report
  real_data.py              carOBD real logs: download, trip split, semi-synthetic faults, real-car evaluation
api/
  main.py                   FastAPI service, all routes under /api/v1, create_app() factory
  schemas.py                Pydantic request/response contracts
frontend/
  index.html                standalone dashboard (wraps the widget)
  widget/car-fault-widget.js  <car-fault-diagnosis> web component (Shadow DOM, no deps)
notebooks/analysis.ipynb    full analysis walkthrough for presentation / viva
examples/                   host-website embed demo + FastAPI mount demo
module.json                 integration manifest (endpoints, widget tag, events)
```

## The ML

| Stage | What happens |
|---|---|
| **Sensor feed** | 15 OBD-II / telematics signals: RPM, speed, load, throttle, coolant and intake temps, MAF, short and long fuel trims, O2 voltage, battery voltage, fuel pressure, timing advance, misfire count, oil pressure. At least 8 must be present. Engine run time (PID 0x1F) is optional |
| **Preprocessing** | Clips values to plausible ranges. **Context-aware imputation:** a missing sensor is filled with what a *healthy* car reads at the current RPM, load, throttle and speed, using small models learned from healthy training data. So a sensor the car doesn't report can never look like a fault. Adds 5 features: total fuel trim, MAF vs. healthy expectation, oil pressure vs. healthy expectation, coolant − intake, charging deviation |
| **Training data** | 24k synthetic readings, plus 20k **semi-synthetic** readings: real healthy driving from 24 carOBD trips with fault signatures injected onto it. Some rows have the commonly unlogged sensors blanked, so the model learns to judge cars that don't report every PID |
| **ML detection** | Compares a Random Forest and a Gradient Boosting model with stratified 5-fold CV and keeps the better one (Random Forest). Probabilities are then **isotonic-calibrated** |
| **Fault check** | 8 classes. A **multi-label Random Forest** also reports simultaneous faults. A **Local Outlier Factor** novelty detector flags readings unlike anything in training ("Unrecognised Abnormal Pattern") |
| **Confidence score** | Calibrated probability. With two faults at once, the multi-fault detector's per-fault probability is used, so one fault doesn't hide the other. Risk = severity-weighted probability, 0–100, banded LOW / MEDIUM / HIGH / CRITICAL |
| **Explain** | **SHAP** values (TreeExplainer) show which features pushed the decision and in which direction. A rule-based list shows which sensors are outside the healthy range |
| **Dashboard alert** | Fires when the same fault appears in **3 of the last 5** readings of the same vehicle, within 5 minutes and lasting at least 2 s. Overheating and low oil pressure alert **immediately**. Weak or slow faults must persist longer: 30 s for unknown patterns, 60 s for charging and fuel-trim faults. **Enabling conditions**, as in real OBD-II monitors: no charging verdict in the first 60 s after engine start, and no fuel-trim verdict in the first 3 minutes. Readings without a `vehicle_id` are never pooled together |

## Results on synthetic data

All numbers are on held-out data the models never trained on. Charts are in `reports/figures/`.

| Metric | Value |
|---|---|
| Test accuracy / macro-F1 | **93.1% / 0.936** |
| ROC-AUC per class | 0.975 – 1.000 |
| Calibration error (ECE) | 0.080 → **0.016** after calibration |
| False alerts on healthy readings | 6.3% per reading → **0.6%** with persistence |
| Healthy cars with ≥1 false alert in 20 readings | 79% → **11%** with persistence |
| Time to alert | 1 reading (critical faults), ~3 readings (others) |
| Two simultaneous faults: exact match | 91.7% (precision 1.00, recall 0.96) |
| Unseen faults flagged as novel | **100%**, with 0.8% of known data flagged |

**Robustness:** what happens when the data doesn't look exactly like the training data.

| Scenario | Accuracy | Macro-F1 |
|---|---|---|
| Clean test set | 0.931 | 0.936 |
| Unseen vehicles (new random seed) | 0.932 | 0.936 |
| Sensor noise +10% / +25% of sd | 0.899 / 0.856 | 0.910 / 0.875 |
| Sensor drift (ageing sensors) | 0.895 | 0.900 |
| 10% of readings missing | 0.861 | 0.865 |
| Oil pressure not reported | 0.858 | 0.811 |
| Mild faults only | 0.866 | 0.865 |

## Real-world check (carOBD dataset)

The model is also checked on real driving: **112,963 readings from 35 trips** of a 2014 Toyota Etios. This is the [carOBD dataset](https://github.com/eron93br/carOBD), 1 Hz logs of normal use with no fault labels. Please cite the authors' IEEE paper if you use it. The logs contain 10 of our 15 sensors; MAF, O2 voltage, fuel pressure, misfire count and oil pressure are not logged.

The data isn't stored in this repo. Fetch it with `python -m car_fault.real_data --download` (35 files, ~33 MB), then retrain and run `python -m car_fault.real_data`.

Trips are split **by trip**: 24 are used for training and 11 are held out (35,236 readings, about 9.8 hours of driving). v1 was trained on synthetic data only; v2 is the current model.

| Held-out real healthy trips | v1 | v2 |
|---|---|---|
| Readings raising a fault (raw) | 74.4% | **1.9%** |
| Readings with a confirmed alert | 56.5% | **0.0%** |
| Trips with at least one confirmed alert | 100% | **0%** |
| Readings flagged as novel | 74.3% | 0.9% |

| Faults injected onto held-out real trips, % correctly diagnosed | v1, 10 logged sensors | v2, 10 logged sensors | v2, all 15 sensors |
|---|---|---|---|
| Engine overheating | 98% | 100% | 100% |
| Battery / charging | 100% | 99% | 99% |
| O2 sensor fault | 0% | 89% | 96% |
| Fuel system lean | 0% | 68% | 96% |
| MAF sensor fault | 4% | 50% | 85% |
| Low oil pressure | 0% | 0% | 89% |
| Ignition misfire | 0% | 0% | 100% |

**Final test on never-seen trips.** The rules below were tuned on the 11 held-out trips, so I downloaded **57 more trips (36.3 hours, 130,811 readings)** that played no part in training or tuning, and ran the model on them once, with no changes afterwards. Run it with `python -m car_fault.real_data --download-fresh` and then `--fresh`.

| Never-seen real trips (v2) | Result |
|---|---|
| Readings raising a fault (raw) | 6.0% |
| Readings with a confirmed alert | **0.008%** (10 of 130,811) |
| Trips with at least one confirmed alert | **1 of 57** |
| Faults injected onto these trips, 10 logged sensors | overheating 100%, battery 99%, O2 85%, lean 65%, MAF 47% |

The raw rate is higher than on the tuning trips (6.0% vs. 1.9%, mostly brief MAF and fuel-trim calls), so the model alone is less clean on new driving than the tuned figure suggested. The persistence and enabling rules still hold up: about one false alert in 36 hours of real driving.

**Why v1 failed.** It filled missing sensors with training averages that didn't fit the car's RPM and load. Its simulator was also wrong about this car: the real Etios has a hotter intake (46 °C vs. 30 °C), much less ignition advance (9° vs. 23°), and a lower charging voltage (13.7 V vs. 14.1 V). See `reports/figures/real_vs_synthetic_ranges.png`.

**What fixed it.** Context-aware imputation, learning "normal" from real healthy trips, training with blanked sensors, OBD-style enabling conditions, and longer persistence for slow faults. See `reports/figures/real_data_before_after.png`.

**What it can't do.** With only this car's 10 sensors, **low oil pressure and misfires are undetectable**, because their evidence lives in sensors the car doesn't report.

**Caveats for the viva:**
- **The rules were tuned on the held-out trips.** So the 0% there is optimistic. The untouched final test (57 trips) is the honest figure: 1 trip alerted in 36 hours.
- **The faults are still simulated signatures.** No publicly available dataset labels real faults, so real fault *detection* is still unproven.
- **It's one car.** Other makes and engines will have different "normal" ranges.

**Other design decisions:**
- **Isolation Forest was tried first for novelty detection and failed** (it caught 2–3% of unseen faults). It splits on one random feature at a time, so a reading that is extreme on just one sensor barely stands out. Local Outlier Factor compares local density across all features and catches 100%.
- **Fitting the novelty detector on all 77k real rows made it worse** (more false alarms and a 45 MB model). LOF measures *relative* density, so denser training data makes new trips look sparser. It's fitted on a 20k-row sample instead.
- **MAF-sensor faults are the hardest class** (F1 0.78) because, like a lean fuel system, they push fuel trims positive. A mechanic faces the same ambiguity.
- **SHAP explains a random 60 of the forest's 300 trees.** A forest predicts the average of its trees, so this is an unbiased estimate of the full SHAP values (0.993 correlation in testing) at about a fifth of the cost.

Full numbers are in `reports/metrics.json`, `reports/evaluation.json` and `reports/real_data_evaluation.json` (v1 is in `real_data_evaluation_v1.json`).

## Limitations

- **One real car, no real faults.** Real healthy data comes from a single 2014 Toyota Etios, and faults are injected signatures rather than real failures. On the untouched final test, raw per-reading false faults are 6%; persistence brings confirmed alerts down to 1 trip in 57.
- **Simulated sensors in training.** The 5 sensors carOBD doesn't log are simulated for the semi-synthetic training rows, using the generator's physics.
- **Snapshot model.** Each reading is classified on its own. Persistence adds smoothing over time, but sequence models (rolling-window features, LSTM) could catch slowly developing faults earlier.
- **Limited fault combinations.** Multi-fault training covers 10 compatible two-fault pairs. Novelty detection is tested on 2 unseen faults, and subtler unknown faults may look normal. A stuck-rich regulator is sometimes labelled an O2 fault, but it is still flagged as novel and alerted.
- **Demo-grade state.** Alert history and persistence are kept in memory, so they are lost on restart and break if the server runs several worker processes. There is no live OBD-II adapter yet; a real ELM327 bridge would call the same `/api/v1/diagnose` endpoint.
- **Diagnose calls are not authenticated.** Only the reset endpoint is protected. Anyone who can reach the API can send readings under any `vehicle_id`, so a production deployment would need per-vehicle authentication, such as device tokens.

## API (v1)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Liveness check and whether the model is loaded |
| GET | `/api/v1/manifest` | Integration manifest (`module.json`) |
| POST | `/api/v1/diagnose` | Diagnose one reading. Persistence is tracked per `vehicle_id` using the server clock; without an ID the status is `untracked`; `?stateless=true` skips persistence |
| POST | `/api/v1/diagnose/batch` | Up to 5,000 time-ordered readings sent as JSON; `?explain=true` adds SHAP |
| POST | `/api/v1/diagnose/csv` | Diagnose an uploaded fleet CSV log, up to 5 MB and 5,000 rows. Blank cells count as missing sensors. Optional `vehicle_id` and `timestamp` columns |
| POST | `/api/v1/vehicles/{id}/reset` | Clear a vehicle's alert history, e.g. after a repair. Needs an `X-Admin-Key` header matching the `CARFAULT_ADMIN_KEY` environment variable; disabled if that variable is unset |
| GET | `/api/v1/simulate?scenario=…` | Synthetic reading for demos, including multi-fault and unseen-fault scenarios |
| GET | `/api/v1/simulate/scenarios` | List of demo scenarios |
| GET | `/api/v1/alerts` | Recent confirmed alerts |
| GET | `/api/v1/sensors`, `/api/v1/faults` | Input schema and fault catalog |
| GET | `/api/v1/model/info`, `/api/v1/model/metrics` | Model metadata and evaluation |

```bash
curl -X POST http://localhost:8005/api/v1/diagnose -H "Content-Type: application/json" -d '{"vehicle_id":"AP39-1234","engine_rpm":2200,"vehicle_speed":60,"engine_load":40,"throttle_pos":22,"coolant_temp":121,"intake_air_temp":35,"maf_rate":14,"short_fuel_trim":0,"long_fuel_trim":1,"o2_voltage":0.45,"battery_voltage":14.1,"fuel_pressure":380,"timing_advance":8,"misfire_count":1}'
```

Each response includes these fields:

| Field | Meaning |
|---|---|
| `fault` | The diagnosed fault, or `unknown_anomaly` |
| `confidence` | Calibrated confidence |
| `risk_score`, `risk_level` | 0–100 risk and its band |
| `raw_alert` | This single reading indicates a fault |
| `alert` | The alert is confirmed after the persistence check |
| `persistence` | How many recent readings showed the fault, and the status |
| `detected_faults`, `multiple_faults` | All faults found, and whether there is more than one |
| `novelty` | Whether the reading looks unlike the training data |
| `shap_contributions` | Features that pushed the decision, with SHAP values |
| `explanations` | Sensors outside the healthy range |
| `recommended_actions` | What the driver should do |
| `warnings` | Missing or clipped sensors |

## Integrating into the team website or app

The website can use this module in any of four ways. Pick the one that fits the shared site.

**1. Embed the widget (easiest).** Works with any website: plain HTML, React, Vue, or a WebView in a mobile app.
```html
<script src="http://localhost:8005/widget.js"></script>
<car-fault-diagnosis api-base="http://localhost:8005/" vehicle-id="AP39-1234"></car-fault-diagnosis>
<script>
  document.addEventListener("carfault:alert", (e) => console.log(e.detail.fault_label, e.detail.risk_level));
</script>
```
See `examples/host-integration.html`. The `carfault:result` event fires on every diagnosis; `carfault:alert` fires only for confirmed alerts.

**2. Call the REST API.** Build your own UI and call `POST /api/v1/diagnose`. The contracts are in `/openapi.json`, so you can generate typed clients from it.

**3. Use an API gateway or reverse proxy.** Run each team member's service on its own port and route by path prefix:
```nginx
location /car-fault/ { proxy_pass http://car-fault:8005/; }
```
The frontend uses relative URLs, so it works under any prefix.

**4. Mount it in-process.** If the shared backend is FastAPI: `host.mount("/car-fault", create_app())`. See `examples/mount_in_host.py`.

To limit which sites can call the API, set `CARFAULT_CORS_ORIGINS=https://team-site.example`. To enable vehicle resets, set `CARFAULT_ADMIN_KEY` to a long random secret, and keep it on the server only, never in a web page. Model and data paths can be overridden with `CARFAULT_MODEL_DIR` and `CARFAULT_DATA_DIR`.

**Docker:**
```bash
docker build -t car-fault-diagnosis .
docker run -p 8005:8005 car-fault-diagnosis
```

# MicroTwin

![CI](https://github.com/cedl-dev/twin-poc-cv/actions/workflows/ci.yml/badge.svg)

A one-day, end-to-end **Micro Digital Twin** proof of concept: a synthetic particle-accelerator sensor feed, an ETL pipeline, an MLOps-tracked unsupervised anomaly detector, and a REST inference API — built as a portfolio piece for an **AI Software Support Engineer** application on the **TwinRISE** project at **GANIL** (a nuclear physics research facility).

It is not a real accelerator integration — there is no live EPICS/TANGO connection, and the sensor data is generated, not measured. The goal is to demonstrate the full software engineering pattern (data → ETL → tracked training → served model) end to end, with the same rigor (tests, reproducibility, containerization) a real deployment would require.

## Architecture

```
twin-poc-cv/
├── src/microtwin/
│   ├── config.py            # centralized settings, overridable via env vars
│   ├── cli.py                # console-script entry points (Slurm/PBS-friendly)
│   ├── data/
│   │   ├── generate.py       # vectorized NumPy mock sensor data (no Python loops)
│   │   └── etl.py             # cleaning, float32 downcasting, Parquet persistence
│   ├── training/
│   │   └── train.py           # Isolation Forest + MLflow tracking/registry
│   └── api/
│       ├── schemas.py         # Pydantic request/response models
│       └── main.py             # FastAPI inference service
├── tests/                    # pytest, one file per pipeline stage
├── docker/Dockerfile          # multi-stage build, non-root runtime user
├── docker-compose.yml         # trainer -> {api, mlflow} orchestration
└── .github/workflows/ci.yml   # lint + type-check + test, on every push
```

Each pipeline stage is a single-responsibility, independently testable module, composed by thin CLI entry points (`microtwin-build-dataset`, `microtwin-train`) rather than one monolithic script.

## Quickstart

### Option A — Docker (no local Python needed)

```bash
docker compose up --build
```

This runs a `trainer` job (generates the dataset, trains and registers the model), then starts the API and MLflow UI once training succeeds:

- API: [http://localhost:8000/docs](http://localhost:8000/docs) (interactive Swagger UI)
- MLflow: [http://localhost:5001](http://localhost:5001) (experiment runs and the model registry)

### Option B — Local virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

microtwin-build-dataset   # generates data/processed/sensor_data.parquet
microtwin-train           # trains and registers the model with MLflow

uvicorn microtwin.api.main:app --reload
```

### Try the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature_c": 18.1, "beam_intensity_ua": 49.8}'
# {"is_anomaly":false,"anomaly_score":0.36,"sensor_status":"NOMINAL","originating_pv":null}

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature_c": 45.0, "beam_intensity_ua": 50.0}'
# {"is_anomaly":true,"anomaly_score":-0.02,"sensor_status":"CRITICAL","originating_pv":"GANIL:CRYO:TEMP01"}
```

`originating_pv` reports an EPICS-style process-variable name (see `config.SENSOR_PV_NAMES`), and `sensor_status` (`NOMINAL`/`WARNING`/`CRITICAL`) is the kind of business-facing alarm tag a real control-system integration would need — not a bare model score.

### Run the tests

```bash
pytest          # 26 tests, fully hermetic — no pre-trained model or dataset required
ruff check .    # lint
mypy src        # type-check
```

## Design decisions

| Choice | Why |
|---|---|
| **Vectorized NumPy generation, no `for` loops** | Generates 5M+ rows in well under a second (asserted by a test) — the difference between interpreted per-row Python and compiled array operations. |
| **Parquet, not CSV** | Columnar and typed: compresses better, preserves the `float32` dtype set by the ETL step (round-trip verified by a test), and is the scientific-data format named in the job posting. |
| **`float64` → `float32` downcasting** | pandas defaults to double precision; a physical sensor's real precision doesn't need it. Roughly halves memory (measured and logged at ETL time), a real concern at HPC scale. |
| **MLflow (SQLite-backed tracking + Model Registry)** | Every training run logs its hyperparameters, metrics, and a versioned model artifact — the API always loads a traceable model version (`models:/microtwin-isolation-forest/latest`), never a loose pickle file with no history. A plain `file:` tracking store doesn't support the Model Registry; SQLite does, at zero extra infrastructure cost. |
| **FastAPI with a `lifespan` startup hook** | The model is loaded once, at process startup, and kept resident in memory — not reloaded from MLflow on every request. Pydantic validates incoming JSON before any inference work happens. |
| **Isolation Forest, unsupervised** | A real accelerator has no labeled "this was a fault" history to train on. Isolation Forest learns what "normal" looks like from unlabeled data and flags what doesn't fit — the realistic setting for this kind of monitoring. |
| **`max_samples=4096` (not scikit-learn's default 256)** | The default caps each tree's depth at 8, which can hide an extreme outlier on a single feature in a crowded leaf before it's isolated. Found by testing the API with a manual out-of-range value, not by a metric — see `cheatsheet/03-training.md` for the full investigation. |
| **`src`-layout + console-script entry points** | `pip install -e .` gives real, importable, testable modules and real CLI commands (`microtwin-train`, etc.) — the same commands a Slurm/PBS job step would call, no notebook state involved. |
| **Multi-stage Dockerfile, non-root runtime user** | The final image ships only the built virtualenv, not build tools or intermediate layers; running as a non-root user is a baseline container security practice. |

## Testing philosophy

Every test is hermetic: none of them depend on a pre-existing trained model or dataset on disk. `tests/test_api.py` replaces the MLflow-loaded model with a small one trained in-memory (via `monkeypatch`) specifically so the suite passes on a fresh clone, before `microtwin-train` has ever run — verified by actually simulating a fresh clone and running the full CI sequence against it before enabling CI.

## What's deliberately out of scope

This is a one-day PoC, not a production deployment. These items from the broader job posting are acknowledged, not implemented:

- **Federated learning** (FedBioMed, Flower) — the training pipeline's single-responsibility structure (a pure function taking a DataFrame, returning a fitted model) would plug into a federated round without restructuring, but no federated node/aggregator is implemented here.
- **HPC job submission** (Slurm, PBS) — the pipeline is Slurm-submittable in shape (pure CLI entry points, no notebook state, config via environment variables), but no job scripts are included.
- **HDF5 / JSON-LD / RDF** — Parquet is used for this tabular sensor log; HDF5 would be the natural fit for raw waveform/multi-dimensional scientific data instead.
- **Kubernetes / Apptainer** — Docker Compose only; the same image would need no changes to run under Kubernetes or convert to a Singularity/Apptainer image for an HPC cluster.
- **Real EPICS/TANGO integration** — `originating_pv` simulates the *shape* of that integration (a PV name and an alarm-level tag in the response); there is no live connection to a control system.

## Tech stack

Python 3.9+ · pandas · NumPy · PyArrow (Parquet) · scikit-learn · MLflow · FastAPI · Pydantic · pytest · ruff · mypy · Docker / Docker Compose · GitHub Actions

## License

MIT

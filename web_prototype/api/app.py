"""
api/app.py
===========
FastAPI backend wired directly to the real Red Team / Blue Team pipeline
in this repo. No numbers are hardcoded here or in the frontend that
consumes this API -- every report endpoint reads the actual output files
on disk, live, and the pipeline endpoints trigger the actual scripts as
subprocesses. A separate, genuine online scoring path (POST /api/score,
see inference.py) loads the trained Stage 1+2 model once at startup and
scores single transactions without retraining or shelling out to any
batch script.

Run, from anywhere (cwd-independent -- see web_prototype/run_api.py):

    python3 web_prototype/run_api.py --reload

The previously documented `python -m uvicorn api.app:app` invocation
from the repo root does NOT work on its own -- `api` is a package under
`web_prototype/`, not a top-level package at the repo root, so
`import api.app` raises `ModuleNotFoundError` unless the process's cwd
already happens to be `web_prototype/`:

    cd web_prototype && uvicorn api.app:app --reload --port 8000

Then open http://localhost:8000/docs for interactive API docs, or point
the dashboard frontend's fetch() calls at http://localhost:8000/api/...

CORS / environment configuration is read from process env vars via
settings.py. Local dev with zero env vars set behaves exactly as before
(CORS wide open); staging/production REQUIRE an explicit
CORS_ALLOWED_ORIGINS or the process refuses to start -- see
settings.py's docstring, and web_prototype/api/.env.example for the
full list of variables (deployment notes for Render in
web_prototype/api/README.md).
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Make `import config`, `import reports`, etc. work regardless of how
# uvicorn was launched (as `api.app:app` from repo root, or `app:app`
# from inside api/), AND make `import red_team...` (needed by schemas.py /
# inference.py) work even when the documented `PYTHONPATH=src` prefix was
# forgotten -- see run_api.py's own docstring about this exact class of bug.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for _p in (str(_THIS_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.encoders import jsonable_encoder  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import inference  # noqa: E402
import pipeline_runner  # noqa: E402
import reports  # noqa: E402
from config import REPORTS, STAGES  # noqa: E402
from logging_config import configure_logging  # noqa: E402
from schemas import ObservableAttackTrace, ScoreResponse  # noqa: E402
from settings import ConfigError, load_settings  # noqa: E402

try:
    settings = load_settings()
except ConfigError as exc:
    # Fail loudly at import time -- an unsafe CORS config in staging/prod
    # must never silently start. See settings.py's docstring.
    raise SystemExit(f"[api/app.py] Refusing to start: {exc}") from exc

configure_logging(settings.log_level)
logger = logging.getLogger("ai_defense.api")

model_registry = inference.ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once, here -- never inside a request handler (requirement H).
    # A failed load does NOT crash the process: /healthz still answers,
    # /readyz honestly reports not-ready, and /api/score returns 503
    # instead of a raw traceback.
    model_registry.load()
    yield


app = FastAPI(
    title="AI Defense Lab -- Red Team / Blue Team API",
    description="Backend for the AI Defense Lab closed-loop prototype.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Assigns a request_id, times the request, and logs a single
    structured line per request. Deliberately logs only method/path/
    status/latency -- never the request body (see logging_config.py's
    docstring re: not logging transaction data)."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.exception(
            "request_failed",
            extra={"request_id": request_id, "method": request.method, "path": request.url.path, "latency_ms": round(latency_ms, 2)},
        )
        raise
    latency_ms = (time.perf_counter() - t0) * 1000.0
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_handled",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
        },
    )
    return response


# ---------------------------------------------------------------------------
# Structured error handling (requirement F) -- never leak tracebacks,
# filesystem paths, or other internals to API clients.
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Pydantic's default error detail is already client-safe (field paths +
    # messages, no internals) -- pass it through, just in our own envelope.
    # jsonable_encoder is required here: exc.errors() can embed the raw
    # invalid input value (e.g. a Decimal from a failed `gt=0` check on
    # `amount`), which json.dumps() cannot serialize on its own -- this was
    # caught by test_negative_amount_rejected actually failing with a 500
    # instead of the intended 422, not assumed.
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            {
                "error": "validation_error",
                "request_id": getattr(request.state, "request_id", None),
                "detail": exc.errors(),
            }
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    logger.exception("unhandled_exception", extra={"request_id": request_id})
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "request_id": request_id,
            "detail": "An unexpected error occurred. This has been logged.",
        },
    )


# ---------------------------------------------------------------------------
# Health / readiness (requirement D)
# ---------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    """Liveness only -- the process is up and answering HTTP. Does NOT
    imply the model is loaded; see /readyz for that."""
    return {"status": "alive"}


@app.get("/readyz")
def readyz():
    """Readiness -- true only if the Stage 1+2 inference artifacts loaded
    successfully at startup. A load balancer / orchestrator should route
    traffic to POST /api/score only when this is true; the
    pipeline-dashboard endpoints below don't depend on this at all."""
    body = {
        "ready": model_registry.ready,
        "model_version": model_registry.model_version if model_registry.ready else None,
    }
    if not model_registry.ready:
        body["error"] = model_registry.load_error
        return JSONResponse(status_code=503, content=body)
    return body


@app.get("/api/health")
def health():
    """Deprecated alias for /healthz, kept for backward compatibility with
    the existing dashboard frontend -- do not remove without updating
    web_prototype/dashboard/index.html first."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Online single-transaction scoring (Stage 1+2 only -- see inference.py's
# module docstring for exactly what this does and does not compute, and why)
# ---------------------------------------------------------------------------
@app.post("/api/score", response_model=ScoreResponse)
def score(trace: ObservableAttackTrace, request: Request):
    if not model_registry.ready:
        raise HTTPException(503, "Model artifacts are not loaded yet. Check /readyz.")

    trace_dict = trace.model_dump(mode="json")
    try:
        result = inference.score_trace(trace_dict, model_registry)
    except inference.ModelNotReady as exc:
        raise HTTPException(503, str(exc)) from exc

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(
        "score_computed",
        extra={
            "request_id": request_id,
            "trace_id": trace.trace_id,
            "decision": result["decision"],
            "risk_score": result["risk_score"],
            "model_version": result["model_version"],
            "inference_latency_ms": result["inference_latency_ms"],
        },
    )
    return {"trace_id": trace.trace_id, "request_id": request_id, **result}


# ---------------------------------------------------------------------------
# Report endpoints -- live reads of real output files, every call
# ---------------------------------------------------------------------------
@app.get("/api/reports/dashboard")
def get_dashboard():
    """Everything the dashboard needs in one call. Every field is a live
    disk read (see reports.dashboard_summary) -- re-run a stage and the
    very next call to this endpoint reflects it."""
    return reports.dashboard_summary()


@app.get("/api/reports/{key}")
def get_report(key: str):
    if key not in REPORTS:
        raise HTTPException(404, f"Unknown report key. Known keys: {sorted(REPORTS)}")
    path = REPORTS[key]
    if path.suffix == ".jsonl":
        return reports.read_jsonl(key)
    if path.suffix == ".json":
        return reports.read_json(key)
    raise HTTPException(400, f"Report '{key}' is not JSON/JSONL (it's {path.suffix}); use /api/reports/{key}/file")


@app.get("/api/reports/{key}/file")
def get_report_file(key: str):
    """Serve a report file as-is -- used for the SHAP PNG, and works for
    any other file-type report too."""
    if key not in REPORTS:
        raise HTTPException(404, f"Unknown report key. Known keys: {sorted(REPORTS)}")
    path = REPORTS[key]
    if not path.exists():
        raise HTTPException(404, f"{path.name} hasn't been generated yet -- run the pipeline first.")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Pipeline control
# ---------------------------------------------------------------------------
@app.get("/api/pipeline/stages")
def get_stages():
    """Static catalog of what the closed loop consists of -- lets the
    frontend render the pipeline diagram without hardcoding it too."""
    return {"stages": STAGES}


class RunRequest(BaseModel):
    stage_ids: list[str] | None = None  # None = run all stages, in order


@app.post("/api/pipeline/run")
def run_pipeline(req: RunRequest):
    try:
        job_id = pipeline_runner.start_run(req.stage_ids)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"job_id": job_id}


@app.get("/api/pipeline/status/{job_id}")
def pipeline_status(job_id: str):
    job = pipeline_runner.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    return job


@app.get("/api/pipeline/jobs")
def pipeline_jobs():
    return {"jobs": pipeline_runner.list_jobs()}

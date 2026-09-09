from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("ENVIRONMENT", "development")

API_DIR = Path(__file__).resolve().parent.parent   # web_prototype/api
REPO_ROOT = API_DIR.parent.parent                   # repo root
SRC_DIR = REPO_ROOT / "src"

for p in (str(API_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def real_ato_trace() -> dict:
    """A real, unmodified observable_trace straight from the Red Team ATO
    corpus this repo already ships -- used for the genuine end-to-end
    inference test (requirement I: at least one real test using actual
    repository artifacts, not a mock)."""
    corpus_path = REPO_ROOT / "reports" / "ato_corpus_raw.json"
    with open(corpus_path) as f:
        corpus = json.load(f)
    return corpus[0]["observable_trace"]


@pytest.fixture()
def app_client():
    """Fresh TestClient per test, running the real lifespan (loads the real
    model artifacts once, same as production)."""
    from fastapi.testclient import TestClient

    import app as api_app

    with TestClient(api_app.app) as client:
        yield client

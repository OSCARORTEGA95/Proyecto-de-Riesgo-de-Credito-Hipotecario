from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def solicitudes() -> list:
    return [
        {"credit_score": 780, "dti": 28, "oltv": 75, "orig_upb": 320000, "orig_rate": 6.5, "orig_term": 360,
         "purpose": "P", "prop_type": "SF", "state": "TX"},
        {"credit_score": 640, "dti": 48, "oltv": 97, "orig_upb": 180000, "orig_rate": 7.9, "orig_term": 360,
         "purpose": "C", "prop_type": "CO", "state": "FL"},
    ]


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}

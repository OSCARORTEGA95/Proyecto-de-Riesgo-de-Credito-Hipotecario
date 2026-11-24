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


@pytest.fixture(scope="module")
def creditos_p2() -> list:
    """Tres créditos al día de la muestra de prueba de 2024 (ejemplo guardado por MLflow en knn-p2)."""
    return [{"loan_id": "F22Q10315716", "period": 202409, "edad": 30.0, "pct_saldo": 0.9531533622559654, "dif_tasa": 0.0, "eltv": 99.0, "atrasos_12m": 0.0, "max_atraso_12m": 0.0, "meses_desde_atraso": None, "indulgencia_12m": 0.0, "credit_score": 787.0, "dti": 42.0, "oltv": 95.0, "ocltv": 95.0, "orig_rate": 3.875, "orig_upb": 461000.0, "orig_term": 360.0, "mi_pct": 30.0, "num_borrowers": 2.0, "num_units": 1.0, "con_atraso_previo": False, "diferimiento": False, "diferimiento_mes": False, "saldo_sin_interes": False, "mora_desastre": False, "modificado": False, "plan_asistencia": None, "purpose": "P", "occupancy": "P", "channel": "R", "first_time_homebuyer": "N", "prop_type": "CO", "valuation_method": 2.0}, {"loan_id": "F21Q40631396", "period": 202409, "edad": 32.0, "pct_saldo": 0.9253067510548524, "dif_tasa": 0.0, "eltv": 48.0, "atrasos_12m": 0.0, "max_atraso_12m": 0.0, "meses_desde_atraso": None, "indulgencia_12m": 0.0, "credit_score": 663.0, "dti": 39.0, "oltv": 57.0, "ocltv": 57.0, "orig_rate": 2.99, "orig_upb": 237000.0, "orig_term": 300.0, "mi_pct": 0.0, "num_borrowers": 2.0, "num_units": 1.0, "con_atraso_previo": False, "diferimiento": False, "diferimiento_mes": False, "saldo_sin_interes": False, "mora_desastre": False, "modificado": False, "plan_asistencia": None, "purpose": "C", "occupancy": "P", "channel": "C", "first_time_homebuyer": "N", "prop_type": "SF", "valuation_method": 2.0}, {"loan_id": "F21Q12247918", "period": 202407, "edad": 39.0, "pct_saldo": 0.9150040298507462, "dif_tasa": 0.0, "eltv": 54.0, "atrasos_12m": 0.0, "max_atraso_12m": 0.0, "meses_desde_atraso": None, "indulgencia_12m": 0.0, "credit_score": 711.0, "dti": 35.0, "oltv": 80.0, "ocltv": 80.0, "orig_rate": 2.875, "orig_upb": 268000.0, "orig_term": 360.0, "mi_pct": 0.0, "num_borrowers": 2.0, "num_units": 1.0, "con_atraso_previo": False, "diferimiento": False, "diferimiento_mes": False, "saldo_sin_interes": False, "mora_desastre": False, "modificado": False, "plan_asistencia": None, "purpose": "P", "occupancy": "P", "channel": "C", "first_time_homebuyer": "N", "prop_type": "PU", "valuation_method": 1.0}]


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}

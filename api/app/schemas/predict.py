from typing import Any, List, Optional

from modelo_riesgo.p2 import DataInputSchemaP2
from modelo_riesgo.processing.validation import DataInputSchema
from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any] = None
    version: str
    predictions: Optional[List[float]] = None


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "inputs": [
                        {
                            "credit_score": 640,
                            "dti": 48,
                            "oltv": 97,
                            "orig_upb": 180000,
                            "orig_rate": 7.9,
                            "orig_term": 360,
                            "purpose": "C",
                            "prop_type": "CO",
                            "state": "FL",
                        }
                    ]
                }
            ]
        }
    }


class MultipleDataInputsP2(BaseModel):
    """Créditos activos y al día en un mes de corte (pregunta 2)."""
    inputs: List[DataInputSchemaP2]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {'inputs': [{'loan_id': 'F21Q40369447',
                             'period': 202409,
                             'edad': 33.0,
                             'pct_saldo': 0.9415245033112584,
                             'dif_tasa': 0.0,
                             'eltv': 50.0,
                             'atrasos_12m': 0.0,
                             'max_atraso_12m': 0.0,
                             'meses_desde_atraso': None,
                             'indulgencia_12m': 0.0,
                             'credit_score': 694.0,
                             'dti': 33.0,
                             'oltv': 66.0,
                             'ocltv': 66.0,
                             'orig_rate': 3.25,
                             'orig_upb': 151000.0,
                             'orig_term': 360.0,
                             'mi_pct': 0.0,
                             'num_borrowers': 1.0,
                             'num_units': 1.0,
                             'con_atraso_previo': False,
                             'diferimiento': False,
                             'diferimiento_mes': False,
                             'saldo_sin_interes': False,
                             'mora_desastre': False,
                             'modificado': False,
                             'plan_asistencia': None,
                             'purpose': 'C',
                             'occupancy': 'P',
                             'channel': 'R',
                             'first_time_homebuyer': 'N',
                             'prop_type': 'SF',
                             'valuation_method': 1.0,
                             'state': 'TX'}]}
            ]
        }
    }

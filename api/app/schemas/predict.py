from typing import Any, List, Optional

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

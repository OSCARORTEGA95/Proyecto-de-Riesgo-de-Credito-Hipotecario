from typing import Union

import numpy as np
import pandas as pd

from modelo_riesgo import __version__ as _version
from modelo_riesgo.pipeline import corregir
from modelo_riesgo.processing.data_manager import load_pipeline, nombre_modelo
from modelo_riesgo.processing.validation import validate_inputs

_modelo = None


def _cargar():
    global _modelo
    if _modelo is None:
        _modelo = load_pipeline(file_name=nombre_modelo())
    return _modelo


def make_prediction(*, input_data: Union[pd.DataFrame, dict, list]) -> dict:
    """Probabilidad de incumplir (90 días o más) en los primeros 12 meses, una por solicitud."""
    datos = pd.DataFrame([input_data] if isinstance(input_data, dict) else input_data)
    validados, errors = validate_inputs(input_data=datos)
    results = {"predictions": None, "version": _version, "errors": errors}
    if not errors:
        modelo = _cargar()
        q = modelo["pipeline"].predict_proba(validados)[:, 1]
        results["predictions"] = [float(p) for p in np.round(corregir(q, modelo["r"]), 6)]
    return results

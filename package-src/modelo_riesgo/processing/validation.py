from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from modelo_riesgo.config.core import config


class DataInputSchema(BaseModel):
    """Una solicitud de crédito: las variables disponibles al momento de originar."""
    credit_score: Optional[float] = None
    dti: Optional[float] = None
    oltv: Optional[float] = None
    ocltv: Optional[float] = None
    orig_rate: Optional[float] = None
    orig_upb: Optional[float] = None
    orig_term: Optional[float] = None
    mi_pct: Optional[float] = None
    num_borrowers: Optional[float] = None
    num_units: Optional[float] = None
    purpose: Optional[str] = None
    occupancy: Optional[str] = None
    channel: Optional[str] = None
    first_time_homebuyer: Optional[str] = None
    prop_type: Optional[str] = None
    valuation_method: Optional[str] = None
    state: Optional[str] = None


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Revisa que estén las variables del modelo y que cada valor tenga el tipo correcto.

    Las variables que el modelo no usa pueden faltar; si falta una que sí usa, se
    devuelve un error con su nombre.
    """
    faltantes = [c for c in config.variables if c not in input_data.columns]
    if faltantes:
        return input_data, {"faltan_variables": faltantes}
    datos = input_data[config.variables].copy()
    try:
        registros = datos.astype(object).replace({np.nan: None}).to_dict(orient="records")
        MultipleDataInputs(inputs=registros)
    except ValidationError as error:
        return datos, {"validacion": error.errors(include_url=False, include_context=False)}
    return datos, None

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger
from modelo_riesgo import __version__ as model_version
from modelo_riesgo.p2 import describir_p2, make_prediction_p2
from modelo_riesgo.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


# Ruta para verificar que la API se esté ejecutando correctamente
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """Nombre de la API, su versión, la del paquete de modelos y el modelo de cada pregunta."""
    try:
        p2 = describir_p2()
    except FileNotFoundError as error:
        p2 = {"error": str(error)}
    health = schemas.Health(name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version,
                            model_p1="red neuronal densa (red-p1, versión 5, 9 campos del formulario)", model_p2=p2)
    return health.model_dump()


# Ruta para realizar las predicciones
@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """Probabilidad de incumplir (90 días de mora o más) en los primeros 12 meses, una por solicitud.

    Solo se envían las variables escritas en la solicitud: si se borra una que el modelo
    usa, la API responde 400 con el nombre de la variable que falta.
    """
    filas = [solicitud.model_dump(exclude_unset=True) for solicitud in input_data.inputs]
    input_df = pd.DataFrame(filas)

    logger.info(f"Making prediction on inputs: {filas}")
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(json.dumps(results["errors"], default=str)))

    logger.info(f"Prediction results: {results.get('predictions')}")
    return results


# Ruta para las predicciones de la pregunta 2
@api_router.post("/predict_p2", response_model=schemas.PredictionResults, status_code=200)
async def predict_p2(input_data: schemas.MultipleDataInputsP2) -> Any:
    """Probabilidad de que un crédito activo y al día llegue a 90 días de mora o más en los próximos 3 meses.

    Recibe una fila por crédito con las 31 variables del modelo (historial de los últimos 12 meses,
    banderas del mes y datos de la solicitud original). Los valores pueden ser nulos, pero si falta
    una variable que el modelo usa, la API responde 400 con su nombre.
    """
    filas = [credito.model_dump(exclude_unset=True) for credito in input_data.inputs]
    input_df = pd.DataFrame(filas)

    logger.info(f"Making P2 prediction on {len(filas)} loans")
    results = make_prediction_p2(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"P2 prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(json.dumps(results["errors"], default=str)))

    logger.info(f"P2 prediction results: {len(results.get('predictions'))} probabilities")
    return results

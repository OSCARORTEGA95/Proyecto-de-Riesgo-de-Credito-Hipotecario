import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger
from modelo_riesgo import __version__ as model_version
from modelo_riesgo.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


# Ruta para verificar que la API se esté ejecutando correctamente
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """Nombre de la API, su versión y la versión del modelo empaquetado."""
    health = schemas.Health(name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version)
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

from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd

from modelo_riesgo import __version__ as _version
from modelo_riesgo.config.core import ROOT, TRAINED_MODEL_DIR, config


def ruta_datos() -> Path:
    ruta = Path(config.ruta_datos)
    return ruta if ruta.is_absolute() else (ROOT / ruta).resolve()


def load_dataset(trimestres: List[str]) -> pd.DataFrame:
    """Lee los trimestres pedidos de la tabla de la pregunta 1 (Parquet particionado por trimestre).

    Solo se usa al entrenar, por eso pyarrow no es una dependencia del paquete.
    """
    columnas = ["loan_id"] + config.variables + [config.etiqueta]
    df = pd.read_parquet(ruta_datos(), columns=columnas, filters=[("trimestre", "in", list(trimestres))])
    df = df[df[config.etiqueta].notna()].copy()
    df[config.etiqueta] = df[config.etiqueta].astype(int)
    return df.sort_values("loan_id", kind="stable").reset_index(drop=True)


def muestrear(df: pd.DataFrame, semilla: int) -> Tuple[pd.DataFrame, float]:
    """Todos los positivos y proporcion_negativos negativos por cada uno; devuelve la muestra y r."""
    y = config.etiqueta
    pos, neg = df[df[y] == 1], df[df[y] == 0]
    n_neg = min(len(neg), len(pos) * config.proporcion_negativos)
    muestra_neg = neg.sample(n=n_neg, random_state=semilla)
    r = len(muestra_neg) / len(neg)  # fracción de negativos conservada (los positivos se conservan todos)
    return pd.concat([pos, muestra_neg]).sort_index(), r


def nombre_modelo() -> str:
    return f"{config.pipeline_save_file}{_version}.pkl"


def save_pipeline(*, modelo: dict) -> Path:
    """Guarda el pipeline con su corrección y metadatos, y borra las versiones anteriores."""
    ruta = TRAINED_MODEL_DIR / nombre_modelo()
    for archivo in TRAINED_MODEL_DIR.iterdir():
        if archivo.name not in ("__init__.py", ruta.name):
            archivo.unlink()
    joblib.dump(modelo, ruta, compress=3)
    return ruta


def load_pipeline(*, file_name: str) -> dict:
    return joblib.load(TRAINED_MODEL_DIR / file_name)

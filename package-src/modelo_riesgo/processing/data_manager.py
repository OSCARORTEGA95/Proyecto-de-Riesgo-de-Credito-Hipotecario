"""Lectura de datos para entrenar y guardado del modelo.

La tabla de la pregunta 1 tiene millones de créditos y no cabe completa en la memoria de
una t3.medium junto a MLflow, así que se recorre por lotes con pyarrow y solo se guarda lo
necesario: la muestra de entrenamiento (todos los positivos y una fracción de negativos),
una muestra de validación, y en la prueba solo la etiqueta y la probabilidad de cada lote.
Las fracciones se eligen con un hash del loan_id, así que la muestra es siempre la misma.
pyarrow solo se usa al entrenar, por eso no es una dependencia del paquete.
"""
from pathlib import Path
from typing import Iterator, List, Tuple

import joblib
import numpy as np
import pandas as pd

from modelo_riesgo import __version__ as _version
from modelo_riesgo.config.core import ROOT, TRAINED_MODEL_DIR, config

TAM_LOTE = 200_000


def ruta_datos() -> Path:
    ruta = Path(config.ruta_datos)
    return ruta if ruta.is_absolute() else (ROOT / ruta).resolve()


def _dataset():
    import pyarrow.dataset as ds
    return ds.dataset(ruta_datos(), format="parquet", partitioning="hive")


def _filtro(trimestres: List[str]):
    import pyarrow.dataset as ds
    return ds.field("trimestre").isin(list(trimestres)) & ds.field(config.etiqueta).is_valid()


def contar(trimestres: List[str]) -> Tuple[int, int]:
    """Positivos y negativos con etiqueta en los trimestres, leyendo solo la etiqueta."""
    y = _dataset().to_table(columns=[config.etiqueta], filter=_filtro(trimestres)).column(config.etiqueta)
    y = y.to_numpy(zero_copy_only=False)
    return int((y == 1).sum()), int((y == 0).sum())


def _azar(ids) -> np.ndarray:
    """Número en [0, 1) fijo para cada loan_id."""
    return pd.util.hash_array(np.asarray(ids, dtype=object)).astype(np.float64) / 2.0 ** 64


def recorrer(trimestres: List[str], fraccion_negativos: float = 1.0,
             fraccion_positivos: float = 1.0) -> Iterator[pd.DataFrame]:
    """Lotes de la tabla con las variables del modelo y la etiqueta, opcionalmente submuestreados."""
    import pyarrow as pa
    y = config.etiqueta
    columnas = ["loan_id"] + config.variables + [y]
    for lote in _dataset().to_batches(columns=columnas, filter=_filtro(trimestres), batch_size=TAM_LOTE):
        if fraccion_negativos < 1.0 or fraccion_positivos < 1.0:
            etiqueta = lote.column(y).to_numpy(zero_copy_only=False)
            azar = _azar(lote.column("loan_id").to_pylist())
            umbral = np.where(etiqueta == 1, fraccion_positivos, fraccion_negativos)
            conservar = azar < umbral
            if not conservar.any():
                continue
            lote = lote.filter(pa.array(conservar))
        df = lote.to_pandas()
        df[y] = df[y].astype(int)
        yield df


def _juntar(lotes) -> pd.DataFrame:
    partes = list(lotes)
    columnas = ["loan_id"] + config.variables + [config.etiqueta]
    df = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=columnas)
    return df.sort_values("loan_id", kind="stable").reset_index(drop=True)


def load_muestra(trimestres: List[str], semilla: int) -> Tuple[pd.DataFrame, float]:
    """Todos los positivos y proporcion_negativos negativos por positivo; devuelve la muestra y r.

    r es la fracción de negativos conservada sobre el total real, para corregir la probabilidad.
    """
    positivos, negativos = contar(trimestres)
    objetivo = positivos * config.proporcion_negativos
    fraccion = min(1.0, 1.2 * objetivo / max(negativos, 1))  # un poco de más y luego se recorta
    df = _juntar(recorrer(trimestres, fraccion_negativos=fraccion))
    y = config.etiqueta
    pos, neg = df[df[y] == 1], df[df[y] == 0]
    neg = neg.sample(n=min(len(neg), objetivo), random_state=semilla)
    r = len(neg) / negativos
    return pd.concat([pos, neg]).sort_index(), r


def load_validacion(trimestres: List[str]) -> pd.DataFrame:
    """Muestra aleatoria de hasta max_validacion créditos, sin cargar la tabla completa."""
    positivos, negativos = contar(trimestres)
    fraccion = min(1.0, config.max_validacion / max(positivos + negativos, 1))
    return _juntar(recorrer(trimestres, fraccion_negativos=fraccion, fraccion_positivos=fraccion))


def nombre_modelo() -> str:
    """Archivo de la red de la pregunta 1. Si se subió la versión del paquete sin reentrenar la red
    (por ejemplo, para agregar la pregunta 2), se usa el último archivo entrenado que exista."""
    nombre = f"{config.pipeline_save_file}{_version}.pkl"
    if not (TRAINED_MODEL_DIR / nombre).exists():
        existentes = sorted(TRAINED_MODEL_DIR.glob(f"{config.pipeline_save_file}*.pkl"))
        if existentes:
            return existentes[-1].name
    return nombre


def save_pipeline(*, modelo: dict) -> Path:
    """Guarda el pipeline con su corrección y metadatos, y borra las versiones anteriores de la red.

    Solo borra archivos de la red de la pregunta 1: el modelo de la pregunta 2 se conserva.
    """
    ruta = TRAINED_MODEL_DIR / f"{config.pipeline_save_file}{_version}.pkl"
    for archivo in TRAINED_MODEL_DIR.glob(f"{config.pipeline_save_file}*.pkl"):
        if archivo.name != ruta.name:
            archivo.unlink()
    joblib.dump(modelo, ruta, compress=3)
    return ruta


def load_pipeline(*, file_name: str) -> dict:
    return joblib.load(TRAINED_MODEL_DIR / file_name)

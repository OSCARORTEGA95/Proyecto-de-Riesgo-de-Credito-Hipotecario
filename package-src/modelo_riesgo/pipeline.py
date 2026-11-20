"""Pipeline de scikit-learn: tipos, imputación, escalado, variables indicadoras y red neuronal.

Recibe el DataFrame tal como llega a la API (una fila por solicitud) y devuelve la
probabilidad de la muestra de entrenamiento; predict.py la corrige al muestreo.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from modelo_riesgo.config.core import config
from modelo_riesgo.red import RedNeuronal

SIN_DATO = "Sin dato"


def tipar(df: pd.DataFrame, numericas, categoricas) -> pd.DataFrame:
    """Numéricas a decimal (vacío a nulo) y categóricas a texto (nulo a 'Sin dato')."""
    salida = pd.DataFrame(index=df.index)
    for c in numericas:
        salida[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in categoricas:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce").round().astype("Int64")
        salida[c] = s.astype("string").str.strip().replace("", pd.NA).fillna(SIN_DATO).astype(object)
    return salida


def crear_pipeline(epocas: int, paciencia=None) -> Pipeline:
    numericas, categoricas = config.variables_numericas, config.variables_categoricas
    r = config.red
    preparacion = ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
    ])
    return Pipeline([
        ("tipos", FunctionTransformer(tipar, kw_args={"numericas": numericas, "categoricas": categoricas})),
        ("preparacion", preparacion),
        ("float32", FunctionTransformer(np.float32)),
        ("red", RedNeuronal(capas=tuple(r.capas), dropout=r.dropout, epocas=epocas, tam_lote=r.tam_lote,
                            tasa_aprendizaje=r.tasa_aprendizaje, l2=r.l2, paciencia=paciencia,
                            semilla=config.semilla, verbose=True)),
    ])


def corregir(q, r: float):
    """Deshace el submuestreo de negativos: p = r q / (r q + 1 - q)."""
    q = np.asarray(q, dtype="float64")
    return r * q / (r * q + 1.0 - q)

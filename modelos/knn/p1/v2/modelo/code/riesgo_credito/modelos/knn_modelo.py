"""Preparación de variables y modelo KNN empaquetado para MLflow (lo que se necesita al predecir)."""
import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd

SIN_DATO = "Sin dato"
MESES_SIN_ATRASO = 120
LIMITE_DIF_TASA = 5.0


def a_numero(s):
    try:
        return s.astype("float64")
    except (TypeError, ValueError):
        return pd.to_numeric(s, errors="coerce").astype("float64")


def a_bandera(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype("float64")
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) > 0).astype("float64")
    t = s.astype("string").str.strip().str.upper()
    return (t.notna() & ~t.isin(["", "N", "0", "FALSE"])).fillna(False).astype("float64")


def a_categoria(s):
    if pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce").round().astype("Int64")
    return s.astype("string").fillna(SIN_DATO).astype(object)


def preparar(df, numericas, banderas, categoricas):
    salida = pd.DataFrame(index=df.index)
    for c in numericas:
        v = a_numero(df[c])
        if c == "meses_desde_atraso":
            v = v.fillna(MESES_SIN_ATRASO)
        if c == "dif_tasa":
            v = v.clip(-LIMITE_DIF_TASA, LIMITE_DIF_TASA)
        salida[c] = v
    for c in banderas:
        salida[c] = a_bandera(df[c])
    for c in categoricas:
        salida[c] = a_categoria(df[c])
    return salida


def corregir(q, r):
    q = np.asarray(q, dtype="float64")
    return r * q / (r * q + 1.0 - q)


def probabilidad(pipeline, X, r, lote=20000):
    q = np.concatenate([pipeline.predict_proba(X.iloc[i:i + lote])[:, 1] for i in range(0, len(X), lote)])
    return corregir(q, r)


class ModeloKNN(mlflow.pyfunc.PythonModel):
    def __init__(self, numericas, banderas, categoricas, r):
        self.numericas = list(numericas)
        self.banderas = list(banderas)
        self.categoricas = list(categoricas)
        self.r = float(r)

    def __getstate__(self):
        estado = self.__dict__.copy()
        estado.pop("pipeline", None)  # el pipeline se guarda aparte, comprimido
        return estado

    def load_context(self, context):
        self.pipeline = joblib.load(context.artifacts["pipeline"])

    def predict(self, context, model_input: pd.DataFrame, params=None):
        X = preparar(pd.DataFrame(model_input), self.numericas, self.banderas, self.categoricas)
        return probabilidad(self.pipeline, X, self.r)

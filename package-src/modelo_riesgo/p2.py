"""Pregunta 2: probabilidad de que un crédito activo y al día llegue a 90 días de mora o más
en los próximos 3 meses.

El modelo es el k vecinos de la versión 1 del experimento knn-p2 (el mejor en validación).
Recibe una fila por crédito y mes de corte con 31 variables: 8 del historial de pagos de los
últimos 12 meses, 6 banderas del mes (diferimientos, modificaciones, mora por desastre) y las
17 de la solicitud original salvo el estado. La preparación (tipos, meses sin atraso, límite a la
diferencia de tasa) es la misma que se usó al entrenar en src/riesgo_credito/modelos/knn_modelo.py.
"""
from typing import Any, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, ValidationError

from modelo_riesgo import __version__ as _version
from modelo_riesgo.config.core import TRAINED_MODEL_DIR, config

SIN_DATO = "Sin dato"
MESES_SIN_ATRASO = 120   # un crédito que nunca se ha atrasado: "hace 120 meses"
LIMITE_DIF_TASA = 5.0    # la diferencia de tasa se limita a ±5 puntos

Numero = Optional[float]
Bandera = Optional[Union[bool, float, str]]
Categoria = Optional[Union[str, float]]


class DataInputSchemaP2(BaseModel):
    """Un crédito activo y al día en un mes de corte."""
    loan_id: Optional[str] = None
    period: Optional[int] = None
    # Historial de pagos (últimos 12 meses)
    edad: Numero = None
    pct_saldo: Numero = None
    dif_tasa: Numero = None
    eltv: Numero = None
    atrasos_12m: Numero = None
    max_atraso_12m: Numero = None
    meses_desde_atraso: Numero = None
    indulgencia_12m: Numero = None
    # Banderas del mes
    con_atraso_previo: Bandera = None
    diferimiento: Bandera = None
    diferimiento_mes: Bandera = None
    saldo_sin_interes: Bandera = None
    mora_desastre: Bandera = None
    modificado: Bandera = None
    plan_asistencia: Categoria = None
    # Solicitud original
    credit_score: Numero = None
    dti: Numero = None
    oltv: Numero = None
    ocltv: Numero = None
    orig_rate: Numero = None
    orig_upb: Numero = None
    orig_term: Numero = None
    mi_pct: Numero = None
    num_borrowers: Numero = None
    num_units: Numero = None
    purpose: Categoria = None
    occupancy: Categoria = None
    channel: Categoria = None
    first_time_homebuyer: Categoria = None
    prop_type: Categoria = None
    valuation_method: Categoria = None
    state: Optional[str] = None  # el modelo no lo usa; el tablero lo muestra


class MultipleDataInputsP2(BaseModel):
    inputs: List[DataInputSchemaP2]


# ---------- preparación (igual a la del entrenamiento) ----------

def _numero(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _bandera(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype("float64")
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) > 0).astype("float64")
    t = s.astype("string").str.strip().str.upper()
    return (t.notna() & ~t.isin(["", "N", "0", "0.0", "FALSE"])).fillna(False).astype("float64")


def _categoria(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        s = pd.to_numeric(s, errors="coerce").round().astype("Int64")
    else:
        # un código numérico que llega como texto ("2.0") se lleva a "2", como al entrenar
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() == s.notna().sum() and s.notna().any():
            s = num.round().astype("Int64")
    return s.astype("string").fillna(SIN_DATO).astype(object)


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    c = config.p2
    salida = pd.DataFrame(index=df.index)
    for col in c.variables_numericas:
        v = _numero(df[col])
        if col == "meses_desde_atraso":
            v = v.fillna(MESES_SIN_ATRASO)
        if col == "dif_tasa":
            v = v.clip(-LIMITE_DIF_TASA, LIMITE_DIF_TASA)
        salida[col] = v
    for col in c.variables_banderas:
        salida[col] = _bandera(df[col])
    for col in c.variables_categoricas:
        salida[col] = _categoria(df[col])
    return salida


def corregir(q, r: float) -> np.ndarray:
    """Deshace el submuestreo de negativos: p = r q / (r q + 1 - q)."""
    q = np.asarray(q, dtype="float64")
    return r * q / (r * q + 1.0 - q)


# ---------- validación y predicción ----------

def validate_inputs_p2(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Revisa que estén las 31 variables del modelo y que cada valor tenga un tipo válido.

    Los valores pueden ser nulos (el modelo imputa), pero la columna debe venir: si falta una
    que el modelo usa, se devuelve su nombre.
    """
    faltantes = [col for col in config.p2.variables if col not in input_data.columns]
    if faltantes:
        return input_data, {"faltan_variables": faltantes}
    datos = input_data[config.p2.variables].copy()
    try:
        registros = datos.astype(object).replace({np.nan: None}).to_dict(orient="records")
        MultipleDataInputsP2(inputs=registros)
    except ValidationError as error:
        return datos, {"validacion": error.errors(include_url=False, include_context=False)}
    return datos, None


_modelo_p2: Optional[dict] = None


def cargar_modelo_p2() -> dict:
    global _modelo_p2
    if _modelo_p2 is None:
        ruta = TRAINED_MODEL_DIR / config.p2.archivo
        if not ruta.exists():
            raise FileNotFoundError(f"No está {ruta.name}: corre 'tox run -e p2' en package-src antes de construir el .whl")
        _modelo_p2 = joblib.load(ruta)
    return _modelo_p2


def describir_p2() -> dict:
    """Versión y procedencia del modelo de la pregunta 2, para la ruta de salud de la API."""
    m = cargar_modelo_p2()
    return {"descripcion": m["descripcion"], "experimento": m["experimento"], "run_id": m["run_id"],
            "k": m["k"], "auc_validacion": m["metricas"].get("auc_validacion"),
            "auc_prueba": m["metricas"].get("auc_roc")}


def make_prediction_p2(*, input_data: Union[pd.DataFrame, dict, list]) -> dict:
    """Probabilidad de llegar a 90 días de mora o más en los próximos 3 meses, una por crédito."""
    datos = pd.DataFrame([input_data] if isinstance(input_data, dict) else input_data)
    validados, errors = validate_inputs_p2(input_data=datos)
    results: dict[str, Any] = {"predictions": None, "version": _version, "errors": errors}
    if not errors:
        m = cargar_modelo_p2()
        X = preparar(validados)
        # working_memory acota la matriz de distancias de k vecinos (la máquina tiene 4 GB)
        with sklearn.config_context(working_memory=256):
            q = np.concatenate([m["pipeline"].predict_proba(X.iloc[i:i + 5000])[:, 1]
                                for i in range(0, len(X), 5000)])
        results["predictions"] = [float(p) for p in np.round(corregir(q, m["r"]), 8)]
    return results

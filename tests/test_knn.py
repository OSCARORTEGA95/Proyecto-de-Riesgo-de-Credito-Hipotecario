import numpy as np
import pandas as pd

from riesgo_credito.modelos.knn import metricas, muestrear
from riesgo_credito.modelos.knn_modelo import corregir, preparar


def test_correccion_de_probabilidad():
    q = np.array([0.0, 0.1, 0.5, 1.0])
    assert np.allclose(corregir(q, 1.0), q)
    assert np.isclose(corregir([0.1], 0.05)[0], 0.005 / 0.905)
    assert np.all(np.diff(corregir(q, 0.05)) > 0)


def test_muestreo_conserva_positivos_y_calcula_r():
    df = pd.DataFrame({"y": [1] * 10 + [0] * 990})
    m, info = muestrear(df, "y", proporcion=10, maximo=None, semilla=42)
    assert info["positivos"] == 10 and info["negativos"] == 100
    assert np.isclose(info["r"], 100 / 990)


def test_captura_del_decil_superior():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2])
    m = metricas(y, p)
    assert m["captura_decil"] == 0.5 and m["precision_decil"] == 1.0


def test_preparar_maneja_nulos_y_tipos():
    df = pd.DataFrame({"credit_score": [700, None], "valuation_method": [2.0, None],
                       "mora_desastre": ["Y", None], "meses_desde_atraso": [None, 3]})
    X = preparar(df, ["credit_score", "meses_desde_atraso"], ["mora_desastre"], ["valuation_method"])
    assert X["valuation_method"].tolist() == ["2", "Sin dato"]
    assert X["mora_desastre"].tolist() == [1.0, 0.0]
    assert X["meses_desde_atraso"].tolist() == [120.0, 3.0]
    assert np.isnan(X["credit_score"].iloc[1])

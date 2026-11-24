import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from riesgo_credito.modelos.rnn_modelo import (RedRecurrente, SecuenciaDeColumnas, SecuenciaMensual,
                                               construir_ancho, nombre_mes)


def secuencias(n=4000, T=8, semilla=0):
    """La etiqueta depende de si la variable subió en los últimos meses: un patrón en el tiempo."""
    rng = np.random.default_rng(semilla)
    x = rng.standard_normal((n, T, 2)).astype(np.float32)
    tendencia = x[:, -3:, 0].mean(axis=1) - x[:, :3, 0].mean(axis=1)
    y = (rng.random(n) < 1 / (1 + np.exp(-(3 * tendencia - 1)))).astype(int)
    mascara = np.ones((n, T), dtype=np.float32)
    mascara[: n // 4, :2] = 0  # una cuarta parte sin los dos meses más antiguos
    x[mascara == 0] = 0
    return np.concatenate([x.reshape(n, -1), mascara], axis=1), y


def test_aprende_patron_temporal_y_conserva_mejor_epoca():
    X, y = secuencias()
    Xv, yv = secuencias(1500, semilla=1)
    red = RedRecurrente(n_pasos=8, n_entrada=2, unidades=16, densas=(8,), epocas=12, paciencia=4).fit(X, y, Xv, yv)
    auc = roc_auc_score(yv, red.predict_proba(Xv)[:, 1])
    assert auc > 0.75
    assert red.mejor_auc_validacion_ == pytest.approx(auc, abs=1e-6)
    assert {"perdida", "perdida_validacion", "auc_validacion"} <= set(red.historial_[0])


def test_con_estaticas_dropout_y_serializable():
    X, y = secuencias(1500)
    estatica = np.random.default_rng(3).standard_normal((len(X), 3)).astype(np.float32)
    X = np.concatenate([X, estatica], axis=1)
    red = RedRecurrente(n_pasos=8, n_entrada=2, unidades=8, densas=(8, 4), dropout=0.3, epocas=2).fit(X, y)
    assert red.n_estaticas_ == 3
    p = red.predict_proba(X[:30])
    assert np.allclose(p, red.predict_proba(X[:30]))
    assert np.allclose(p, pickle.loads(pickle.dumps(red)).predict_proba(X[:30]))


def test_mes_ausente_no_cambia_el_estado():
    """Con máscara 0, el valor del paso no debe influir en la predicción."""
    X, y = secuencias(600)
    red = RedRecurrente(n_pasos=8, n_entrada=2, unidades=6, densas=(4,), epocas=1).fit(X, y)
    fila = X[:1].copy()
    fila[0, 8 * 2 + 1] = 0.0  # máscara del paso 2 en 0
    otra = fila.copy()
    otra[0, 2:4] = 99.0  # valores del paso 2 cambiados
    assert np.allclose(red.predict_proba(fila), red.predict_proba(otra))


def test_secuencia_de_columnas():
    X = np.array([[1.0, 2.0, 3.0]])
    s = SecuenciaDeColumnas().fit(X).transform(X)
    assert s.shape == (1, 3 * 4 + 3)
    pasos = s[0, :12].reshape(3, 4)
    assert np.allclose(pasos[:, 0], [1, 2, 3]) and np.allclose(pasos[:, 1:], np.eye(3))
    assert np.allclose(s[0, 12:], 1)


def test_construir_ancho_y_secuencia_mensual():
    historia = pd.DataFrame({"loan_id": ["A"] * 4 + ["B"] * 2,
                             "period": [202301, 202302, 202304, 202305, 202304, 202305],
                             "x": [1.0, 2.0, 4.0, 5.0, 10.0, np.nan],
                             "meses_desde_atraso": [np.nan, 1, 2, 3, np.nan, np.nan]})
    objetivos = pd.DataFrame({"loan_id": ["A", "B"], "period": [202305, 202305], "y": [1, 0]})
    ancho = construir_ancho(objetivos, historia, ["x", "meses_desde_atraso"], n_pasos=3)
    a = ancho.iloc[0]
    assert a[nombre_mes("x", 0)] == 5.0 and a[nombre_mes("x", 1)] == 4.0
    assert not a[nombre_mes("presente", 2)]  # 202303 no existe para A
    b = ancho.iloc[1]
    assert b[nombre_mes("presente", 0)] and np.isnan(b[nombre_mes("x", 0)])
    t = SecuenciaMensual(["x", "meses_desde_atraso"], n_pasos=3).fit(ancho)
    salida = t.transform(ancho)
    assert salida.shape == (2, 3 * 2 + 3)
    assert np.allclose(salida[:, 6:], [[0, 1, 1], [0, 1, 1]])  # máscara: paso 0 = mes más antiguo
    assert np.allclose(salida[:, :2], 0)  # mes ausente queda en cero
    assert not np.isnan(salida).any()

import pickle

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from riesgo_credito.modelos.red_modelo import RedNeuronal


def datos(n=6000, semilla=0):
    rng = np.random.default_rng(semilla)
    X = rng.standard_normal((n, 8)).astype(np.float32)
    logit = 1.5 * X[:, 0] - X[:, 1] + X[:, 2] * X[:, 3] - 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


@pytest.mark.parametrize("capas,dropout", [((16, 8), 0.0), ((128, 128, 64, 64, 32, 32, 16), 0.3)])
def test_aprende_y_registra_epocas(capas, dropout):
    X, y = datos()
    Xv, yv = datos(2000, 1)
    red = RedNeuronal(capas=capas, dropout=dropout, epocas=15, tam_lote=256, paciencia=5).fit(X, y, Xv, yv)
    assert roc_auc_score(yv, red.predict_proba(Xv)[:, 1]) > 0.8
    assert len(red.pesos_) == len(capas) + 1
    assert 1 <= len(red.historial_) <= 15
    assert {"epoca", "perdida", "auc_validacion", "perdida_validacion"} <= set(red.historial_[0])
    # conserva los pesos de la mejor época
    assert red.mejor_auc_validacion_ == pytest.approx(roc_auc_score(yv, red.predict_proba(Xv)[:, 1]), abs=1e-6)


def test_prediccion_determinista_y_serializable():
    X, y = datos(2000)
    red = RedNeuronal(capas=(8, 8), dropout=0.5, epocas=3).fit(X, y)
    p = red.predict_proba(X[:50])
    assert np.allclose(p, red.predict_proba(X[:50]))  # sin dropout al predecir
    assert np.allclose(p.sum(axis=1), 1)
    copia = pickle.loads(pickle.dumps(red))
    assert np.allclose(p, copia.predict_proba(X[:50]))
    assert red.mejor_epoca_ == 3 and red.mejor_auc_validacion_ is None


def test_en_pipeline_de_sklearn():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = datos(2000)
    pipe = Pipeline([("escalar", StandardScaler()), ("red", RedNeuronal(capas=(8,), epocas=2))]).fit(X, y)
    assert pipe.predict_proba(X).shape == (2000, 2)

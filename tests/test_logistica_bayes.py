import numpy as np
import pandas as pd

from riesgo_credito.modelos.knn_modelo import corregir, preparar
from riesgo_credito.modelos.logistica_bayes import crear_clasificador, crear_preparacion


def datos(n=600, semilla=0):
    g = np.random.default_rng(semilla)
    df = pd.DataFrame({"credit_score": g.integers(600, 820, n).astype(float), "dti": g.integers(10, 50, n).astype(float),
                       "mora_desastre": g.choice(["Y", None], n), "purpose": g.choice(["P", "C", "N"], n)})
    df.loc[:5, "credit_score"] = np.nan
    y = (g.random(n) < 1 / (1 + np.exp((df["credit_score"].fillna(700) - 650) / 30))).astype(int)
    return df, y


def test_tramos_produce_columnas_binarias():
    df, _ = datos()
    X = crear_preparacion("tramos", ["credit_score", "dti"], ["mora_desastre"], ["purpose"], 5).fit_transform(
        preparar(df, ["credit_score", "dti"], ["mora_desastre"], ["purpose"]))
    assert set(np.unique(X)) <= {0.0, 1.0}
    assert X.shape[1] == 5 + 5 + 1 + 3


def test_los_tres_clasificadores_ordenan_el_riesgo():
    df, y = datos()
    X = preparar(df, ["credit_score", "dti"], [], ["purpose"])
    for tipo, prep, valor in [("logistica", "estandar", 1.0), ("gaussiano", "estandar", 1e-9), ("bernoulli", "tramos", 1.0)]:
        p = crear_preparacion(prep, ["credit_score", "dti"], [], ["purpose"], 5)
        modelo = p.fit(X).transform(X)
        q = crear_clasificador(tipo, valor).fit(modelo, y).predict_proba(modelo)[:, 1]
        assert np.corrcoef(q, -X["credit_score"].fillna(700))[0, 1] > 0
        assert np.all(np.diff(corregir(np.sort(q), 0.1)) >= 0)

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from modelo_riesgo.config.core import config
from modelo_riesgo.p2 import make_prediction_p2


@pytest.fixture()
def creditos():
    """Cinco créditos al día de la muestra de prueba de 2024 (ejemplo guardado por MLflow en knn-p2)."""
    return pd.DataFrame(json.loads((Path(__file__).parent / "datos_prueba_p2.json").read_text(encoding="utf-8")))


def test_prediccion_p2(creditos):
    resultado = make_prediction_p2(input_data=creditos)
    p = resultado["predictions"]
    assert resultado["errors"] is None
    assert len(p) == len(creditos)
    assert all(isinstance(x, float) and 0.0 <= x <= 1.0 and not math.isnan(x) for x in p)


def test_atrasos_recientes_suben_el_riesgo(creditos):
    """El mismo crédito con dos atrasos en el último año y uno reciente es más riesgoso."""
    base = creditos.iloc[[0]].copy()
    atrasado = base.assign(atrasos_12m=2.0, max_atraso_12m=2.0, meses_desde_atraso=1.0, con_atraso_previo=True)
    p = make_prediction_p2(input_data=pd.concat([base, atrasado], ignore_index=True))["predictions"]
    assert p[1] > p[0]


def test_falta_variable_p2(creditos):
    usada = config.p2.variables[0]
    resultado = make_prediction_p2(input_data=creditos.drop(columns=[usada]))
    assert resultado["predictions"] is None
    assert usada in resultado["errors"]["faltan_variables"]


def test_tipo_invalido_p2(creditos):
    datos = creditos.astype(object)
    datos.loc[0, "edad"] = "muchos"
    resultado = make_prediction_p2(input_data=datos)
    assert resultado["predictions"] is None and "validacion" in resultado["errors"]


def test_un_credito_como_diccionario(creditos):
    fila = creditos.iloc[0].to_dict()
    resultado = make_prediction_p2(input_data=fila)
    assert resultado["errors"] is None and len(resultado["predictions"]) == 1

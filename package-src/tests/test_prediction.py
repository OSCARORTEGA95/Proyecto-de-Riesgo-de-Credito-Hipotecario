import math

from modelo_riesgo.config.core import config
from modelo_riesgo.predict import make_prediction


def test_make_prediction(sample_input_data):
    result = make_prediction(input_data=sample_input_data)

    predictions = result.get("predictions")
    assert result.get("errors") is None
    assert isinstance(predictions, list)
    assert len(predictions) == len(sample_input_data)
    assert all(isinstance(p, float) and 0.0 <= p <= 1.0 and not math.isnan(p) for p in predictions)


def test_mayor_riesgo_con_peor_perfil(sample_input_data):
    """La solicitud de puntaje 640, deuda-ingreso 48 y préstamo-valor 97 es más riesgosa que la de 780."""
    p = make_prediction(input_data=sample_input_data)["predictions"]
    assert p[1] > p[0]


def test_una_sola_solicitud_como_diccionario(sample_input_data):
    result = make_prediction(input_data=sample_input_data.iloc[0].to_dict())
    assert result["errors"] is None and len(result["predictions"]) == 1


def test_falta_variable_del_modelo(sample_input_data):
    usada = config.variables[0]
    result = make_prediction(input_data=sample_input_data.drop(columns=[usada]))
    assert result["predictions"] is None
    assert usada in result["errors"]["faltan_variables"]


def test_variable_no_usada_puede_faltar_o_sobrar(sample_input_data):
    datos = sample_input_data.assign(channel="R")  # el modelo no usa canal
    assert make_prediction(input_data=datos)["errors"] is None


def test_tipo_invalido(sample_input_data):
    datos = sample_input_data.astype(object)
    datos.loc[0, "dti"] = "alto"
    result = make_prediction(input_data=datos)
    assert result["predictions"] is None and "validacion" in result["errors"]

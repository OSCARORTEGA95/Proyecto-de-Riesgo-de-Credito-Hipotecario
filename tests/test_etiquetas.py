"""Prueba de las etiquetas con créditos sintéticos de resultado conocido (los mismos del EDA)."""

import pytest
from pyspark.sql import SparkSession

from riesgo_credito.procesamiento import etl


@pytest.fixture(scope="module")
def spark():
    s = SparkSession.builder.master("local[1]").config("spark.ui.showConsoleProgress", "false").getOrCreate()
    yield s
    s.stop()


def credito(loan_id, estados, salida=None, mes_inicial=24240):
    filas = []
    for edad, estado in enumerate(estados, start=1):
        zb = salida[1] if salida and edad == salida[0] else None
        filas.append((loan_id, mes_inicial + edad, edad, estado, int(estado) if estado.isdigit() else None, zb))
        if zb is not None:
            break
    return filas


AL_DIA = ["00"] * 15
SINTETICOS = (
    credito("SANO", AL_DIA)
    + credito("MORA_MES8", ["00"] * 4 + ["01", "02", "02", "03"] + ["04"] * 7)
    + credito("MORA_MES14", ["00"] * 10 + ["01", "02", "02", "03", "04"])
    + credito("PREPAGO_MES6", AL_DIA, salida=(6, 1))
    + credito("VENTA_CORTA_MES9", ["00"] * 6 + ["01", "02", "00"], salida=(9, 3))
    + credito("DEFECTO_MES5", AL_DIA, salida=(5, 96))
    + credito("ACTIVO_MES5", ["00"] * 5)
    + credito("ADJUDICADO_MES10", ["00"] * 9 + ["RA"])
)


@pytest.fixture(scope="module")
def resumen(spark):
    perf = spark.createDataFrame(
        SINTETICOS, "loan_id string, mes int, edad int, dlq_status string, dlq_num int, zero_balance_code int"
    )
    return etl.resumen_creditos(etl.marcar_incumplimiento(perf))


def test_etiqueta_p1(resumen):
    p1 = {r.loan_id: r.y_p1 for r in etl.etiqueta_p1(resumen, 12).collect()}
    assert p1 == {
        "SANO": 0,
        "MORA_MES8": 1,
        "MORA_MES14": 0,
        "PREPAGO_MES6": 0,
        "VENTA_CORTA_MES9": 1,
        "ACTIVO_MES5": None,      # ventana sin observar completa
        "ADJUDICADO_MES10": 1,
    }

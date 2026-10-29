"""Prueba de las etiquetas con créditos sintéticos de resultado conocido (los mismos del EDA)."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

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
    + credito("CURADO", ["00"] * 3 + ["01"] + ["00"] * 3)
)


@pytest.fixture(scope="module")
def perf(spark):
    df = spark.createDataFrame(
        SINTETICOS, "loan_id string, mes int, edad int, dlq_status string, dlq_num int, zero_balance_code int"
    )
    return etl.marcar_incumplimiento(df)


@pytest.fixture(scope="module")
def resumen(perf):
    return etl.resumen_creditos(perf)


@pytest.fixture(scope="module")
def p2(perf, resumen):
    perf = (perf.withColumn("period", F.col("mes")).withColumn("es_ra", F.col("dlq_status") == "RA")
            .withColumn("current_upb", F.lit(100.0)).withColumn("current_rate", F.lit(4.0))
            .withColumn("eltv", F.lit(70)).withColumn("assistance_plan", F.lit(None).cast("string"))
            .withColumn("payment_deferral", F.lit(None).cast("string")).withColumn("non_interest_upb", F.lit(0.0))
            .withColumn("dlq_disaster", F.lit(None).cast("string")).withColumn("modificado", F.lit(False))
            .withColumn("en_pandemia", F.lit(False)))
    orig = resumen.select("loan_id", "defecto", F.lit(200.0).alias("orig_upb"), F.lit(4.5).alias("orig_rate"))
    return etl.construir_p2(perf, orig, ventana=3, historial=12).toPandas()


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
        "CURADO": None,
    }


def positivos(p2, loan_id, etiqueta="y_p2", dlq=0):
    filas = p2[(p2.loan_id == loan_id) & (p2.dlq_num == dlq)]
    return sorted(filas.loc[filas[etiqueta] == 1, "edad"].tolist())


def test_etiqueta_p2(p2):
    assert positivos(p2, "MORA_MES8") == []                 # último mes al día a 4 meses del evento
    assert positivos(p2, "MORA_MES8", dlq=1) == [5]         # desde 30 días el evento está a 3 meses
    assert positivos(p2, "MORA_MES8", "y_p2_60") == [3, 4]  # 60 días llega en el mes 6
    assert positivos(p2, "VENTA_CORTA_MES9") == [6]         # último mes al día a 3 meses de la venta corta
    assert "DEFECTO_MES5" not in set(p2.loan_id)
    sano = p2[p2.loan_id == "SANO"]
    assert sano.loc[sano.y_p2.notna(), "edad"].max() == 12   # los últimos 3 meses no se alcanzan a observar
    assert p2.loc[p2.loan_id == "PREPAGO_MES6", "y_p2"].tolist() == [0] * 5


def test_variables_p2(p2):
    curado = p2[p2.loan_id == "CURADO"].set_index("edad")
    assert curado.loc[3, "atrasos_12m"] == 0 and not curado.loc[3, "con_atraso_previo"]
    assert curado.loc[5, "atrasos_12m"] == 1 and curado.loc[5, "meses_desde_atraso"] == 1
    assert curado.loc[7, "meses_desde_atraso"] == 3 and curado.loc[7, "max_atraso_12m"] == 1
    assert curado.loc[5, "pct_saldo"] == 0.5 and curado.loc[5, "dif_tasa"] == -0.5

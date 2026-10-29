"""Tablas agregadas para el tablero Mortgage Risk Analytics.

Guardan conteos y sumas, no tasas ni promedios, para que el tablero pueda
sumar cualquier combinación de filtros (estado, trimestre, propósito, rangos)
y dividir después. Por ejemplo, el puntaje promedio es
suma_puntaje / n_puntaje y el incumplimiento a 12 meses es
p1_positivos / p1_etiquetables.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from riesgo_credito.procesamiento.etl import VARIABLES_P1

# Rangos de la maqueta v3
RANGOS_PUNTAJE = [(620, "<620"), (680, "620-679"), (720, "680-719"), (760, "720-759"), (800, "760-799")]
RANGOS_LTV = [(60, "<=60"), (70, "61-70"), (80, "71-80"), (90, "81-90"), (95, "91-95")]


def _rango(col: str, cortes: list[tuple[int, str]], ultimo: str, inclusivo: bool):
    """Etiqueta de rango; con inclusivo=True el corte pertenece al rango de abajo (<=)."""
    expr = F.when(F.col(col).isNull(), F.lit("Sin dato"))
    for corte, etiqueta in cortes:
        expr = expr.when(F.col(col) <= corte if inclusivo else F.col(col) < corte, F.lit(etiqueta))
    return expr.otherwise(F.lit(ultimo))


def originacion(orig: DataFrame, p1: DataFrame) -> DataFrame:
    """Panorama y comparación de la evaluación: un grupo por estado, propósito y rangos."""
    return (
        orig.join(p1.select("loan_id", "y_p1"), "loan_id", "left")
        .withColumn("rango_puntaje", _rango("credit_score", RANGOS_PUNTAJE, ">=800", inclusivo=False))
        .withColumn("rango_ltv", _rango("oltv", RANGOS_LTV, ">95", inclusivo=True))
        .groupBy("state", "purpose", "rango_puntaje", "rango_ltv")
        .agg(
            F.count("*").alias("creditos"),
            F.sum("orig_upb").alias("monto_total"),
            F.sum("credit_score").alias("suma_puntaje"), F.count("credit_score").alias("n_puntaje"),
            F.sum("orig_rate").alias("suma_tasa"), F.count("orig_rate").alias("n_tasa"),
            F.sum("oltv").alias("suma_ltv"), F.count("oltv").alias("n_ltv"),
            F.sum("dti").alias("suma_dti"), F.count("dti").alias("n_dti"),
            F.count("y_p1").alias("p1_etiquetables"),
            F.coalesce(F.sum("y_p1"), F.lit(0)).alias("p1_positivos"),
        )
    )


def serie_mensual(perf: DataFrame, orig: DataFrame) -> DataFrame:
    """Mora y planes de asistencia de los créditos activos, por mes y estado."""
    def cuenta(condicion):
        return F.sum(F.coalesce(condicion, F.lit(False)).cast("int"))

    def saldo_si(condicion):
        return F.sum(F.when(F.coalesce(condicion, F.lit(False)), F.col("current_upb")).otherwise(0.0))

    mora_90 = (F.col("dlq_num") >= 3) | F.col("es_ra")
    return (
        perf.where(F.col("zero_balance_code").isNull())
        .join(orig.select("loan_id", "state"), "loan_id")
        .groupBy("period", "state")
        .agg(
            F.count("*").alias("activos"),
            F.sum("current_upb").alias("saldo"),
            cuenta(F.col("dlq_num") == 0).alias("al_dia"),
            cuenta((F.col("dlq_num") >= 1) | F.col("es_ra")).alias("mora_30"),
            cuenta((F.col("dlq_num") >= 2) | F.col("es_ra")).alias("mora_60"),
            cuenta(mora_90).alias("mora_90"),
            saldo_si(mora_90).alias("saldo_mora_90"),
            cuenta(F.col("assistance_plan") == "F").alias("indulgencia"),
            cuenta(F.col("assistance_plan") == "R").alias("plan_repago"),
            cuenta(F.col("assistance_plan") == "T").alias("periodo_prueba"),
            cuenta(F.col("dlq_disaster") == "Y").alias("mora_desastre"),
            cuenta(F.col("payment_deferral").isNotNull()).alias("diferimiento"),
        )
    )


def cartera_activa(p2: DataFrame, orig: DataFrame, fecha_corte: int) -> DataFrame:
    """Créditos al día en la fecha de corte, con todo lo que la vista Seguimiento envía a la API y muestra."""
    return (
        p2.where((F.col("period") == fecha_corte) & (F.col("dlq_num") == 0))
        .drop("y_p2", "y_p2_60", "grupo_muestra")
        .join(orig.select("loan_id", "first_payment_date", *VARIABLES_P1), "loan_id")
    )

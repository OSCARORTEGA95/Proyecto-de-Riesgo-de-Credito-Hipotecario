"""ETL de un trimestre del Single-Family Loan-Level Dataset de Freddie Mac con Spark.

Lee el zip trimestral de data/crudos, lo descomprime en una carpeta temporal y
escribe en Parquet:

- data/limpios/originacion/trimestre=AAAATn: un crédito por fila.
- data/limpios/desempeno/trimestre=AAAATn: un crédito por mes.
- data/modelado/p1/trimestre=AAAATn: variables de la solicitud y etiqueta de la pregunta 1.
- data/metricas/etl/AAAATn.json: conteos para validar la ejecución.

Uso, desde la raíz del repositorio:
    python -m riesgo_credito.procesamiento.etl --trimestre 2019Q1
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from riesgo_credito.procesamiento import esquema

# Columnas de originación sin información (vacías o constantes) o casi siempre no disponibles
DESCARTAR_ORIGINACION = ["ppm_flag", "amort_type", "io_flag", "vantage_score", "valuation_method"]

# Columnas de desempeño que no se usan: posteriores al cierre, constantes o reemplazadas por la edad calculada
DESCARTAR_DESEMPENO = esquema.POSTERIORES_AL_CIERRE + ["ddlpi", "step_mod_flag", "cramdown_costs", "loan_age"]

# Variables disponibles al momento de la solicitud, usadas por el modelo de aprobación y la API
VARIABLES_P1 = [
    "credit_score", "dti", "oltv", "ocltv", "orig_rate", "orig_upb", "orig_term", "mi_pct",
    "purpose", "occupancy", "channel", "first_time_homebuyer", "num_borrowers", "num_units",
    "prop_type", "state",
]

COLUMNAS_ORIGINACION = [c for c in esquema.ORIGINACION_NUEVO if c not in DESCARTAR_ORIGINACION]
COLUMNAS_DESEMPENO = [c for c in esquema.DESEMPENO_NUEVO if c not in DESCARTAR_DESEMPENO]


# --------------------------------------------------------------------------
# Lectura
# --------------------------------------------------------------------------

def crear_sesion(memoria_driver: str = "4g", particiones: int = 8) -> SparkSession:
    return (
        SparkSession.builder.master("local[*]")
        .appName("etl-riesgo-credito")
        .config("spark.driver.memory", memoria_driver)
        .config("spark.sql.shuffle.partitions", particiones)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def extraer_zip(ruta_zip: Path, destino: Path) -> tuple[Path, Path]:
    """Descomprime el zip trimestral y devuelve las rutas de originación y desempeño.

    Reconoce los nombres del formato nuevo (orig_, perf_) y del anterior
    (historical_data_, historical_data_time_).
    """
    with zipfile.ZipFile(ruta_zip) as z:
        nombres = [n for n in z.namelist() if n.endswith(".txt")]
        orig = [n for n in nombres if Path(n).name.startswith("orig_")
                or (Path(n).name.startswith("historical_data_") and "_time_" not in n)]
        perf = [n for n in nombres if Path(n).name.startswith("perf_") or "_time_" in n]
        if len(orig) != 1 or len(perf) != 1:
            raise ValueError(f"{ruta_zip}: no se encontró un archivo de originación y uno de desempeño ({nombres})")
        z.extract(orig[0], destino)
        z.extract(perf[0], destino)
    return destino / orig[0], destino / perf[0]


def _columna_tipada(col: str):
    """Misma conversión que esquema._expresion: no disponibles y vacíos a nulos, luego el tipo."""
    valor = F.trim(F.col(col))
    codigos = esquema.NO_DISPONIBLE.get(col)
    if codigos:
        valor = F.when(valor.isin(codigos), F.lit(None)).otherwise(valor)
    valor = F.when(valor == "", F.lit(None)).otherwise(valor)
    if col in esquema.ENTEROS:
        return valor.try_cast("int").alias(col)
    if col in esquema.DECIMALES:
        return valor.try_cast("double").alias(col)
    return valor.alias(col)


def leer_crudo(spark: SparkSession, ruta: Path, tipo: str, columnas: list[str]) -> DataFrame:
    """Lee un archivo crudo con nombres, tipos y nulos correctos.

    Las columnas pedidas que no existen en el formato del archivo salen como nulas.
    """
    _, todas = esquema.detectar_formato(ruta, tipo)
    schema = StructType([StructField(c, StringType()) for c in todas])
    df = (
        spark.read.option("sep", "|").option("header", "false").option("quote", "")
        .option("mode", "FAILFAST").option("encoding", "iso-8859-1")
        .schema(schema).csv(str(ruta))
    )
    return df.select([_columna_tipada(c) if c in todas else F.lit(None).cast("string").alias(c)
                      for c in columnas])


# --------------------------------------------------------------------------
# Transformaciones
# --------------------------------------------------------------------------

def indice_mes(col):
    """AAAAMM a un entero que avanza de a uno por mes."""
    return (col / 100).cast("int") * 12 + col % 100


def marcar_incumplimiento(perf: DataFrame) -> DataFrame:
    """Agrega incumple: 90 días o más de mora, adjudicación (RA) o salida por 02, 03 o 09."""
    return perf.withColumn(
        "incumple",
        F.coalesce(F.col("dlq_num") >= esquema.ESTADOS_INCUMPLIMIENTO_MIN, F.lit(False))
        | F.coalesce(F.col("dlq_status") == "RA", F.lit(False))
        | F.coalesce(F.col("zero_balance_code").isin(list(esquema.CODIGOS_CIERRE_INCUMPLIMIENTO)), F.lit(False)),
    )


def limpiar_desempeno(perf: DataFrame, orig: DataFrame, pandemia: tuple[int, int]) -> DataFrame:
    primer_pago = orig.select("loan_id", indice_mes(F.col("first_payment_date")).alias("mes_primer_pago"))
    perf = (
        perf.withColumn("zero_balance_code", F.col("zero_balance_code").try_cast("int"))
        .withColumn("mes", indice_mes(F.col("period")))
        .withColumn("dlq_num", F.col("dlq_status").try_cast("int"))
        .withColumn("es_ra", F.col("dlq_status") == "RA")
        .withColumn("modificado", F.coalesce(F.col("mod_flag").isin("Y", "P"), F.lit(False)))
        .withColumn("en_pandemia", F.col("period").between(*pandemia))
        .join(F.broadcast(primer_pago), "loan_id", "left")
        # loan_age se reinicia al modificar el crédito, por eso la edad se calcula desde el primer pago
        .withColumn("edad", F.col("mes") - F.col("mes_primer_pago") + 1)
        .drop("mes_primer_pago")
    )
    return marcar_incumplimiento(perf)


def resumen_creditos(perf: DataFrame) -> DataFrame:
    """Una fila por crédito con lo necesario para las etiquetas."""
    return perf.groupBy("loan_id").agg(
        F.max("edad").alias("edad_maxima"),
        F.min(F.when(F.col("incumple"), F.col("edad"))).alias("edad_primer_incumplimiento"),
        F.max("zero_balance_code").alias("codigo_salida"),
        F.coalesce(F.max(F.col("zero_balance_code") == esquema.CODIGO_CIERRE_DEFECTO), F.lit(False)).alias("defecto"),
    )


def etiqueta_p1(resumen: DataFrame, ventana: int) -> DataFrame:
    """y_p1 = 1 si el primer incumplimiento ocurre en los primeros `ventana` meses.

    Excluye los créditos con código 96. y_p1 queda nulo cuando la ventana no se
    alcanzó a observar y el crédito sigue activo sin haber incumplido.
    """
    incumple_en_ventana = F.coalesce(F.col("edad_primer_incumplimiento") <= ventana, F.lit(False))
    etiquetable = (incumple_en_ventana | (F.col("edad_maxima") >= ventana)
                   | F.col("codigo_salida").isNotNull())
    return (
        resumen.where(~F.col("defecto"))
        .select("loan_id", F.when(etiquetable, incumple_en_ventana.cast("int")).alias("y_p1"))
    )


# --------------------------------------------------------------------------
# Escritura y ejecución
# --------------------------------------------------------------------------

def escribir(df: DataFrame, ruta: Path, archivos: int = 1) -> None:
    df.coalesce(archivos).write.mode("overwrite").option("compression", "snappy").parquet(str(ruta))
    for crc in ruta.glob(".*.crc"):
        crc.unlink()


def ejecutar(trimestre: str, params: dict, raiz: Path = Path(".")) -> dict:
    p_spark, p_etq = params["spark"], params["etiquetas"]
    pandemia = tuple(params["pandemia"])
    particion = f"trimestre={trimestre}"
    salidas = {
        "originacion": raiz / "data/limpios/originacion" / particion,
        "desempeno": raiz / "data/limpios/desempeno" / particion,
        "p1": raiz / "data/modelado/p1" / particion,
    }

    spark = crear_sesion(p_spark["memoria_driver"], p_spark["particiones"])
    temporal = Path(tempfile.mkdtemp(prefix=f"etl_{trimestre}_"))
    try:
        ruta_orig, ruta_perf = extraer_zip(raiz / f"data/crudos/historical_data_{trimestre}.zip", temporal)

        orig = leer_crudo(spark, ruta_orig, "originacion", COLUMNAS_ORIGINACION)
        perf = leer_crudo(spark, ruta_perf, "desempeno", COLUMNAS_DESEMPENO)
        perf = limpiar_desempeno(perf, orig, pandemia)
        escribir(perf.sortWithinPartitions("loan_id", "mes"), salidas["desempeno"], archivos=4)

        # El resto se calcula desde el Parquet ya escrito, que es más rápido que releer el texto
        perf = spark.read.parquet(str(salidas["desempeno"]))
        resumen = resumen_creditos(perf).cache()
        orig = orig.join(resumen.select("loan_id", "defecto"), "loan_id", "left")
        escribir(orig, salidas["originacion"])

        orig = spark.read.parquet(str(salidas["originacion"]))
        p1 = (
            etiqueta_p1(resumen, p_etq["ventana_p1"])
            .join(orig.select("loan_id", "first_payment_date", *VARIABLES_P1), "loan_id")
            .select("loan_id", "first_payment_date", *VARIABLES_P1, "y_p1")
        )
        escribir(p1, salidas["p1"])
        p1 = spark.read.parquet(str(salidas["p1"]))

        metricas = {
            "trimestre": trimestre,
            "creditos": orig.count(),
            "registros_desempeno": perf.count(),
            "periodo_inicial": perf.agg(F.min("period")).first()[0],
            "fecha_corte": perf.agg(F.max("period")).first()[0],
            "excluidos_codigo_96": orig.where(F.col("defecto")).count(),
            "p1_creditos": p1.count(),
            "p1_etiquetables": p1.where(F.col("y_p1").isNotNull()).count(),
            "p1_positivos": p1.where(F.col("y_p1") == 1).count(),
        }
        destino = raiz / "data/metricas/etl" / f"{trimestre}.json"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(metricas, indent=2) + "\n")
        return metricas
    finally:
        shutil.rmtree(temporal, ignore_errors=True)
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL de un trimestre de Freddie Mac")
    parser.add_argument("--trimestre", required=True, help="por ejemplo 2019Q1")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    params = yaml.safe_load(Path(args.params).read_text())
    t0 = time.time()
    print(json.dumps(ejecutar(args.trimestre, params), indent=2))
    print(f"Tiempo: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

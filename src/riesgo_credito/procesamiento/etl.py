"""ETL de un trimestre del Single-Family Loan-Level Dataset de Freddie Mac con Spark.

Lee el zip trimestral de data/crudos, lo descomprime en una carpeta temporal y
escribe en Parquet:

- data/limpios/originacion/trimestre=AAAATn: un crédito por fila.
- data/limpios/desempeno/trimestre=AAAATn: un crédito por mes.
- data/modelado/p1/trimestre=AAAATn: variables de la solicitud y etiqueta de la pregunta 1.
- data/modelado/p2/trimestre=AAAATn: variables mensuales y etiquetas de la pregunta 2.
- data/tablero/{originacion,serie_mensual,cartera_activa}/trimestre=AAAATn: agregados del tablero.
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

import pyarrow.parquet as pq
import yaml
from pyspark.sql import DataFrame, SparkSession, Window
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


def construir_p2(perf: DataFrame, orig: DataFrame, ventana: int, historial: int) -> DataFrame:
    """Una fila por crédito-mes activo, al día o con 30 días de atraso, sin código 96.

    Las variables usan solo información hasta el mes t. y_p2 = 1 si hay un
    incumplimiento en los `ventana` meses siguientes; y_p2_60 hace lo mismo
    con 60 días o más. Las dos quedan nulas cuando esos meses no se alcanzaron
    a observar y el crédito no salió ni incumplió antes.
    """
    credito = Window.partitionBy("loan_id")
    orden = credito.orderBy("mes")
    siguientes = orden.rangeBetween(1, ventana)
    previos = orden.rangeBetween(-historial, -1)
    pasado = orden.rangeBetween(Window.unboundedPreceding, -1)

    atrasado = F.coalesce(F.col("dlq_num") >= 1, F.lit(False))
    incumple_60 = (F.coalesce(F.col("dlq_num") >= 2, F.lit(False)) | F.col("es_ra")
                   | F.coalesce(F.col("zero_balance_code").isin(list(esquema.CODIGOS_CIERRE_INCUMPLIMIENTO)), F.lit(False)))
    observado = ((F.col("mes") + ventana <= F.col("mes_ultimo")) | F.col("codigo_salida").isNotNull())

    base = (
        perf.join(orig.select("loan_id", "orig_upb", "orig_rate", "defecto"), "loan_id")
        .where(~F.col("defecto"))
        .withColumn("mes_ultimo", F.max("mes").over(credito))
        .withColumn("codigo_salida", F.max("zero_balance_code").over(credito))
        .withColumn("evento_90", F.max(F.col("incumple").cast("int")).over(siguientes))
        .withColumn("evento_60", F.max(incumple_60.cast("int")).over(siguientes))
        .withColumn("atrasos_12m", F.coalesce(F.sum(atrasado.cast("int")).over(previos), F.lit(0)))
        .withColumn("max_atraso_12m", F.coalesce(F.max("dlq_num").over(previos), F.lit(0)))
        .withColumn("indulgencia_12m", F.coalesce(
            F.sum(F.coalesce(F.col("assistance_plan") == "F", F.lit(False)).cast("int")).over(previos), F.lit(0)))
        .withColumn("meses_desde_atraso", F.col("mes") - F.max(F.when(atrasado, F.col("mes"))).over(pasado))
    )
    return (
        base.where(F.col("dlq_num").isin(0, 1) & F.col("zero_balance_code").isNull())
        .select(
            "loan_id", "period", "edad", "dlq_num",
            "current_upb", (F.col("current_upb") / F.col("orig_upb")).alias("pct_saldo"),
            "current_rate", (F.col("current_rate") - F.col("orig_rate")).alias("dif_tasa"),
            "eltv",
            "atrasos_12m", "max_atraso_12m", "meses_desde_atraso",
            F.col("meses_desde_atraso").isNotNull().alias("con_atraso_previo"),
            F.col("assistance_plan").alias("plan_asistencia"), "indulgencia_12m",
            F.col("payment_deferral").isNotNull().alias("diferimiento"),
            F.coalesce(F.col("non_interest_upb") > 0, F.lit(False)).alias("saldo_sin_interes"),
            F.coalesce(F.col("dlq_disaster") == "Y", F.lit(False)).alias("mora_desastre"),
            "modificado", "en_pandemia",
            F.when((F.col("evento_90") == 1) | observado, F.coalesce(F.col("evento_90"), F.lit(0))).alias("y_p2"),
            F.when((F.col("evento_60") == 1) | observado, F.coalesce(F.col("evento_60"), F.lit(0))).alias("y_p2_60"),
            F.pmod(F.xxhash64("loan_id"), F.lit(100)).cast("int").alias("grupo_muestra"),
        )
    )


# --------------------------------------------------------------------------
# Escritura y ejecución
# --------------------------------------------------------------------------

def escribir(df: DataFrame, ruta: Path, orden: list[str], archivos: int = 1) -> None:
    """Escribe Parquet de forma reproducible: los mismos datos producen los mismos archivos.

    Spark ordena las filas y escribe en una carpeta temporal; luego cada archivo
    se reescribe con pyarrow con un nombre fijo. Spark deja nombres aleatorios y
    un pie de archivo que varía entre ejecuciones, lo que haría que DVC
    volviera a subir tablas que no cambiaron.
    """
    temporal = ruta.parent / f".tmp_{ruta.name.replace('=', '_')}"
    shutil.rmtree(temporal, ignore_errors=True)
    df.coalesce(archivos).sortWithinPartitions(*orden).write.mode("overwrite").parquet(str(temporal))
    shutil.rmtree(ruta, ignore_errors=True)
    ruta.mkdir(parents=True)
    for i, parte in enumerate(sorted(temporal.glob("part-*.parquet"))):
        origen = pq.ParquetFile(parte)
        with pq.ParquetWriter(ruta / f"part-{i:05d}.parquet", origen.schema_arrow, compression="snappy") as destino:
            for grupo in range(origen.num_row_groups):
                destino.write_table(origen.read_row_group(grupo))
    shutil.rmtree(temporal)


def ejecutar(trimestre: str, params: dict, raiz: Path = Path(".")) -> dict:
    p_spark, p_etq = params["spark"], params["etiquetas"]
    pandemia = tuple(params["pandemia"])
    particion = f"trimestre={trimestre}"
    salidas = {
        "originacion": raiz / "data/limpios/originacion" / particion,
        "desempeno": raiz / "data/limpios/desempeno" / particion,
        "p1": raiz / "data/modelado/p1" / particion,
        "p2": raiz / "data/modelado/p2" / particion,
        "tablero_originacion": raiz / "data/tablero/originacion" / particion,
        "tablero_serie_mensual": raiz / "data/tablero/serie_mensual" / particion,
        "tablero_cartera_activa": raiz / "data/tablero/cartera_activa" / particion,
    }
    # Importación local: tablero usa VARIABLES_P1 de este módulo
    from riesgo_credito.procesamiento import tablero

    spark = crear_sesion(p_spark["memoria_driver"], p_spark["particiones"])
    temporal = Path(tempfile.mkdtemp(prefix=f"etl_{trimestre}_"))
    try:
        ruta_orig, ruta_perf = extraer_zip(raiz / f"data/crudos/historical_data_{trimestre}.zip", temporal)

        orig = leer_crudo(spark, ruta_orig, "originacion", COLUMNAS_ORIGINACION)
        perf = leer_crudo(spark, ruta_perf, "desempeno", COLUMNAS_DESEMPENO)
        perf = limpiar_desempeno(perf, orig, pandemia)
        escribir(perf, salidas["desempeno"], ["loan_id", "mes"], archivos=4)

        # El resto se calcula desde el Parquet ya escrito, que es más rápido que releer el texto
        perf = spark.read.parquet(str(salidas["desempeno"]))
        resumen = resumen_creditos(perf).cache()
        orig = orig.join(resumen.select("loan_id", "defecto"), "loan_id", "left")
        escribir(orig, salidas["originacion"], ["loan_id"])

        orig = spark.read.parquet(str(salidas["originacion"]))
        p1 = (
            etiqueta_p1(resumen, p_etq["ventana_p1"])
            .join(orig.select("loan_id", "first_payment_date", *VARIABLES_P1), "loan_id")
            .select("loan_id", "first_payment_date", *VARIABLES_P1, "y_p1")
        )
        escribir(p1, salidas["p1"], ["loan_id"])
        p1 = spark.read.parquet(str(salidas["p1"]))

        p2 = construir_p2(perf, orig, p_etq["ventana_p2"], params["variables"]["historial_meses"])
        escribir(p2, salidas["p2"], ["loan_id", "period"], archivos=4)
        p2 = spark.read.parquet(str(salidas["p2"]))

        fecha_corte = perf.agg(F.max("period")).first()[0]
        escribir(tablero.originacion(orig, p1), salidas["tablero_originacion"],
                 ["state", "purpose", "rango_puntaje", "rango_ltv"])
        escribir(tablero.serie_mensual(perf, orig), salidas["tablero_serie_mensual"], ["period", "state"])
        escribir(tablero.cartera_activa(p2, orig, fecha_corte), salidas["tablero_cartera_activa"], ["loan_id"])

        carta = F.col("dlq_num") == 0
        conteos_p2 = p2.agg(
            F.count("*").alias("p2_filas"),
            F.count(F.when(carta, F.col("y_p2"))).alias("p2_etiquetables"),
            F.sum(F.when(carta, F.col("y_p2"))).alias("p2_positivos"),
            F.count("y_p2").alias("p2_30_etiquetables"),
            F.sum("y_p2").alias("p2_30_positivos"),
            F.count(F.when(carta, F.col("y_p2_60"))).alias("p2_60_etiquetables"),
            F.sum(F.when(carta, F.col("y_p2_60"))).alias("p2_60_positivos"),
        ).first().asDict()

        metricas = {
            "trimestre": trimestre,
            "creditos": orig.count(),
            "registros_desempeno": perf.count(),
            "periodo_inicial": perf.agg(F.min("period")).first()[0],
            "fecha_corte": fecha_corte,
            "excluidos_codigo_96": orig.where(F.col("defecto")).count(),
            "p1_creditos": p1.count(),
            "p1_etiquetables": p1.where(F.col("y_p1").isNotNull()).count(),
            "p1_positivos": p1.where(F.col("y_p1") == 1).count(),
            **{k: int(v) for k, v in conteos_p2.items()},
            "activos_al_corte": perf.where((F.col("period") == fecha_corte)
                                           & F.col("zero_balance_code").isNull()).count(),
            "cartera_activa": spark.read.parquet(str(salidas["tablero_cartera_activa"])).count(),
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

"""Puntaje por lotes de la pregunta 2 para la vista Seguimiento del tablero.

Toma los créditos activos y al día del último mes de la tabla de la pregunta 2 (o del mes que se
indique), les agrega los datos de la solicitud original, pide la probabilidad de llegar a 90 días
de mora en los próximos 3 meses a la API (ruta /api/v1/predict_p2) y guarda la lista en
data/tablero/alertas_p2, que el tablero lee para la lista priorizada.

Por memoria y tiempo se puntúa una muestra fija de créditos: los de grupo_muestra >= 100 - grupos
(con --grupos 1, el 1 % de los créditos, la misma regla de la prueba del modelo).

Uso, desde la raíz del repositorio, con la API corriendo:
    python src/riesgo_credito/modelos/puntaje_p2.py --api http://localhost:8001/api/v1/predict_p2
Sin API, con el paquete del modelo instalado en el ambiente (pip install api/model-pkg/*.whl):
    python src/riesgo_credito/modelos/puntaje_p2.py --local
"""
import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

RAIZ = Path(__file__).resolve().parents[3]
RUTA_P2 = RAIZ / "data" / "modelado" / "p2"
RUTA_P1 = RAIZ / "data" / "modelado" / "p1"
SALIDA = RAIZ / "data" / "tablero" / "alertas_p2"
ETIQUETA = "y_p2"

P2_VARS = ["edad", "pct_saldo", "dif_tasa", "eltv", "atrasos_12m", "max_atraso_12m", "meses_desde_atraso",
           "indulgencia_12m", "con_atraso_previo", "diferimiento", "diferimiento_mes", "saldo_sin_interes",
           "mora_desastre", "modificado", "plan_asistencia"]
P1_VARS = ["credit_score", "dti", "oltv", "ocltv", "orig_rate", "orig_upb", "orig_term", "mi_pct",
           "num_borrowers", "num_units", "purpose", "occupancy", "channel", "first_time_homebuyer",
           "prop_type", "valuation_method", "state"]


def log(mensaje):
    print(f"[{time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


def dataset(ruta):
    return ds.dataset(ruta, format="parquet", partitioning="hive")


def leer(d, columnas, filtro):
    """Lee filtrando por lotes, con poca anticipación para no agotar la memoria del EC2."""
    columnas = [c for c in columnas if c in d.schema.names]
    lotes = [b for b in d.scanner(columns=columnas, filter=filtro, batch_size=100_000,
                                  batch_readahead=2, fragment_readahead=1).to_batches() if b.num_rows]
    if not lotes:
        return pd.DataFrame(columns=columnas)
    return pa.Table.from_batches(lotes).to_pandas()


def ultimo_periodo(d):
    """El último mes de la tabla está en los créditos originados en el último trimestre."""
    trimestres = sorted(p.name.split("=", 1)[1] for p in RUTA_P2.glob("trimestre=*"))
    for t in reversed(trimestres):
        periodos = leer(d, ["period"], ds.field("trimestre") == t)["period"]
        if len(periodos):
            return int(periodos.max())
    raise SystemExit(f"No hay datos en {RUTA_P2}: descarga la tabla con DVC")


def creditos_del_mes(periodo, grupos):
    d = dataset(RUTA_P2)
    filtro = ((ds.field("period") == periodo) & (ds.field("dlq_num") == 0)
              & (ds.field("grupo_muestra") >= 100 - grupos))
    log(f"Leyendo los créditos al día de {periodo} con grupo_muestra >= {100 - grupos}")
    df = leer(d, ["loan_id", "period", "trimestre", ETIQUETA] + P2_VARS, filtro)
    log(f"{len(df):,} créditos; agregando los datos de la solicitud original")
    ids = df["loan_id"].unique().tolist()
    p1 = leer(dataset(RUTA_P1), ["loan_id"] + P1_VARS, ds.field("loan_id").isin(ids))
    return df.merge(p1.drop_duplicates("loan_id"), on="loan_id", how="left")


def a_json(valor):
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return None
    if isinstance(valor, np.generic):
        valor = valor.item()
        return None if isinstance(valor, float) and math.isnan(valor) else valor
    if valor is pd.NA or valor is pd.NaT:
        return None
    return valor


def registros(df):
    columnas = ["loan_id", "period"] + P2_VARS + P1_VARS
    salida = []
    for fila in df[columnas].astype(object).itertuples(index=False):
        r = {c: a_json(v) for c, v in zip(columnas, fila)}
        r["loan_id"] = str(r["loan_id"])
        salida.append(r)
    return salida


def puntuar_api(df, url, lote):
    probabilidades = []
    for i in range(0, len(df), lote):
        cuerpo = json.dumps({"inputs": registros(df.iloc[i:i + lote])}).encode("utf-8")
        peticion = urllib.request.Request(url, data=cuerpo, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(peticion, timeout=300) as respuesta:
                probabilidades.extend(json.loads(respuesta.read())["predictions"])
        except urllib.error.HTTPError as error:
            raise SystemExit(f"La API respondió {error.code}: {error.read()[:500]!r}")
        log(f"{min(i + lote, len(df)):,} de {len(df):,} créditos puntuados")
    return np.asarray(probabilidades, dtype="float64")


def puntuar_local(df):
    from modelo_riesgo.p2 import make_prediction_p2
    resultado = make_prediction_p2(input_data=pd.DataFrame(registros(df)))
    if resultado["errors"]:
        raise SystemExit(f"Error de validación: {resultado['errors']}")
    return np.asarray(resultado["predictions"], dtype="float64")


def resumen(df, periodo, fuente, grupos):
    info = {"periodo": int(periodo), "creditos": int(len(df)), "muestra": f"grupo_muestra >= {100 - grupos}",
            "porcentaje_creditos": grupos, "probabilidad_media": float(df["probabilidad"].mean()),
            "fuente": fuente, "generado": time.strftime("%Y-%m-%d %H:%M")}
    if ETIQUETA in df and df[ETIQUETA].notna().mean() > 0.95 and df[ETIQUETA].sum() > 0:
        # El mes ya tiene sus 3 meses siguientes observados: se puede medir qué tanto acertó la lista
        from sklearn.metrics import roc_auc_score
        e = df[df[ETIQUETA].notna()]
        y = e[ETIQUETA].astype(int).to_numpy()
        orden = np.argsort(-e["probabilidad"].to_numpy(), kind="stable")[:math.ceil(0.1 * len(e))]
        info.update({"positivos_observados": int(y.sum()), "tasa_observada": float(y.mean()),
                     "auc_roc": float(roc_auc_score(y, e["probabilidad"])),
                     "captura_decil": float(y[orden].sum() / y.sum())})
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8001/api/v1/predict_p2", help="ruta predict_p2 de la API")
    parser.add_argument("--local", action="store_true", help="usar el paquete instalado en lugar de la API")
    parser.add_argument("--periodo", type=int, help="mes de corte AAAAMM (por defecto, el último de la tabla)")
    parser.add_argument("--grupos", type=int, default=1, help="porcentaje de créditos a puntuar (1 a 100)")
    parser.add_argument("--lote", type=int, default=1000, help="créditos por petición a la API")
    args = parser.parse_args()

    periodo = args.periodo or ultimo_periodo(dataset(RUTA_P2))
    log(f"Mes de corte: {periodo}")
    df = creditos_del_mes(periodo, args.grupos)
    if df.empty:
        raise SystemExit(f"No hay créditos al día en {periodo}")

    t0 = time.time()
    fuente = "paquete local" if args.local else args.api
    df["probabilidad"] = puntuar_local(df) if args.local else puntuar_api(df, args.api, args.lote)
    log(f"Puntaje listo en {time.time() - t0:.0f} s")

    df["saldo_estimado"] = df["pct_saldo"] * df["orig_upb"]
    df = df.sort_values("probabilidad", ascending=False, kind="stable").reset_index(drop=True)
    df["prioridad"] = np.arange(1, len(df) + 1)
    SALIDA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SALIDA / "alertas.parquet", index=False)
    info = resumen(df, periodo, fuente, args.grupos)
    (SALIDA / "resumen.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Guardado en {SALIDA.relative_to(RAIZ)}: {json.dumps(info, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

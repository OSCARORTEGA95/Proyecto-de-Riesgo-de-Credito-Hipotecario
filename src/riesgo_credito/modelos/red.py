"""Red neuronal densa con y sin dropout para la pregunta 1.

Usa la misma partición temporal, el mismo muestreo con corrección de la probabilidad, las
mismas métricas y el mismo registro en MLflow que los modelos de k vecinos
(riesgo_credito.modelos.knn), así que se compara directamente con ellos, con la regresión
logística, con naive Bayes y con la referencia del puntaje. Además de las métricas finales,
cada ejecución guarda en MLflow la curva por época (pérdida y área bajo la curva en
validación), que se ve en la pestaña "Model metrics" de la ejecución, y el historial en CSV.
Cada versión se exporta también a modelos/red/.

Uso, desde la raíz del repositorio:
    export MLFLOW_TRACKING_URI=http://107.22.35.29:8050
    PYTHONPATH=src python -m riesgo_credito.modelos.red                    # referencia y las 8 versiones
    PYTHONPATH=src python -m riesgo_credito.modelos.red --version v1 v5    # solo algunas
    PYTHONPATH=src python -m riesgo_credito.modelos.red --sin-referencia
"""
import argparse
import copy
import gc
import json
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import yaml
from mlflow.entities import Metric
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from riesgo_credito.modelos import knn as base
from riesgo_credito.modelos.knn_modelo import ModeloKNN as ModeloCorregido
from riesgo_credito.modelos.knn_modelo import preparar, probabilidad
from riesgo_credito.modelos.red_modelo import RedNeuronal

CONFIG = Path(os.environ.get("RED_CONFIG", base.RAIZ / "config" / "modelos_red.yaml"))
PREGUNTA = "p1"
log = base.log


def cargar_config():
    cfg = base.cargar_config()  # partición, semilla y lotes de los modelos de k vecinos
    with open(CONFIG, encoding="utf-8") as f:
        propia = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg[PREGUNTA]["max_entrenamiento"] = propia["max_entrenamiento"]
    cfg["mlflow"] = propia["mlflow"]  # base.registrar toma el experimento de cfg["mlflow"]
    cfg["propia"] = propia
    base.EXPORTAR = base.RAIZ / "modelos" / "red"
    return cfg


def datos_muestra(cfg, etapa, proporcion, maximo, semilla):
    """Igual que base.datos + base.muestrear, pero sin cargar todos los créditos en memoria.

    Cuenta positivos y negativos leyendo solo la etiqueta, y después recorre la tabla por
    lotes guardando todos los positivos y un poco más de los negativos necesarios, elegidos
    por un hash del loan_id (la misma muestra siempre, sin importar el orden de los lotes).
    base.muestrear toma de ahí la muestra exacta y calcula r con los totales reales.
    """
    c, y = cfg[PREGUNTA], cfg[PREGUNTA]["etiqueta"]
    d = base.dataset(c["ruta"])
    filtro = ds.field("trimestre").isin(c[etapa]) & ds.field(y).is_valid()
    etiquetas = d.to_table(columns=[y], filter=filtro).column(y).to_numpy(zero_copy_only=False)
    positivos, negativos = int((etiquetas == 1).sum()), int((etiquetas == 0).sum())
    del etiquetas
    fraccion = min(1.0, 1.2 * proporcion * positivos / max(negativos, 1))
    columnas = ["loan_id"] + base.VARS_P1 + [y]
    partes = []
    for lote in d.to_batches(columns=columnas, filter=filtro, batch_size=200_000):
        yy = lote.column(y).to_numpy(zero_copy_only=False)
        ids = np.asarray(lote.column("loan_id").to_pylist(), dtype=object)
        azar = pd.util.hash_array(ids).astype(np.float64) / 2.0 ** 64
        conservar = (yy == 1) | (azar < fraccion)
        if conservar.any():
            partes.append(lote.filter(pa.array(conservar)).to_pandas())
    df = pd.concat(partes, ignore_index=True)
    df[y] = df[y].astype(int)
    df = df.sort_values("loan_id", kind="stable").reset_index(drop=True)
    log(f"{etapa}: {positivos} positivos y {negativos} negativos; leídos {len(df)} para la muestra")
    return base.muestrear(df, y, proporcion, maximo, semilla, total_positivos=positivos, total_negativos=negativos)


def crear_red(v, e, epocas, paciencia, semilla):
    return RedNeuronal(capas=tuple(v["capas"]), dropout=v["dropout"], epocas=epocas, tam_lote=e["tam_lote"],
                       tasa_aprendizaje=e["tasa_aprendizaje"], l2=e["l2"], paciencia=paciencia, semilla=semilla,
                       verbose=True)


def registrar_epocas(run_id, historial_seleccion, historial_final, carpeta):
    """Curvas por época en MLflow (una métrica con un paso por época) y el historial en CSV."""
    ahora = int(time.time() * 1000)
    metricas = []
    for fila in historial_seleccion:
        for clave in ("perdida", "perdida_validacion", "auc_validacion"):
            if clave in fila:
                nombre = "epoca_perdida_seleccion" if clave == "perdida" else f"epoca_{clave}"
                metricas.append(Metric(nombre, fila[clave], ahora, fila["epoca"]))
    for fila in historial_final:
        metricas.append(Metric("epoca_perdida_entrenamiento", fila["perdida"], ahora, fila["epoca"]))
    cliente = mlflow.MlflowClient()
    for i in range(0, len(metricas), 900):  # límite de 1000 métricas por llamada
        cliente.log_batch(run_id, metrics=metricas[i:i + 900])
    tabla = pd.DataFrame(historial_seleccion).add_suffix("_seleccion").rename(columns={"epoca_seleccion": "epoca"})
    final = pd.DataFrame(historial_final)[["epoca", "perdida", "segundos"]].add_suffix("_entrenamiento")
    tabla = tabla.merge(final.rename(columns={"epoca_entrenamiento": "epoca"}), on="epoca", how="outer")
    ruta = carpeta / "historial_epocas.csv"
    tabla.to_csv(ruta, index=False)
    cliente.log_artifact(run_id, str(ruta))


def correr_version(cfg, version):
    base.preparar_mlflow(cfg, PREGUNTA)
    v = cfg["propia"]["versiones"][PREGUNTA][version]
    e = cfg["propia"]["entrenamiento"]
    numericas, banderas, categoricas = base.CONJUNTOS[v["variables"]]
    y, semilla, lote = cfg[PREGUNTA]["etiqueta"], cfg["semilla"], cfg["lote_prediccion"]
    maximo = cfg[PREGUNTA]["max_entrenamiento"]

    log(f"red {PREGUNTA} {version} ({v['descripcion']}): capas {v['capas']}, dropout {v['dropout']}; "
        f"selección del número de épocas")
    t0 = time.time()
    sel, _ = datos_muestra(cfg, "seleccion", v["proporcion"], maximo, semilla)
    val, _ = base.datos(cfg, PREGUNTA, "validacion")
    prep = base.crear_preparacion(numericas + banderas, categoricas)
    A = prep.fit_transform(preparar(sel, numericas, banderas, categoricas))
    B = prep.transform(preparar(val, numericas, banderas, categoricas))
    red_sel = crear_red(v, e, e["epocas_max"], e["paciencia"], semilla).fit(A, sel[y].to_numpy(), B, val[y].to_numpy())
    epocas = red_sel.mejor_epoca_
    auc_val = red_sel.mejor_auc_validacion_
    historial_sel = red_sel.historial_
    t_sel = time.time() - t0
    log(f"Mejor época {epocas} de {len(historial_sel)}: área bajo la curva en validación {auc_val:.4f}")
    del sel, val, A, B, red_sel
    gc.collect()

    log(f"Entrenamiento final con {epocas} épocas")
    ent, muestra = datos_muestra(cfg, "entrenamiento", v["proporcion"], maximo, semilla)
    t1 = time.time()
    prep = base.crear_preparacion(numericas + banderas, categoricas)
    A = prep.fit_transform(preparar(ent, numericas, banderas, categoricas))
    red = crear_red(v, e, epocas, None, semilla).fit(A, ent[y].to_numpy())
    t_fit = time.time() - t1
    modelo = ModeloCorregido(numericas, banderas, categoricas, muestra["r"])
    modelo.pipeline = Pipeline(prep.steps + [("red", red)])
    n_columnas = A.shape[1]
    del ent, A
    gc.collect()

    log("Predicción sobre prueba")
    prueba, _ = base.datos(cfg, PREGUNTA, "prueba")
    t2 = time.time()
    p = probabilidad(modelo.pipeline, preparar(prueba, numericas, banderas, categoricas), muestra["r"], lote)
    t_pred = time.time() - t2
    metr = base.metricas(prueba[y], p)
    metr.update({"auc_validacion": auc_val, "mejor_epoca": epocas, "epocas_seleccion": len(historial_sel),
                 "perdida_final": red.historial_[-1]["perdida"], "n_parametros": red.n_parametros(),
                 "tiempo_seleccion_s": t_sel, "tiempo_entrenamiento_s": t_fit, "tiempo_prediccion_s": t_pred,
                 "n_entrenamiento": muestra["positivos"] + muestra["negativos"],
                 "positivos_entrenamiento": muestra["positivos"]})
    params = {"pregunta": PREGUNTA, "version": version, "descripcion": v["descripcion"],
              "algoritmo": "red neuronal densa (perceptrón multicapa)", "n_capas_ocultas": len(v["capas"]),
              "capas": "-".join(map(str, v["capas"])), "dropout": v["dropout"], "activacion": "relu",
              "salida": "sigmoide", "optimizador": "adam", "tasa_aprendizaje": e["tasa_aprendizaje"],
              "l2": e["l2"], "tam_lote": e["tam_lote"], "epocas_max": e["epocas_max"], "paciencia": e["paciencia"],
              "epocas": epocas, "variables": v["variables"],
              "lista_variables": ",".join(numericas + banderas + categoricas), "n_columnas_codificadas": n_columnas,
              "proporcion_negativos": v["proporcion"], "r": muestra["r"], "frac_positivos": muestra["frac_positivos"],
              "frac_negativos": muestra["frac_negativos"],
              "periodo_seleccion": base.descripcion_etapa(cfg, PREGUNTA, "seleccion"),
              "periodo_validacion": base.descripcion_etapa(cfg, PREGUNTA, "validacion"),
              "periodo_entrenamiento": base.descripcion_etapa(cfg, PREGUNTA, "entrenamiento"),
              "periodo_prueba": base.descripcion_etapa(cfg, PREGUNTA, "prueba"),
              "regla_validacion": base.regla_evaluacion(cfg, PREGUNTA, "validacion"),
              "regla_prueba": base.regla_evaluacion(cfg, PREGUNTA, "prueba"), "semilla": semilla}
    log(f"Prueba: área bajo la curva {metr['auc_roc']:.4f}, precisión promedio {metr['auc_pr']:.4f}, captura del "
        f"decil superior {100 * metr['captura_decil']:.1f} %, probabilidad media {metr['prob_media']:.5f} frente a "
        f"tasa {metr['tasa_observada']:.5f}")
    columnas = ["loan_id"] + numericas + banderas + categoricas
    ejemplo = base.entrada_json(prueba[columnas].sample(n=min(2000, len(prueba)), random_state=semilla),
                                numericas, banderas, categoricas)
    base.registrar(cfg, PREGUNTA, f"{version}_{v['descripcion']}", params, metr, modelo, modelo.pipeline, ejemplo)

    destino = base.EXPORTAR / PREGUNTA / version
    run_id = json.loads((destino / "parametros.json").read_text(encoding="utf-8"))["run_id"]
    registrar_epocas(run_id, historial_sel, red.historial_, destino)


def main():
    parser = argparse.ArgumentParser(description="Red neuronal de la pregunta 1")
    parser.add_argument("--version", nargs="+", default=[], help="por defecto, las 8 versiones")
    parser.add_argument("--sin-referencia", action="store_true", help="no registrar la referencia del puntaje")
    args = parser.parse_args()
    cfg = cargar_config()
    versiones = args.version or list(cfg["propia"]["versiones"][PREGUNTA])
    inicio = time.time()
    if not args.sin_referencia:
        base.correr_referencia(cfg, PREGUNTA)
    for version in versiones:
        correr_version(cfg, version)
    log(f"Terminó en {(time.time() - inicio) / 60:.1f} minutos")


if __name__ == "__main__":
    main()

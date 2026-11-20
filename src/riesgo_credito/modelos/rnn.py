"""Redes recurrentes (GRU) de las preguntas 1 y 2, registradas en MLflow y comparadas con los demás modelos.

Usa la misma partición temporal, el mismo muestreo con corrección de la probabilidad, las
mismas métricas y el mismo registro que k vecinos y la red densa. Cada ejecución guarda en
MLflow la curva por época (pestaña "Model metrics") y el historial en CSV, y se exporta en
modelos/rnn/.

Uso, desde la raíz del repositorio:
    export MLFLOW_TRACKING_URI=http://localhost:8050
    PYTHONPATH=src python -m riesgo_credito.modelos.rnn --pregunta p2               # referencia y todas las versiones
    PYTHONPATH=src python -m riesgo_credito.modelos.rnn --pregunta p2 --version v1
    PYTHONPATH=src python -m riesgo_credito.modelos.rnn --pregunta p1 --sin-referencia
    PYTHONPATH=src python -m riesgo_credito.modelos.rnn --accion comparar             # tabla contra red densa y KNN
"""
import argparse
import copy
import gc
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml
from sklearn.pipeline import Pipeline

from riesgo_credito.modelos import knn as base
from riesgo_credito.modelos import red
from riesgo_credito.modelos.knn_modelo import ModeloKNN as ModeloCorregido
from riesgo_credito.modelos.knn_modelo import corregir, preparar, probabilidad
from riesgo_credito.modelos.rnn_modelo import (RedRecurrente, SecuenciaDeColumnas, SecuenciaMensual,
                                               construir_ancho, indice_mes, nombre_mes)

CONFIG = Path(os.environ.get("RNN_CONFIG", base.RAIZ / "config" / "modelos_rnn.yaml"))
log = base.log
# preparar() agrega las ~200 columnas mensuales una a una; el aviso de fragmentación solo ensucia el registro
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# variables de cada mes de la secuencia de la pregunta 2 (dlq_num: 0 al día, 1 con 30 días de atraso)
MES_NUM = base.P2_NUM + ["dlq_num"]
MES_BANDERAS = base.P2_BANDERAS
ESTATICAS = {"solo_mes": ([], base.P2_CAT), "completa": (base.P1_NUM, base.P2_CAT + base.P1_CAT)}


def cargar_config():
    cfg = copy.deepcopy(base.cargar_config())
    with open(CONFIG, encoding="utf-8") as f:
        propia = yaml.safe_load(f)
    for p in ("p1", "p2"):
        cfg[p]["max_entrenamiento"] = propia["max_entrenamiento"][p]
    cfg["mlflow"] = propia["mlflow"]
    cfg["propia"] = propia
    base.EXPORTAR = base.RAIZ / "modelos" / "rnn"
    return cfg


def crear_rnn(v, e, n_pasos, n_entrada, epocas, paciencia, semilla):
    return RedRecurrente(n_pasos=n_pasos, n_entrada=n_entrada, unidades=v["unidades"], densas=tuple(v["densas"]),
                         dropout=v["dropout"], epocas=epocas, tam_lote=e["tam_lote"],
                         tasa_aprendizaje=e["tasa_aprendizaje"], l2=e["l2"], recorte=e["recorte"],
                         paciencia=paciencia, semilla=semilla, verbose=True)


def parametros_comunes(cfg, pregunta, v, e, red_final, epocas, extra):
    return {"pregunta": pregunta, "algoritmo": "red recurrente GRU", "celda": "GRU", "unidades": v["unidades"],
            "densas": "-".join(map(str, v["densas"])), "dropout": v["dropout"], "optimizador": "adam",
            "tasa_aprendizaje": e["tasa_aprendizaje"], "l2": e["l2"], "recorte_gradiente": e["recorte"],
            "tam_lote": e["tam_lote"], "epocas_max": e["epocas_max"], "paciencia": e["paciencia"], "epocas": epocas,
            "n_parametros": red_final.n_parametros(), "proporcion_negativos": v["proporcion"],
            "periodo_seleccion": base.descripcion_etapa(cfg, pregunta, "seleccion"),
            "periodo_validacion": base.descripcion_etapa(cfg, pregunta, "validacion"),
            "periodo_entrenamiento": base.descripcion_etapa(cfg, pregunta, "entrenamiento"),
            "periodo_prueba": base.descripcion_etapa(cfg, pregunta, "prueba"),
            "regla_validacion": base.regla_evaluacion(cfg, pregunta, "validacion"),
            "regla_prueba": base.regla_evaluacion(cfg, pregunta, "prueba"), "semilla": cfg["semilla"], **extra}


def cerrar(cfg, pregunta, version, v, params, metr, modelo, pipeline, ejemplo, historial_sel, historial_final):
    log(f"Prueba: área bajo la curva {metr['auc_roc']:.4f}, precisión promedio {metr['auc_pr']:.4f}, captura del "
        f"decil superior {100 * metr['captura_decil']:.1f} %, probabilidad media {metr['prob_media']:.5f} frente a "
        f"tasa {metr['tasa_observada']:.5f}")
    base.registrar(cfg, pregunta, f"{version}_{v['descripcion']}", params, metr, modelo, pipeline, ejemplo)
    destino = base.EXPORTAR / pregunta / version
    run_id = json.loads((destino / "parametros.json").read_text(encoding="utf-8"))["run_id"]
    red.registrar_epocas(run_id, historial_sel, historial_final, destino)


# ---------- pregunta 1: la solicitud como secuencia de columnas ----------

def correr_p1(cfg, version):
    pregunta = "p1"
    base.preparar_mlflow(cfg, pregunta)
    v, e = cfg["propia"]["versiones"][pregunta][version], cfg["propia"]["entrenamiento"]
    numericas, banderas, categoricas = base.CONJUNTOS[v["variables"]]
    y, semilla, lote = cfg[pregunta]["etiqueta"], cfg["semilla"], cfg["lote_prediccion"]
    maximo = cfg[pregunta]["max_entrenamiento"]
    log(f"rnn p1 {version} ({v['descripcion']}): GRU de {v['unidades']}, densas {v['densas']}, dropout {v['dropout']}")

    t0 = time.time()
    sel, _ = red.datos_muestra(cfg, "seleccion", v["proporcion"], maximo, semilla)
    val, _ = base.datos(cfg, pregunta, "validacion")
    if len(val) > cfg["propia"]["max_validacion"][pregunta]:
        val = val.sample(n=cfg["propia"]["max_validacion"][pregunta], random_state=semilla)
    prep = Pipeline(base.crear_preparacion(numericas + banderas, categoricas).steps + [("secuencia", SecuenciaDeColumnas())])
    A = prep.fit_transform(preparar(sel, numericas, banderas, categoricas))
    B = prep.transform(preparar(val, numericas, banderas, categoricas))
    D = prep.named_steps["secuencia"].n_columnas_
    red_sel = crear_rnn(v, e, D, D + 1, e["epocas_max"], e["paciencia"], semilla).fit(A, sel[y].to_numpy(),
                                                                                       B, val[y].to_numpy())
    epocas, auc_val, historial_sel, t_sel = red_sel.mejor_epoca_, red_sel.mejor_auc_validacion_, red_sel.historial_, time.time() - t0
    log(f"Mejor época {epocas} de {len(historial_sel)}: área bajo la curva en validación {auc_val:.4f}")
    del sel, val, A, B, red_sel
    gc.collect()

    log(f"Entrenamiento final con {epocas} épocas")
    ent, muestra = red.datos_muestra(cfg, "entrenamiento", v["proporcion"], maximo, semilla)
    t1 = time.time()
    prep = Pipeline(base.crear_preparacion(numericas + banderas, categoricas).steps + [("secuencia", SecuenciaDeColumnas())])
    A = prep.fit_transform(preparar(ent, numericas, banderas, categoricas))
    red_final = crear_rnn(v, e, D, D + 1, epocas, None, semilla).fit(A, ent[y].to_numpy())
    t_fit = time.time() - t1
    pipeline = Pipeline(prep.steps + [("rnn", red_final)])
    modelo = ModeloCorregido(numericas, banderas, categoricas, muestra["r"])
    modelo.pipeline = pipeline
    del ent, A
    gc.collect()

    log("Predicción sobre prueba")
    prueba, _ = base.datos(cfg, pregunta, "prueba")
    t2 = time.time()
    p = probabilidad(pipeline, preparar(prueba, numericas, banderas, categoricas), muestra["r"], lote)
    metr = base.metricas(prueba[y], p)
    metr.update({"auc_validacion": auc_val, "mejor_epoca": epocas, "epocas_seleccion": len(historial_sel),
                 "perdida_final": red_final.historial_[-1]["perdida"], "n_parametros": red_final.n_parametros(),
                 "tiempo_seleccion_s": t_sel, "tiempo_entrenamiento_s": t_fit, "tiempo_prediccion_s": time.time() - t2,
                 "n_entrenamiento": muestra["positivos"] + muestra["negativos"],
                 "positivos_entrenamiento": muestra["positivos"]})
    params = parametros_comunes(cfg, pregunta, v, e, red_final, epocas, {
        "version": version, "descripcion": v["descripcion"], "variables": v["variables"],
        "secuencia": "columnas codificadas de la solicitud, un paso por columna", "n_pasos": D,
        "lista_variables": ",".join(numericas + banderas + categoricas), "r": muestra["r"],
        "frac_positivos": muestra["frac_positivos"], "frac_negativos": muestra["frac_negativos"]})
    columnas = ["loan_id"] + numericas + banderas + categoricas
    ejemplo = base.entrada_json(prueba[columnas].sample(n=min(2000, len(prueba)), random_state=semilla),
                                numericas, banderas, categoricas)
    cerrar(cfg, pregunta, version, v, params, metr, modelo, pipeline, ejemplo, historial_sel, red_final.historial_)


# ---------- pregunta 2: 12 meses de comportamiento ----------

def mes_menos(period, meses):
    i = indice_mes(period) - meses
    return int((i // 12) * 100 + i % 12 + 1)


def historia_p2(cfg, objetivos, etiqueta_cache):
    """Meses de la tabla de la pregunta 2 de los créditos de `objetivos`, desde n_meses − 1 antes del primer mes."""
    c, n_meses = cfg["p2"], cfg["propia"]["n_meses"]
    desde, hasta = int(objetivos["period"].min()), int(objetivos["period"].max())
    ids = objetivos["loan_id"].unique().tolist()

    def leer():
        inicio = mes_menos(desde, n_meses - 1)
        log(f"P2: leyendo la historia de {len(ids)} créditos, {inicio}-{hasta}")
        filtro = (ds.field("trimestre").isin(base.trimestres_hasta(c["ruta"], hasta))
                  & (ds.field("period") >= inicio) & (ds.field("period") <= hasta) & ds.field("loan_id").isin(ids))
        h = base.dataset(c["ruta"]).to_table(columns=["loan_id", "period"] + MES_NUM + MES_BANDERAS,
                                              filter=filtro).to_pandas()
        for col in MES_NUM:
            h[col] = pd.to_numeric(h[col], errors="coerce").astype("float32")
        return h, {}

    return base.en_cache(f"rnn_p2_historia_{etiqueta_cache}_{desde}_{hasta}_{len(ids)}", leer)[0]


def columnas_p2(estaticas, n_meses):
    est_num, est_cat = ESTATICAS[estaticas]
    num = [nombre_mes(v, k) for k in range(n_meses) for v in MES_NUM] + est_num
    ban = [nombre_mes(v, k) for k in range(n_meses) for v in MES_BANDERAS + ["presente"]]
    return num, ban, est_cat


def ancho_preparado(cfg, objetivos, historia, estaticas):
    n_meses = cfg["propia"]["n_meses"]
    num, ban, cat = columnas_p2(estaticas, n_meses)
    ancho = construir_ancho(objetivos, historia, MES_NUM + MES_BANDERAS, n_meses)
    return preparar(ancho, num, ban, cat), ancho


def pipeline_p2(estaticas, n_meses):
    est_num, est_cat = ESTATICAS[estaticas]
    return Pipeline([("secuencia", SecuenciaMensual(MES_NUM + MES_BANDERAS, n_meses,
                                                    estaticas=base.crear_preparacion(est_num, est_cat)))])


def correr_p2(cfg, version):
    pregunta = "p2"
    base.preparar_mlflow(cfg, pregunta)
    v, e = cfg["propia"]["versiones"][pregunta][version], cfg["propia"]["entrenamiento"]
    y, semilla, n_meses = cfg[pregunta]["etiqueta"], cfg["semilla"], cfg["propia"]["n_meses"]
    maximo = cfg[pregunta]["max_entrenamiento"]
    n_entrada = len(MES_NUM + MES_BANDERAS)
    log(f"rnn p2 {version} ({v['descripcion']}): {n_meses} meses × {n_entrada} variables, GRU de {v['unidades']}, "
        f"densas {v['densas']}, dropout {v['dropout']}, estáticas {v['estaticas']}")

    t0 = time.time()
    sel, info = base.datos(cfg, pregunta, "seleccion")
    sel, _ = base.muestrear(sel, y, v["proporcion"], maximo, semilla, **info)
    val, _ = base.datos(cfg, pregunta, "validacion")
    if len(val) > cfg["propia"]["max_validacion"][pregunta]:
        val = val.sample(n=cfg["propia"]["max_validacion"][pregunta], random_state=semilla)
    S, _ = ancho_preparado(cfg, sel, historia_p2(cfg, sel, "seleccion"), v["estaticas"])
    V, _ = ancho_preparado(cfg, val, historia_p2(cfg, val, "validacion"), v["estaticas"])
    prep = pipeline_p2(v["estaticas"], n_meses)
    A, B = prep.fit_transform(S), prep.transform(V)
    red_sel = crear_rnn(v, e, n_meses, n_entrada, e["epocas_max"], e["paciencia"], semilla).fit(
        A, sel[y].to_numpy(), B, val[y].to_numpy())
    epocas, auc_val, historial_sel, t_sel = red_sel.mejor_epoca_, red_sel.mejor_auc_validacion_, red_sel.historial_, time.time() - t0
    log(f"Mejor época {epocas} de {len(historial_sel)}: área bajo la curva en validación {auc_val:.4f}")
    del sel, val, S, V, A, B, red_sel
    gc.collect()

    log(f"Entrenamiento final con {epocas} épocas")
    ent, info = base.datos(cfg, pregunta, "entrenamiento")
    ent, muestra = base.muestrear(ent, y, v["proporcion"], maximo, semilla, **info)
    t1 = time.time()
    E, _ = ancho_preparado(cfg, ent, historia_p2(cfg, ent, "entrenamiento"), v["estaticas"])
    prep = pipeline_p2(v["estaticas"], n_meses)
    A = prep.fit_transform(E)
    red_final = crear_rnn(v, e, n_meses, n_entrada, epocas, None, semilla).fit(A, ent[y].to_numpy())
    t_fit = time.time() - t1
    pipeline = Pipeline(prep.steps + [("rnn", red_final)])
    num, ban, cat = columnas_p2(v["estaticas"], n_meses)
    modelo = ModeloCorregido(num, ban, cat, muestra["r"])
    modelo.pipeline = pipeline
    del ent, E, A
    gc.collect()

    log("Predicción sobre prueba, por bloques")
    prueba, _ = base.datos(cfg, pregunta, "prueba")
    historia = historia_p2(cfg, prueba, "prueba")
    t2 = time.time()
    tam, probs, ejemplo = cfg["propia"]["lote_construccion"], [], None
    for i in range(0, len(prueba), tam):
        bloque = prueba.iloc[i:i + tam]
        X, ancho = ancho_preparado(cfg, bloque, historia, v["estaticas"])
        probs.append(probabilidad(pipeline, X, muestra["r"], cfg["lote_prediccion"]))
        if ejemplo is None:
            ejemplo = base.entrada_json(ancho[["loan_id", "period"] + num + ban + cat].head(200), num, ban, cat)
        log(f"  {min(i + tam, len(prueba))} de {len(prueba)} filas")
    p = np.concatenate(probs)
    metr = base.metricas(prueba[y], p)
    metr.update({"auc_validacion": auc_val, "mejor_epoca": epocas, "epocas_seleccion": len(historial_sel),
                 "perdida_final": red_final.historial_[-1]["perdida"], "n_parametros": red_final.n_parametros(),
                 "tiempo_seleccion_s": t_sel, "tiempo_entrenamiento_s": t_fit, "tiempo_prediccion_s": time.time() - t2,
                 "n_entrenamiento": muestra["positivos"] + muestra["negativos"],
                 "positivos_entrenamiento": muestra["positivos"]})
    params = parametros_comunes(cfg, pregunta, v, e, red_final, epocas, {
        "version": version, "descripcion": v["descripcion"], "estaticas": v["estaticas"],
        "secuencia": f"últimos {n_meses} meses del crédito, un paso por mes", "n_pasos": n_meses,
        "variables_mes": ",".join(MES_NUM + MES_BANDERAS), "variables_estaticas": ",".join(sum(ESTATICAS[v["estaticas"]], [])),
        "r": muestra["r"], "frac_positivos": muestra["frac_positivos"], "frac_negativos": muestra["frac_negativos"]})
    cerrar(cfg, pregunta, version, v, params, metr, modelo, pipeline, ejemplo, historial_sel, red_final.historial_)


# ---------- comparación ----------

COMPARAR = {"p1": [("red densa", "red"), ("red recurrente", "rnn")],
            "p2": [("k vecinos", "knn"), ("red recurrente", "rnn")]}


def comparar():
    """Tabla de todas las versiones exportadas; el ganador de cada pregunta se elige por el AUC de validación."""
    filas = []
    for pregunta, familias in COMPARAR.items():
        for nombre, carpeta in familias:
            for metr in sorted((base.RAIZ / "modelos" / carpeta / pregunta).glob("v*/metricas.json")):
                m = json.loads(metr.read_text(encoding="utf-8"))
                prm = json.loads((metr.parent / "parametros.json").read_text(encoding="utf-8"))
                filas.append({"pregunta": pregunta, "modelo": nombre, "version": metr.parent.name,
                              "descripcion": prm.get("descripcion", ""), "auc_validacion": m.get("auc_validacion"),
                              "auc_roc_prueba": m["auc_roc"], "auc_pr_prueba": m["auc_pr"],
                              "captura_decil": m["captura_decil"]})
    if not filas:
        raise SystemExit("No hay modelos exportados en modelos/{red,knn,rnn}/")
    tabla = pd.DataFrame(filas).sort_values(["pregunta", "auc_validacion"], ascending=[True, False])
    pd.set_option("display.width", 200)
    for pregunta, t in tabla.groupby("pregunta"):
        print(f"\n=== {pregunta} (ordenado por AUC de validación) ===")
        print(t.drop(columns="pregunta").round(4).to_string(index=False))
        g = t.iloc[0]
        print(f"Ganador en validación: {g['modelo']} {g['version']} ({g['descripcion']}), "
              f"AUC de validación {g['auc_validacion']:.4f}, AUC de prueba {g['auc_roc_prueba']:.4f}")
    ruta = base.RAIZ / "modelos" / "comparacion_rnn.csv"
    tabla.to_csv(ruta, index=False)
    print(f"\nTabla guardada en {ruta.relative_to(base.RAIZ)}")
    return tabla


def main():
    parser = argparse.ArgumentParser(description="Redes recurrentes de las preguntas 1 y 2")
    parser.add_argument("--pregunta", choices=["p1", "p2"])
    parser.add_argument("--accion", choices=["entrenar", "comparar"], default="entrenar")
    parser.add_argument("--version", nargs="+", default=[], help="por defecto, todas las de la pregunta")
    parser.add_argument("--sin-referencia", action="store_true", help="no registrar la referencia del puntaje")
    args = parser.parse_args()
    if args.accion == "comparar":
        comparar()
        return
    if not args.pregunta:
        parser.error("--pregunta es obligatoria para entrenar")
    cfg = cargar_config()
    versiones = args.version or list(cfg["propia"]["versiones"][args.pregunta])
    inicio = time.time()
    if not args.sin_referencia:
        base.correr_referencia(cfg, args.pregunta)
    for version in versiones:
        (correr_p1 if args.pregunta == "p1" else correr_p2)(cfg, version)
    log(f"Terminó en {(time.time() - inicio) / 60:.1f} minutos")


if __name__ == "__main__":
    main()

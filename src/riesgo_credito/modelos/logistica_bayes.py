"""Regresión logística y naive Bayes para las preguntas 1 y 2.

Usan la misma partición temporal, el mismo muestreo con corrección de la probabilidad,
las mismas métricas y el mismo registro en MLflow que los modelos de k vecinos más
cercanos (riesgo_credito.modelos.knn), así que se comparan directamente con ellos y con
la referencia del puntaje. Cada versión queda en MLflow y se exporta a modelos/<familia>/.

Uso, desde la raíz del repositorio:
    export MLFLOW_TRACKING_URI=http://107.22.35.29:8050
    PYTHONPATH=src python -m riesgo_credito.modelos.logistica_bayes --familia logistica --pregunta p1
    PYTHONPATH=src python -m riesgo_credito.modelos.logistica_bayes --familia bayes --pregunta p2 --version v1
    PYTHONPATH=src python -m riesgo_credito.modelos.logistica_bayes --todo
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder

from riesgo_credito.modelos import knn as base
from riesgo_credito.modelos.knn_modelo import ModeloKNN as ModeloCorregido
from riesgo_credito.modelos.knn_modelo import preparar, probabilidad

CONFIG = Path(os.environ.get("LOGISTICA_BAYES_CONFIG", base.RAIZ / "config" / "modelos_logistica_bayes.yaml"))
log = base.log


def cargar_config():
    cfg = base.cargar_config()  # partición, semilla y lotes de los modelos de k vecinos
    with open(CONFIG, encoding="utf-8") as f:
        propia = yaml.safe_load(f)
    for pregunta in ("p1", "p2"):
        cfg[pregunta]["max_entrenamiento"] = propia["max_entrenamiento"][pregunta]
    cfg["propia"] = propia
    return cfg


def config_familia(cfg, familia):
    """Copia de la configuración con los experimentos de la familia (base.registrar los lee de cfg['mlflow'])."""
    c = copy.deepcopy(cfg)
    c["mlflow"] = cfg["propia"]["mlflow"][familia]
    base.EXPORTAR = base.RAIZ / "modelos" / familia
    return c


def discretizador(n):
    argumentos = dict(n_bins=n, encode="onehot-dense", strategy="quantile", subsample=200_000, random_state=42)
    try:
        return KBinsDiscretizer(quantile_method="averaged_inverted_cdf", **argumentos)
    except TypeError:  # versiones de scikit-learn anteriores a 1.7
        return KBinsDiscretizer(**argumentos)


def crear_preparacion(tipo, numericas, banderas, categoricas, tramos):
    if tipo == "estandar":
        return base.crear_preparacion(numericas + banderas, categoricas)
    pasos = [("num", Pipeline([("imputar", SimpleImputer(strategy="median")), ("tramos", discretizador(tramos))]), numericas)]
    if banderas:
        pasos.append(("banderas", "passthrough", banderas))
    if categoricas:
        pasos.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas))
    return Pipeline([("prep", ColumnTransformer(pasos)), ("float32", FunctionTransformer(np.float32))])


def tipo_clasificador(familia, preparacion):
    if familia == "logistica":
        return "logistica"
    return "gaussiano" if preparacion == "estandar" else "bernoulli"


def crear_clasificador(tipo, valor):
    if tipo == "logistica":
        return LogisticRegression(C=valor, max_iter=2000)
    if tipo == "gaussiano":
        return GaussianNB(var_smoothing=valor)
    return BernoulliNB(alpha=valor, binarize=None)  # las columnas de "tramos" ya son 0 o 1


NOMBRES = {"logistica": "regresión logística", "gaussiano": "naive Bayes gaussiano",
           "bernoulli": "naive Bayes de Bernoulli"}


def coeficientes(pipeline):
    nombres = pipeline.named_steps["prep"].get_feature_names_out()
    tabla = pd.DataFrame({"variable": nombres, "coeficiente": pipeline.named_steps["clf"].coef_[0]})
    return tabla.reindex(tabla["coeficiente"].abs().sort_values(ascending=False).index)


def correr_referencia(cfg_global, familia, pregunta):
    cfg = config_familia(cfg_global, familia)
    base.correr_referencia(cfg, pregunta)


def correr_version(cfg_global, familia, pregunta, version):
    cfg = config_familia(cfg_global, familia)
    base.preparar_mlflow(cfg, pregunta)
    v = cfg["propia"]["versiones"][familia][pregunta][version]
    numericas, banderas, categoricas = base.CONJUNTOS[v["variables"]]
    y, semilla, lote = cfg[pregunta]["etiqueta"], cfg["semilla"], cfg["lote_prediccion"]
    maximo, tramos = cfg[pregunta]["max_entrenamiento"], cfg["propia"]["tramos"]
    tipo = tipo_clasificador(familia, v["preparacion"])
    parametro, valores = cfg["propia"]["candidatos"][tipo]["parametro"], cfg["propia"]["candidatos"][tipo]["valores"]

    log(f"{familia} {pregunta} {version} ({v['descripcion']}): selección de {parametro}")
    t0 = time.time()
    sel, info = base.datos(cfg, pregunta, "seleccion")
    sel, _ = base.muestrear(sel, y, v["proporcion"], maximo, semilla, **info)
    val, _ = base.datos(cfg, pregunta, "validacion")
    prep = crear_preparacion(v["preparacion"], numericas, banderas, categoricas, tramos)
    A = prep.fit_transform(preparar(sel, numericas, banderas, categoricas))
    B = prep.transform(preparar(val, numericas, banderas, categoricas))
    aucs = {}
    for valor in valores:
        clf = crear_clasificador(tipo, valor).fit(A, sel[y].to_numpy())
        aucs[valor] = roc_auc_score(val[y], clf.predict_proba(B)[:, 1])
    elegido = max(aucs, key=aucs.get)
    t_sel = time.time() - t0
    log("Área bajo la curva en validación: " + ", ".join(f"{parametro}={k}: {a:.4f}" for k, a in aucs.items())
        + f" -> {parametro} = {elegido}")
    del sel, val, A, B

    log("Entrenamiento final")
    ent, info = base.datos(cfg, pregunta, "entrenamiento")
    ent, muestra = base.muestrear(ent, y, v["proporcion"], maximo, semilla, **info)
    modelo = ModeloCorregido(numericas, banderas, categoricas, muestra["r"])
    modelo.pipeline = Pipeline(crear_preparacion(v["preparacion"], numericas, banderas, categoricas, tramos).steps
                               + [("clf", crear_clasificador(tipo, elegido))])
    t1 = time.time()
    modelo.pipeline.fit(preparar(ent, numericas, banderas, categoricas), ent[y].to_numpy())
    t_fit = time.time() - t1
    n_columnas = modelo.pipeline[:-1].transform(preparar(ent.head(1), numericas, banderas, categoricas)).shape[1]
    coef = coeficientes(modelo.pipeline) if tipo == "logistica" else None
    del ent

    log("Predicción sobre prueba")
    prueba, _ = base.datos(cfg, pregunta, "prueba")
    t2 = time.time()
    p = probabilidad(modelo.pipeline, preparar(prueba, numericas, banderas, categoricas), muestra["r"], lote)
    t_pred = time.time() - t2
    metr = base.metricas(prueba[y], p)
    metr.update({f"auc_validacion_{parametro}_{k}": a for k, a in aucs.items()})
    metr.update({"auc_validacion": aucs[elegido], "tiempo_seleccion_s": t_sel, "tiempo_entrenamiento_s": t_fit,
                 "tiempo_prediccion_s": t_pred, "n_entrenamiento": muestra["positivos"] + muestra["negativos"],
                 "positivos_entrenamiento": muestra["positivos"]})
    params = {"pregunta": pregunta, "version": version, "descripcion": v["descripcion"], "algoritmo": NOMBRES[tipo],
              "preparacion": v["preparacion"], "variables": v["variables"],
              "lista_variables": ",".join(numericas + banderas + categoricas), "n_columnas_codificadas": n_columnas,
              "parametro": parametro, parametro: elegido, "candidatos": ",".join(map(str, valores)),
              "proporcion_negativos": v["proporcion"], "r": muestra["r"], "frac_positivos": muestra["frac_positivos"],
              "frac_negativos": muestra["frac_negativos"],
              "periodo_seleccion": base.descripcion_etapa(cfg, pregunta, "seleccion"),
              "periodo_validacion": base.descripcion_etapa(cfg, pregunta, "validacion"),
              "periodo_entrenamiento": base.descripcion_etapa(cfg, pregunta, "entrenamiento"),
              "periodo_prueba": base.descripcion_etapa(cfg, pregunta, "prueba"),
              "regla_validacion": base.regla_evaluacion(cfg, pregunta, "validacion"),
              "regla_prueba": base.regla_evaluacion(cfg, pregunta, "prueba"), "semilla": semilla}
    if v["preparacion"] == "tramos":
        params["tramos"] = tramos
    log(f"Prueba: área bajo la curva {metr['auc_roc']:.4f}, precisión promedio {metr['auc_pr']:.4f}, captura del "
        f"decil superior {100 * metr['captura_decil']:.1f} %, probabilidad media {metr['prob_media']:.5f} frente a "
        f"tasa {metr['tasa_observada']:.5f}")
    columnas = (["loan_id", "period"] if pregunta == "p2" else ["loan_id"]) + numericas + banderas + categoricas
    ejemplo = base.entrada_json(prueba[columnas].sample(n=min(2000, len(prueba)), random_state=semilla),
                                numericas, banderas, categoricas)
    base.registrar(cfg, pregunta, f"{version}_{v['descripcion']}", params, metr, modelo, modelo.pipeline, ejemplo)

    if coef is not None:  # coeficientes de la regresión, ordenados por magnitud, en Git y en MLflow
        destino = base.EXPORTAR / pregunta / version
        ruta = destino / "coeficientes.csv"
        coef.to_csv(ruta, index=False)
        run_id = json.loads((destino / "parametros.json").read_text(encoding="utf-8"))["run_id"]
        mlflow.MlflowClient().log_artifact(run_id, str(ruta))


def main():
    parser = argparse.ArgumentParser(description="Regresión logística y naive Bayes de las preguntas 1 y 2")
    parser.add_argument("--familia", choices=["logistica", "bayes"])
    parser.add_argument("--pregunta", choices=["p1", "p2"])
    parser.add_argument("--version", nargs="+", default=[], help="por defecto, todas las de la familia y la pregunta")
    parser.add_argument("--sin-referencia", action="store_true", help="no registrar la referencia del puntaje")
    parser.add_argument("--todo", action="store_true", help="las dos familias y las dos preguntas")
    args = parser.parse_args()
    cfg = cargar_config()
    if args.todo:
        trabajos = [(f, p) for f in ("logistica", "bayes") for p in ("p1", "p2")]
    elif args.familia and args.pregunta:
        trabajos = [(args.familia, args.pregunta)]
    else:
        raise SystemExit("Indica --familia y --pregunta, o usa --todo")
    inicio = time.time()
    for familia, pregunta in trabajos:
        if not args.sin_referencia:
            correr_referencia(cfg, familia, pregunta)
        for version in (args.version or list(cfg["propia"]["versiones"][familia][pregunta])):
            correr_version(cfg, familia, pregunta, version)
    log(f"Terminó en {(time.time() - inicio) / 60:.1f} minutos")


if __name__ == "__main__":
    main()

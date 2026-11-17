"""Entrena la red neuronal de la pregunta 1 y la guarda dentro del paquete.

1. Con las cosechas de selección y validación elige el número de épocas (parada temprana).
2. Reentrena con todas las cosechas de entrenamiento durante esas épocas.
3. Evalúa en las cosechas de prueba y guarda el pipeline, la corrección r y las métricas.
4. Si MLFLOW_TRACKING_URI está definida, registra la ejecución en el experimento de config.yml.

Uso, desde package-src:  tox run -e train   (o  PYTHONPATH=. python modelo_riesgo/train_pipeline.py)
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite correrlo como script

from modelo_riesgo import __version__ as _version  # noqa: E402
from modelo_riesgo.config.core import TRAINED_MODEL_DIR, config  # noqa: E402
from modelo_riesgo.pipeline import corregir, crear_pipeline  # noqa: E402
from modelo_riesgo.processing.data_manager import load_dataset, muestrear, save_pipeline  # noqa: E402


def log(mensaje):
    print(f"[{time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


def metricas(y, p) -> dict:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    n_top = math.ceil(0.1 * len(y))
    top = np.argsort(-p, kind="stable")[:n_top]
    captura = y[top].sum() / y.sum()
    return {"auc_roc": float(roc_auc_score(y, p)), "auc_pr": float(average_precision_score(y, p)),
            "captura_decil": float(captura), "lift_decil": float(captura / (n_top / len(y))),
            "tasa_observada": float(y.mean()), "prob_media": float(p.mean()), "n_prueba": int(len(y)),
            "positivos_prueba": int(y.sum())}


def registrar_mlflow(params, metr, historial, ruta_modelo):
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        log("MLFLOW_TRACKING_URI no está definida: no se registra en MLflow")
        return
    try:
        import mlflow
    except ImportError:
        log("mlflow no está instalado: no se registra en MLflow")
        return
    mlflow.set_experiment(config.experimento_mlflow)
    with mlflow.start_run(run_name=f"paquete_v{_version}"):
        mlflow.set_tags({"origen": "package-src", "version_paquete": _version})
        mlflow.log_params(params)
        mlflow.log_metrics(metr)
        for fila in historial:
            pasos = {"epoca_perdida_seleccion": fila["perdida"]}
            if "auc_validacion" in fila:
                pasos["epoca_auc_validacion"] = fila["auc_validacion"]
                pasos["epoca_perdida_validacion"] = fila["perdida_validacion"]
            mlflow.log_metrics(pasos, step=fila["epoca"])
        mlflow.log_artifact(str(ruta_modelo))
    log(f"Registrado en MLflow, experimento {config.experimento_mlflow}, ejecución paquete_v{_version}")


def run_training() -> dict:
    y = config.etiqueta
    log(f"Versión {_version}: capas {config.red.capas}, dropout {config.red.dropout}, variables {config.variables}")

    log(f"Selección del número de épocas: {config.seleccion[0]}-{config.seleccion[-1]}, "
        f"validación {config.validacion[0]}-{config.validacion[-1]}")
    sel, _ = muestrear(load_dataset(config.seleccion), config.semilla)
    val = load_dataset(config.validacion)
    if len(val) > config.max_validacion:
        val = val.sample(n=config.max_validacion, random_state=config.semilla)
    pipe_sel = crear_pipeline(config.red.epocas_max, config.red.paciencia)
    prep = pipe_sel[:-1].fit(sel[config.variables], sel[y])
    red_sel = pipe_sel[-1].fit(prep.transform(sel[config.variables]), sel[y].to_numpy(),
                               prep.transform(val[config.variables]), val[y].to_numpy())
    epocas, historial = red_sel.mejor_epoca_, red_sel.historial_
    log(f"Mejor época {epocas}: área bajo la curva en validación {red_sel.mejor_auc_validacion_:.4f}")
    del sel, val

    log(f"Entrenamiento final: {config.entrenamiento[0]}-{config.entrenamiento[-1]}, {epocas} épocas")
    ent, r = muestrear(load_dataset(config.entrenamiento), config.semilla)
    pipeline = crear_pipeline(epocas).fit(ent[config.variables], ent[y].to_numpy())
    n_entrenamiento, positivos = len(ent), int(ent[y].sum())
    del ent

    log(f"Prueba: {config.prueba[0]}-{config.prueba[-1]}")
    prueba = load_dataset(config.prueba)
    p = corregir(pipeline.predict_proba(prueba[config.variables])[:, 1], r)
    metr = metricas(prueba[y], p)
    metr.update({"auc_validacion": red_sel.mejor_auc_validacion_, "mejor_epoca": epocas,
                 "n_entrenamiento": n_entrenamiento, "positivos_entrenamiento": positivos,
                 "n_parametros": pipeline[-1].n_parametros()})
    log(f"Prueba: área bajo la curva {metr['auc_roc']:.4f}, precisión promedio {metr['auc_pr']:.4f}, "
        f"captura del decil superior {100 * metr['captura_decil']:.1f} %")

    modelo = {"pipeline": pipeline, "r": r, "version": _version, "variables": config.variables, "metricas": metr}
    ruta = save_pipeline(modelo=modelo)
    (TRAINED_MODEL_DIR / "metricas.json").write_text(json.dumps(metr, indent=2), encoding="utf-8")
    log(f"Modelo guardado en {ruta}")

    params = {"capas": "-".join(map(str, config.red.capas)), "n_capas_ocultas": len(config.red.capas),
              "dropout": config.red.dropout, "epocas": epocas, "tasa_aprendizaje": config.red.tasa_aprendizaje,
              "l2": config.red.l2, "tam_lote": config.red.tam_lote, "variables": ",".join(config.variables),
              "proporcion_negativos": config.proporcion_negativos, "r": r,
              "periodo_entrenamiento": f"{config.entrenamiento[0]}-{config.entrenamiento[-1]}",
              "periodo_prueba": f"{config.prueba[0]}-{config.prueba[-1]}", "version_paquete": _version}
    registrar_mlflow(params, metr, historial, ruta)
    return metr


if __name__ == "__main__":
    run_training()

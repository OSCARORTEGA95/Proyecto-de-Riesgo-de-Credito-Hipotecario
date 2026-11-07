"""Registra en un servidor de MLflow los modelos KNN exportados en esta carpeta, sin reentrenarlos.

Uso, desde la raíz del repositorio:
    export MLFLOW_TRACKING_URI=http://<IP>:<PUERTO>
    python modelos/knn/registrar_en_mlflow.py
"""
import json
import os
from pathlib import Path

import mlflow

CARPETA = Path(__file__).resolve().parent

if not os.environ.get("MLFLOW_TRACKING_URI"):
    raise SystemExit("Falta MLFLOW_TRACKING_URI")

for metricas_json in sorted(CARPETA.glob("p*/*/metricas.json")):
    carpeta = metricas_json.parent
    params = json.loads((carpeta / "parametros.json").read_text(encoding="utf-8"))
    metricas = json.loads(metricas_json.read_text(encoding="utf-8"))
    mlflow.set_experiment(params.pop("experimento"))
    run_name = params.pop("run_name")
    run_original = params.pop("run_id")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"run_original": run_original, "origen": "modelos/knn"})
        mlflow.log_params(params)
        mlflow.log_metrics(metricas)
        if (carpeta / "modelo" / "MLmodel").exists():
            mlflow.log_artifacts(str(carpeta / "modelo"), artifact_path="modelo")
    print(f"Registrado {run_name}")

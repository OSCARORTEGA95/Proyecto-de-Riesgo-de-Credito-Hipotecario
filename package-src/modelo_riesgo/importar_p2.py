"""Incorpora al paquete el modelo de la pregunta 2.

El k vecinos se entrenó y se evaluó con src/riesgo_credito/modelos/knn.py, que registró la
ejecución en MLflow (experimento knn-p2) y la exportó en modelos/knn/p2/<versión>. Este script
toma de esa carpeta el pipeline, la corrección del muestreo (r) y las métricas, verifica que
prediga sobre el ejemplo guardado y lo deja en trained_models/ para que viaje en el .whl.

Uso, desde package-src:  tox run -e p2   (o  PYTHONPATH=. python modelo_riesgo/importar_p2.py)
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite correrlo como script

from modelo_riesgo.config.core import ROOT, TRAINED_MODEL_DIR, config  # noqa: E402


def importar() -> Path:
    origen = Path(config.p2.modelo_origen)
    origen = origen if origen.is_absolute() else (ROOT / origen).resolve()
    parametros = json.loads((origen / "parametros.json").read_text(encoding="utf-8"))
    metricas = json.loads((origen / "metricas.json").read_text(encoding="utf-8"))
    pipeline = joblib.load(origen / "modelo" / "artifacts" / "pipeline.joblib")

    variables = parametros["lista_variables"].split(",")
    if sorted(variables) != sorted(config.p2.variables):
        raise SystemExit(f"Las variables de {origen.name} no coinciden con config.yml: {variables}")

    modelo = {"pipeline": pipeline, "r": float(parametros["r"]), "k": int(parametros["k"]),
              "descripcion": config.p2.descripcion, "experimento": parametros["experimento"],
              "run_id": parametros["run_id"], "variables": config.p2.variables, "metricas": metricas,
              "periodo_entrenamiento": parametros["periodo_entrenamiento"],
              "periodo_prueba": parametros["periodo_prueba"]}
    destino = TRAINED_MODEL_DIR / config.p2.archivo
    joblib.dump(modelo, destino, compress=3)
    (TRAINED_MODEL_DIR / "metricas_p2.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")

    # Verificación: el modelo empaquetado predice sobre el ejemplo que guardó MLflow
    from modelo_riesgo import p2
    p2._modelo_p2 = None
    ejemplo = json.loads((origen / "modelo" / "input_example.json").read_text(encoding="utf-8"))
    datos = pd.DataFrame(ejemplo["data"], columns=ejemplo["columns"])
    resultado = p2.make_prediction_p2(input_data=datos)
    if resultado["errors"]:
        raise SystemExit(f"El modelo importado no predice el ejemplo: {resultado['errors']}")

    print(f"Modelo de la pregunta 2 guardado en {destino}")
    print(f"  {config.p2.descripcion}: k = {modelo['k']}, r = {modelo['r']:.6f}, run_id {modelo['run_id']}")
    print(f"  AUC validación {metricas['auc_validacion']:.4f}, AUC prueba {metricas['auc_roc']:.4f}, "
          f"captura del decil superior {100 * metricas['captura_decil']:.1f} %")
    print(f"  Ejemplo: {[round(p, 6) for p in resultado['predictions']]}")
    return destino


if __name__ == "__main__":
    importar()

"""Prueba del paquete instalado (Taller 5, parte 3, paso 18).

Uso, desde esta carpeta y con el .whl instalado:
    python3 test-package.py
"""
import json
from pathlib import Path

import pandas as pd

import modelo_riesgo
from modelo_riesgo.predict import make_prediction

solicitudes = pd.DataFrame(json.loads((Path(__file__).parent / "solicitudes_prueba.json").read_text(encoding="utf-8")))
resultado = make_prediction(input_data=solicitudes)

print(f"Paquete modelo_riesgo {modelo_riesgo.__version__} instalado en {Path(modelo_riesgo.__file__).parent}")
print(f"Errores: {resultado['errors']}")
solicitudes["prob_incumplir_12m"] = resultado["predictions"]
print(solicitudes.to_string(index=False))

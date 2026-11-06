
from pathlib import Path
import os

import pandas as pd
import mlflow


# ==========================================
# 1. CONFIGURACION
# ==========================================

RAIZ = Path(__file__).resolve().parents[4]
RESULTADOS = RAIZ / "resultados" / "arbol_p1"

EXPERIMENTO = "Riesgo_Hipotecario_P1_Arbol_Ana"

# Usar el servidor configurado en PowerShell
servidor = os.environ.get("MLFLOW_TRACKING_URI")

if not servidor:
    raise ValueError(
        "Primero configura MLFLOW_TRACKING_URI en PowerShell."
    )

mlflow.set_tracking_uri(servidor)
mlflow.set_experiment(EXPERIMENTO)

print("Servidor MLflow:", mlflow.get_tracking_uri())
print("Experimento:", EXPERIMENTO)


# ==========================================
# 2. COMPARACION DE ARBOLES
# ==========================================

archivo = RESULTADOS / "comparacion_validacion_2023.csv"
arboles = pd.read_csv(archivo)

for _, fila in arboles.iterrows():

    nombre = str(fila["Modelo"])

    with mlflow.start_run(
        run_name=f"Arbol_{nombre}_Validacion_2023"
    ):

        mlflow.set_tags({
            "pregunta": "P1",
            "algoritmo": "DecisionTreeClassifier",
            "tipo_registro": "Resultado historico",
            "entrenamiento": "2022",
            "validacion": "2023",
            "origen": archivo.name
        })

        mlflow.log_params({
            "max_depth": int(fila["Profundidad"]),
            "min_samples_leaf": int(fila["Min_hoja"]),
            "random_state": 42,
            "umbral": 0.02
        })

        mlflow.log_metrics({
            "average_precision": float(fila["AP"]),
            "roc_auc": float(fila["ROC_AUC"]),
            "precision": float(fila["Precision_2pct"]),
            "recall": float(fila["Recall_2pct"]),
            "alertas": float(fila["Alertas_2pct"])
        })

    print("Registrado:", nombre)


# ==========================================
# 3. COMPARACION DE PERIODOS HISTORICOS
# ==========================================

archivo = RESULTADOS / "comparacion_periodos_2023.csv"
periodos = pd.read_csv(archivo)

for _, fila in periodos.iterrows():

    nombre = str(fila["Escenario"])

    with mlflow.start_run(
        run_name=f"Periodo_{nombre}_Validacion_2023"
    ):

        mlflow.set_tags({
            "pregunta": "P1",
            "algoritmo": "DecisionTreeClassifier",
            "tipo_registro": "Resultado historico",
            "validacion": "2023",
            "origen": archivo.name,
            "muestreo": "Hasta 150000 registros por trimestre"
        })

        mlflow.log_params({
            "escenario": nombre,
            "max_depth": 8,
            "min_samples_leaf": 1000,
            "random_state": 42,
            "registros_entrenamiento": int(
                fila["Registros_train"]
            ),
            "umbral": 0.02
        })

        mlflow.log_metrics({
            "tasa_incumplimiento_train": float(
                fila["Tasa_train"]
            ),
            "average_precision": float(fila["AP"]),
            "roc_auc": float(fila["ROC_AUC"]),
            "precision": float(fila["Precision_2pct"]),
            "recall": float(fila["Recall_2pct"]),
            "alertas": float(fila["Alertas_2pct"])
        })

    print("Registrado periodo:", nombre)


# ==========================================
# 4. EVALUACION FINAL 2024
# ==========================================

archivo = RESULTADOS / "metricas_prueba_2024.csv"
final = pd.read_csv(archivo).iloc[0]

with mlflow.start_run(
    run_name="Arbol_Flexible_Prueba_2024"
):

    mlflow.set_tags({
        "pregunta": "P1",
        "algoritmo": "DecisionTreeClassifier",
        "tipo_registro": "Resultado historico",
        "entrenamiento": "2022-2023",
        "prueba": "2024",
        "origen": archivo.name
    })

    mlflow.log_params({
        "max_depth": int(final["max_depth"]),
        "min_samples_leaf": int(
            final["min_samples_leaf"]
        ),
        "random_state": 42,
        "umbral": float(final["umbral"])
    })

    mlflow.log_metrics({
        "average_precision": float(
            final["average_precision"]
        ),
        "roc_auc": float(final["roc_auc"]),
        "brier_score": float(final["brier_score"]),
        "precision": float(final["precision"]),
        "recall": float(final["recall"]),
        "f1": float(final["f1"]),
        "alertas": float(final["alertas"])
    })

    # Guardar mÃ©tricas y matriz de confusiÃ³n
    mlflow.log_artifact(str(archivo))

    matriz = RESULTADOS / "matriz_confusion_2024.csv"
    mlflow.log_artifact(str(matriz))

    # Guardar el modelo entrenado
    modelo = RESULTADOS / "arbol_decision_p1.joblib"

    if modelo.exists():
        mlflow.log_artifact(
            str(modelo),
            artifact_path="modelo"
        )

print("\n================================")
print("REGISTRO MLflow COMPLETADO")
print("3 configuraciones del arbol")
print("3 escenarios historicos")
print("1 evaluacion final de 2024")
print("Total: 7 ejecuciones registradas")
print("================================")


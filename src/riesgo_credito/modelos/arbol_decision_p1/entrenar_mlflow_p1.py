
from pathlib import Path
import os
import gc

import joblib
import numpy as np
import pandas as pd
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. CONFIGURACION
# ==========================================

RAIZ = Path(__file__).resolve().parents[4]
RUTA = RAIZ / "data" / "modelado" / "p1"
SALIDA = RAIZ / "resultados" / "arbol_p1"

EXPERIMENTO = "Riesgo_Hipotecario_P1_Arbol_Ana"

NUMERICAS = [
    "credit_score", "dti", "oltv", "ocltv",
    "orig_rate", "orig_upb", "orig_term",
    "mi_pct", "num_borrowers", "num_units"
]

CATEGORICAS = [
    "purpose", "occupancy", "channel",
    "first_time_homebuyer", "prop_type",
    "state", "valuation_method"
]

COLUMNAS = NUMERICAS + CATEGORICAS + ["y_p1"]

PARAMETROS = {
    "max_depth": 8,
    "min_samples_leaf": 1000,
    "random_state": 42
}

UMBRAL = 0.02


# ==========================================
# 2. CONECTAR CON MLFLOW
# ==========================================

servidor = os.environ.get("MLFLOW_TRACKING_URI")

if not servidor:
    raise ValueError(
        "Configura primero MLFLOW_TRACKING_URI "
        "en la terminal de PowerShell."
    )

mlflow.set_tracking_uri(servidor)
mlflow.set_experiment(EXPERIMENTO)

print("Servidor MLflow:", servidor)


# ==========================================
# 3. CARGAR Y LIMPIAR DATOS
# ==========================================

def cargar_datos(anio):

    archivos = sorted(
        RUTA.glob(f"trimestre={anio}Q*/*.parquet")
    )

    if len(archivos) != 4:
        raise FileNotFoundError(
            f"Faltan trimestres del anio {anio}"
        )

    partes = [
        pd.read_parquet(archivo, columns=COLUMNAS)
        for archivo in archivos
    ]

    df = pd.concat(partes, ignore_index=True)

    df = df[df["y_p1"].isin([0, 1])].copy()

    rangos = {
        "credit_score": (300, 850),
        "dti": (0, 100),
        "oltv": (0, 200),
        "ocltv": (0, 200),
        "orig_rate": (0.001, 30),
        "orig_term": (1, 600),
        "mi_pct": (0, 100),
        "num_borrowers": (1, 20),
        "num_units": (1, 20)
    }

    for columna in NUMERICAS:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        ).astype("float32")

        if columna in rangos:

            minimo, maximo = rangos[columna]

            df.loc[
                ~df[columna].between(minimo, maximo),
                columna
            ] = np.nan

    df.loc[df["orig_upb"] <= 0, "orig_upb"] = np.nan

    for columna in CATEGORICAS:

        if columna == "valuation_method":

            df[columna] = (
                pd.to_numeric(
                    df[columna],
                    errors="coerce"
                )
                .astype("Int64")
                .astype("string")
                .fillna("Sin dato")
            )

        else:

            df[columna] = (
                df[columna]
                .astype("string")
                .fillna("Sin dato")
            )

    df["y_p1"] = df["y_p1"].astype("int8")

    return df


# ==========================================
# 4. INICIAR EXPERIMENTO
# ==========================================

with mlflow.start_run(
    run_name="Entrenamiento_Directo_Flexible_2022_2023"
) as ejecucion:

    mlflow.set_tags({
        "pregunta": "P1",
        "algoritmo": "DecisionTreeClassifier",
        "tipo_registro": "Entrenamiento directo",
        "entrenamiento": "2022",
        "validacion": "2023"
    })

    mlflow.log_params(PARAMETROS)

    mlflow.log_params({
        "umbral": UMBRAL,
        "periodo_train": "2022",
        "periodo_validacion": "2023"
    })

    print("\nCargando entrenamiento 2022...")
    train = cargar_datos(2022)

    print("Cargando validacion 2023...")
    val = cargar_datos(2023)

    X_train = train[NUMERICAS + CATEGORICAS]
    y_train = train["y_p1"]

    X_val = val[NUMERICAS + CATEGORICAS]
    y_val = val["y_p1"]

    print("Registros entrenamiento:", len(y_train))
    print("Registros validacion:", len(y_val))

    mlflow.log_metrics({
        "registros_train": len(y_train),
        "registros_validacion": len(y_val),
        "prevalencia_validacion": float(y_val.mean())
    })

    # ======================================
    # 5. PREPROCESAMIENTO
    # ======================================

    preprocesamiento = ColumnTransformer(
        transformers=[
            (
                "numericas",
                SimpleImputer(strategy="median"),
                NUMERICAS
            ),
            (
                "categoricas",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32
                ),
                CATEGORICAS
            )
        ],
        sparse_threshold=1.0
    )

    # ======================================
    # 6. ENTRENAR ARBOL
    # ======================================

    arbol = DecisionTreeClassifier(
        **PARAMETROS
    )

    modelo = Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("arbol", arbol)
    ])

    print("\nEntrenando arbol con MLflow activo...")

    modelo.fit(X_train, y_train)

    print("Entrenamiento completado.")

    # ======================================
    # 7. VALIDAR CON 2023
    # ======================================

    probabilidades = modelo.predict_proba(X_val)[:, 1]
    predicciones = (probabilidades >= UMBRAL).astype(int)

    metricas = {
        "average_precision": average_precision_score(
            y_val, probabilidades
        ),
        "roc_auc": roc_auc_score(
            y_val, probabilidades
        ),
        "precision": precision_score(
            y_val, predicciones, zero_division=0
        ),
        "recall": recall_score(
            y_val, predicciones, zero_division=0
        ),
        "f1": f1_score(
            y_val, predicciones, zero_division=0
        ),
        "alertas": int(predicciones.sum())
    }

    mlflow.log_metrics(metricas)

    matriz = confusion_matrix(
        y_val, predicciones, labels=[0, 1]
    )

    mlflow.log_dict(
        {
            "matriz_confusion": matriz.tolist(),
            "filas": ["Real_0", "Real_1"],
            "columnas": ["Predicho_0", "Predicho_1"]
        },
        "evaluacion/matriz_confusion_2023.json"
    )

    mlflow.log_params({
        "profundidad_real": modelo.named_steps[
            "arbol"
        ].get_depth(),
        "numero_hojas": modelo.named_steps[
            "arbol"
        ].get_n_leaves()
    })

    # ======================================
    # 8. GUARDAR MODELO Y CODIGO
    # ======================================

    SALIDA.mkdir(parents=True, exist_ok=True)

    archivo_modelo = (
        SALIDA / "arbol_mlflow_directo_2023.joblib"
    )

    joblib.dump(
        modelo,
        archivo_modelo,
        compress=3
    )

    mlflow.log_artifact(
        str(archivo_modelo),
        artifact_path="modelo"
    )

    mlflow.log_artifact(
        str(Path(__file__).resolve()),
        artifact_path="codigo"
    )

    # ======================================
    # 9. MOSTRAR RESULTADOS
    # ======================================

    print("\n====== VALIDACION 2023 ======")

    for nombre, valor in metricas.items():
        print(nombre, ":", round(float(valor), 6))

    print("\nMatriz de confusion:")
    print(matriz)

    print("\nEjecucion MLflow:", ejecucion.info.run_id)
    print("Registro completado.")

    del train, val, X_train, X_val
    gc.collect()


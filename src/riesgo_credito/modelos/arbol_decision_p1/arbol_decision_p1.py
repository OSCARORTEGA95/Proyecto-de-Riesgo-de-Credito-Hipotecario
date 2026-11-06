
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)



# 1. CONFIGURACION


RUTA = Path("data/modelado/p1")

VARIABLE_OBJETIVO = "y_p1"

NUMERICAS = [
    "credit_score",
    "dti",
    "oltv",
    "ocltv",
    "orig_rate",
    "orig_upb",
    "orig_term",
    "mi_pct",
    "num_borrowers",
    "num_units"
]

CATEGORICAS = [
    "purpose",
    "occupancy",
    "channel",
    "first_time_homebuyer",
    "prop_type",
    "state",
    "valuation_method"
]

COLUMNAS = NUMERICAS + CATEGORICAS + [VARIABLE_OBJETIVO]



# 2. CARGAR LOS DATOS


def cargar_datos(anio):

    archivos = sorted(
        RUTA.glob(f"trimestre={anio}Q*/*.parquet")
    )

    if len(archivos) != 4:
        raise FileNotFoundError(
            f"Se esperaban 4 trimestres para {anio}. "
            f"Se encontraron {len(archivos)}."
        )

    partes = []

    for archivo in archivos:
        partes.append(
            pd.read_parquet(archivo, columns=COLUMNAS)
        )

    datos = pd.concat(partes, ignore_index=True)

    # Eliminar registros sin etiqueta vÃ¡lida
    datos = datos[
        datos[VARIABLE_OBJETIVO].isin([0, 1])
    ].copy()

    datos[VARIABLE_OBJETIVO] = (
        datos[VARIABLE_OBJETIVO].astype("int8")
    )

    return datos



# 3. PREPARAR VARIABLES


def preparar_variables(datos):

    # Rangos razonables para identificar
    # cÃ³digos especiales o valores imposibles
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

        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce"
        ).astype("float32")

        # Convertir valores fuera de rango en faltantes
        if columna in rangos:

            minimo, maximo = rangos[columna]

            datos.loc[
                ~datos[columna].between(minimo, maximo),
                columna
            ] = np.nan

    # El monto original debe ser positivo
    datos.loc[
        datos["orig_upb"] <= 0,
        "orig_upb"
    ] = np.nan

    # Preparar variables categÃ³ricas
    for columna in CATEGORICAS:

        datos[columna] = (
            datos[columna]
            .fillna("Sin dato")
            .astype(str)
        )

    return datos



# 4. CARGAR ENTRENAMIENTO Y VALIDACION


print("Cargando datos de entrenamiento (2022)...")

entrenamiento = cargar_datos(2022)
entrenamiento = preparar_variables(entrenamiento)

print("Cargando datos de validacion (2023)...")

validacion = cargar_datos(2023)
validacion = preparar_variables(validacion)

print("Registros entrenamiento:", len(entrenamiento))
print("Registros validacion:", len(validacion))

print(
    "Tasa incumplimiento entrenamiento:",
    round(entrenamiento["y_p1"].mean() * 100, 3),
    "%"
)

print(
    "Tasa incumplimiento validacion:",
    round(validacion["y_p1"].mean() * 100, 3),
    "%"
)



# 5. SEPARAR X E Y


X_train = entrenamiento[NUMERICAS + CATEGORICAS]
y_train = entrenamiento[VARIABLE_OBJETIVO]

X_val = validacion[NUMERICAS + CATEGORICAS]
y_val = validacion[VARIABLE_OBJETIVO]


# 6. PREPROCESAMIENTO


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


# 7. ARBOL DE DECISION


arbol = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=500,
    random_state=42
)

modelo = Pipeline([
    ("preprocesamiento", preprocesamiento),
    ("arbol", arbol)
])

print("\nEntrenando arbol de decision...")

modelo.fit(X_train, y_train)

print("Entrenamiento completado.")



# 8. EVALUACION EN 2023


probabilidades = modelo.predict_proba(X_val)[:, 1]

# Umbral provisional para revisar los resultados.
# No es una polÃ­tica de aprobaciÃ³n de crÃ©ditos.
umbral = 0.01

predicciones = (
    probabilidades >= umbral
).astype(int)

pr_auc = average_precision_score(
    y_val,
    probabilidades
)

roc_auc = roc_auc_score(
    y_val,
    probabilidades
)

precision = precision_score(
    y_val,
    predicciones,
    zero_division=0
)

recall = recall_score(
    y_val,
    predicciones,
    zero_division=0
)

f1 = f1_score(
    y_val,
    predicciones,
    zero_division=0
)

matriz = confusion_matrix(
    y_val,
    predicciones,
    labels=[0, 1]
)

print("\n========== RESULTADOS 2023 ==========")

print("PR-AUC:", round(pr_auc, 5))
print("ROC-AUC:", round(roc_auc, 5))

print("\nUmbral provisional:", umbral)

print("Precision:", round(precision, 5))
print("Recall:", round(recall, 5))
print("F1:", round(f1, 5))

print("\nMatriz de confusion:")
print(matriz)

print("\nProfundidad real:", modelo.named_steps["arbol"].get_depth())

print(
    "Numero de hojas:",
    modelo.named_steps["arbol"].get_n_leaves()
)

print("=====================================")


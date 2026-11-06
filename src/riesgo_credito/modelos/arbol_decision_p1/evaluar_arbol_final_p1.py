
from pathlib import Path
import gc

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
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


# ==========================================
# 2. PREPARAR LOS DATOS
# ==========================================

def cargar_datos(anios):

    partes = []

    for anio in anios:

        archivos = sorted(
            RUTA.glob(f"trimestre={anio}Q*/*.parquet")
        )

        if len(archivos) != 4:
            raise ValueError(
                f"Faltan trimestres del aÃ±o {anio}"
            )

        for archivo in archivos:
            partes.append(
                pd.read_parquet(
                    archivo,
                    columns=COLUMNAS
                )
            )

    df = pd.concat(partes, ignore_index=True)

    inicial = len(df)

    df = df[df["y_p1"].isin([0, 1])].copy()

    print("Registros sin etiqueta excluidos:",
          inicial - len(df))

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

    df.loc[
        df["orig_upb"] <= 0,
        "orig_upb"
    ] = np.nan

    for columna in CATEGORICAS:
        df[columna] = (
            df[columna]
            .fillna("Sin dato")
            .astype(str)
        )

    return df


# ==========================================
# 3. ENTRENAMIENTO FINAL: 2022 Y 2023
# ==========================================

print("\nCargando entrenamiento 2022-2023...")

train = cargar_datos([2022, 2023])

X_train = train[NUMERICAS + CATEGORICAS].copy()
y_train = train["y_p1"].astype("int8").copy()

print("Registros entrenamiento:", len(X_train))
print(
    "Incumplimiento entrenamiento:",
    round(y_train.mean() * 100, 3), "%"
)

del train
gc.collect()


# ==========================================
# 4. MODELO SELECCIONADO
# ==========================================

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

arbol = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=1000,
    random_state=42
)

modelo = Pipeline([
    ("preprocesamiento", preprocesamiento),
    ("arbol", arbol)
])

print("\nEntrenando modelo seleccionado...")

modelo.fit(X_train, y_train)

print("Entrenamiento finalizado.")

del X_train, y_train
gc.collect()


# ==========================================
# 5. PRUEBA FINAL: 2024
# ==========================================

print("\nCargando prueba 2024...")

test = cargar_datos([2024])

X_test = test[NUMERICAS + CATEGORICAS]
y_test = test["y_p1"].astype(int)

print("Registros prueba:", len(X_test))
print(
    "Incumplimiento prueba:",
    round(y_test.mean() * 100, 3), "%"
)

probabilidades = modelo.predict_proba(X_test)[:, 1]


# ==========================================
# 6. METRICAS DEL MODELO
# ==========================================

ap = average_precision_score(
    y_test, probabilidades
)

roc_auc = roc_auc_score(
    y_test, probabilidades
)

brier = brier_score_loss(
    y_test, probabilidades
)

# Umbral exploratorio fijado antes de usar 2024
umbral = 0.02

predicciones = (
    probabilidades >= umbral
).astype(int)

precision = precision_score(
    y_test, predicciones,
    zero_division=0
)

recall = recall_score(
    y_test, predicciones,
    zero_division=0
)

f1 = f1_score(
    y_test, predicciones,
    zero_division=0
)

matriz = confusion_matrix(
    y_test, predicciones,
    labels=[0, 1]
)

print("\n========== PRUEBA FINAL 2024 ==========")

print("Average Precision:", round(ap, 5))
print("ROC-AUC:", round(roc_auc, 5))
print("Brier Score:", round(brier, 6))
print("Prevalencia:", round(y_test.mean(), 5))

print("\nUmbral exploratorio: 2 %")
print("Precision:", round(precision, 5))
print("Recall:", round(recall, 5))
print("F1:", round(f1, 5))
print("Cantidad de alertas:", int(predicciones.sum()))

print("\nMatriz de confusion:")
print(matriz)

print(
    "\nProfundidad real:",
    modelo.named_steps["arbol"].get_depth()
)

print(
    "Numero de hojas:",
    modelo.named_steps["arbol"].get_n_leaves()
)

print("======================================")


# ==========================================
# 7. GUARDAR RESULTADOS
# ==========================================

SALIDA.mkdir(parents=True, exist_ok=True)

metricas = pd.DataFrame([{
    "modelo": "Arbol flexible",
    "entrenamiento": "2022-2023",
    "prueba": "2024",
    "max_depth": 8,
    "min_samples_leaf": 1000,
    "umbral": umbral,
    "average_precision": ap,
    "roc_auc": roc_auc,
    "brier_score": brier,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "alertas": int(predicciones.sum())
}])

metricas.to_csv(
    SALIDA / "metricas_prueba_2024.csv",
    index=False
)

pd.DataFrame(
    matriz,
    index=["Real_0", "Real_1"],
    columns=["Predicho_0", "Predicho_1"]
).to_csv(
    SALIDA / "matriz_confusion_2024.csv"
)

joblib.dump(
    modelo,
    SALIDA / "arbol_decision_p1.joblib",
    compress=3
)

print("\nModelo y resultados guardados en:")
print(SALIDA)

print("\nLa calibracion de probabilidades "
      "debe revisarse antes de utilizar "
      "el modelo para decisiones reales.")


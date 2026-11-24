
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score
)



# 1. CONFIGURACION


RUTA = Path("data/modelado/p1")

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



# 2. CARGAR DATOS


def cargar_datos(anio):

    archivos = sorted(
        RUTA.glob(f"trimestre={anio}Q*/*.parquet")
    )

    if len(archivos) != 4:
        raise ValueError(
            f"Faltan trimestres del aÃ±o {anio}"
        )

    partes = [
        pd.read_parquet(archivo, columns=COLUMNAS)
        for archivo in archivos
    ]

    df = pd.concat(partes, ignore_index=True)

    df = df[df["y_p1"].isin([0, 1])].copy()

    # Mantener las mismas reglas del modelo base
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
            df[columna], errors="coerce"
        ).astype("float32")

        if columna in rangos:
            minimo, maximo = rangos[columna]

            df.loc[
                ~df[columna].between(minimo, maximo),
                columna
            ] = np.nan

    df.loc[df["orig_upb"] <= 0, "orig_upb"] = np.nan

    for columna in CATEGORICAS:
        df[columna] = (
            df[columna]
            .fillna("Sin dato")
            .astype(str)
        )

    return df


print("Cargando entrenamiento 2022...")
train = cargar_datos(2022)

print("Cargando validacion 2023...")
val = cargar_datos(2023)

y_train = train["y_p1"].astype(int)
y_val = val["y_p1"].astype(int)



# 3. PREPARAR VARIABLES


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

print("Preparando variables...")

X_train = preprocesamiento.fit_transform(
    train[NUMERICAS + CATEGORICAS]
)

X_val = preprocesamiento.transform(
    val[NUMERICAS + CATEGORICAS]
)



# 4. CONFIGURACIONES A COMPARAR


candidatos = [
    ("Conservador", 4, 1000),
    ("Base", 6, 500),
    ("Flexible", 8, 1000)
]

resultados = []

for nombre, profundidad, hoja in candidatos:

    print(f"\nEntrenando modelo: {nombre}")

    arbol = DecisionTreeClassifier(
        max_depth=profundidad,
        min_samples_leaf=hoja,
        random_state=42
    )

    arbol.fit(X_train, y_train)

    probabilidades = arbol.predict_proba(X_val)[:, 1]

    # Se compara tambiÃ©n el umbral provisional del 2 %
    predicciones = (probabilidades >= 0.02).astype(int)

    resultados.append({
        "Modelo": nombre,
        "Profundidad": profundidad,
        "Min_hoja": hoja,
        "AP": average_precision_score(
            y_val, probabilidades
        ),
        "ROC_AUC": roc_auc_score(
            y_val, probabilidades
        ),
        "Precision_2pct": precision_score(
            y_val, predicciones,
            zero_division=0
        ),
        "Recall_2pct": recall_score(
            y_val, predicciones,
            zero_division=0
        ),
        "Alertas_2pct": int(predicciones.sum())
    })



# 5. MOSTRAR Y GUARDAR RESULTADOS

tabla = pd.DataFrame(resultados)

tabla = tabla.sort_values(
    "AP", ascending=False
)

print("\n========== COMPARACION 2023 ==========")

print(tabla.round(5).to_string(index=False))

print("\nMejor configuracion por AP:")
print(tabla.iloc[0]["Modelo"])

print("\nNota: 2024 permanece reservado para prueba final.")

salida = Path("resultados/arbol_p1")
salida.mkdir(parents=True, exist_ok=True)

tabla.to_csv(
    salida / "comparacion_validacion_2023.csv",
    index=False
)

print("\nResultados guardados en:", salida)


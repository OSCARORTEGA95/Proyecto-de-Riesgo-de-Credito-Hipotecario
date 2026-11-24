
from pathlib import Path
import gc

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
    precision_score,
    recall_score
)


# ==========================================
# 1. CONFIGURACION
# ==========================================

RAIZ = Path(__file__).resolve().parents[4]
RUTA = RAIZ / "data" / "modelado" / "p1"
SALIDA = RAIZ / "resultados" / "arbol_p1"

LIMITE_POR_TRIMESTRE = 150000

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

COLUMNAS = NUMERICAS + CATEGORICAS + ["y_p1"]


# ==========================================
# 2. CARGAR TRIMESTRES
# ==========================================

def cargar_trimestres(trimestres, limite=None):

    partes = []

    for anio, trimestre in trimestres:

        carpeta = RUTA / f"trimestre={anio}Q{trimestre}"
        archivos = sorted(carpeta.rglob("*.parquet"))

        if not archivos:
            raise FileNotFoundError(
                f"No se encontraron datos en {carpeta}"
            )

        datos = pd.concat(
            [
                pd.read_parquet(
                    archivo,
                    columns=COLUMNAS
                )
                for archivo in archivos
            ],
            ignore_index=True
        )

        if limite is not None and len(datos) > limite:
            datos = datos.sample(
                n=limite,
                random_state=42
            )

        partes.append(datos)

    df = pd.concat(partes, ignore_index=True)

    df = df[df["y_p1"].isin([0, 1])].copy()

    return df


# ==========================================
# 3. LIMPIEZA DE VARIABLES
# ==========================================

def limpiar(datos):

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

    # Variables numericas
    for columna in NUMERICAS:

        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce"
        ).astype("float32")

        if columna in rangos:

            minimo, maximo = rangos[columna]

            datos.loc[
                ~datos[columna].between(minimo, maximo),
                columna
            ] = np.nan

    datos.loc[
        datos["orig_upb"] <= 0,
        "orig_upb"
    ] = np.nan

    # Variables categoricas: convertir primero a texto
    for columna in CATEGORICAS:

        if columna == "valuation_method":

            # Normalizar codigos numericos entre aÃ±os
            datos[columna] = (
                pd.to_numeric(
                    datos[columna],
                    errors="coerce"
                )
                .astype("Int64")
                .astype("string")
                .fillna("Sin dato")
                .astype("category")
            )

        else:

            datos[columna] = (
                datos[columna]
                .astype("string")
                .fillna("Sin dato")
                .astype("category")
            )

    datos["y_p1"] = datos["y_p1"].astype("int8")

    return datos


# ==========================================
# 4. PERIODOS DE ENTRENAMIENTO
# ==========================================

def generar_trimestres(inicio, fin):

    return [
        (anio, trimestre)
        for anio in range(inicio, fin + 1)
        for trimestre in range(1, 5)
    ]


reciente = generar_trimestres(2022, 2022)

sin_cohortes_criticas = (
    [(2019, 1)]
    + [(2020, q) for q in [2, 3, 4]]
    + generar_trimestres(2021, 2022)
)

historico_completo = generar_trimestres(2019, 2022)

escenarios = [
    ("Reciente_2022", reciente),
    ("Sin_cohortes_criticas", sin_cohortes_criticas),
    ("Historico_completo", historico_completo)
]


# ==========================================
# 5. VALIDACION 2023
# ==========================================

print("Cargando validacion 2023...")

validacion = cargar_trimestres(
    generar_trimestres(2023, 2023)
)

validacion = limpiar(validacion)

X_val = validacion[NUMERICAS + CATEGORICAS]
y_val = validacion["y_p1"]

print("Registros validacion:", len(y_val))
print(
    "Tasa incumplimiento:",
    round(y_val.mean() * 100, 3),
    "%"
)


# ==========================================
# 6. COMPARAR LOS TRES ESCENARIOS
# ==========================================

resultados = []

for nombre, trimestres in escenarios:

    print("\n================================")
    print("ESCENARIO:", nombre)
    print("================================")

    print("Cargando entrenamiento...")

    train = cargar_trimestres(
        trimestres,
        limite=LIMITE_POR_TRIMESTRE
    )

    train = limpiar(train)

    X_train = train[NUMERICAS + CATEGORICAS]
    y_train = train["y_p1"]

    print("Registros:", len(y_train))
    print(
        "Tasa incumplimiento:",
        round(y_train.mean() * 100, 3),
        "%"
    )

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

    print("Entrenando arbol...")

    modelo.fit(X_train, y_train)

    probabilidades = modelo.predict_proba(X_val)[:, 1]

    # Umbral exploratorio previamente definido
    predicciones = (
        probabilidades >= 0.02
    ).astype(int)

    ap = average_precision_score(
        y_val,
        probabilidades
    )

    roc = roc_auc_score(
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

    resultados.append({
        "Escenario": nombre,
        "Registros_train": len(y_train),
        "Tasa_train": y_train.mean(),
        "AP": ap,
        "ROC_AUC": roc,
        "Precision_2pct": precision,
        "Recall_2pct": recall,
        "Alertas_2pct": int(predicciones.sum())
    })

    print("AP:", round(ap, 5))
    print("ROC-AUC:", round(roc, 5))
    print("Precision al 2 %:", round(precision, 5))
    print("Recall al 2 %:", round(recall, 5))

    del modelo, train, X_train, y_train
    gc.collect()


# ==========================================
# 7. RESULTADOS Y ARCHIVO CSV
# ==========================================

tabla = pd.DataFrame(resultados)

tabla = tabla.sort_values(
    "AP",
    ascending=False
)

print("\n========== COMPARACION FINAL ==========")

print(tabla.round(5).to_string(index=False))

print("\nMejor escenario segun AP:")
print(tabla.iloc[0]["Escenario"])

SALIDA.mkdir(
    parents=True,
    exist_ok=True
)

archivo = SALIDA / "comparacion_periodos_2023.csv"

tabla.to_csv(
    archivo,
    index=False
)

print("\nResultados guardados en:")
print(archivo)

print("\nEstos experimentos usan muestras")
print("de hasta 150000 creditos por trimestre.")
print("No representan el entrenamiento definitivo.")


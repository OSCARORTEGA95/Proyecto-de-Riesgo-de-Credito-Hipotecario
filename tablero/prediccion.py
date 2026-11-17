"""Probabilidad de la pregunta 1 en la vista Evaluación.

Por defecto consulta la API de inferencia (carpeta api/, red neuronal empaquetada en
package-src), que recibe los nueve campos del formulario. La dirección se define con
API_P1_URL (por defecto http://localhost:8001/api/v1/predict) y el nombre que se muestra
con MODELO_P1_DESCRIPCION.

Para volver al modelo local de MLflow sin API, define PREDICCION_P1=mlflow y, si quieres,
MODELO_P1 con una ruta local o una dirección de MLflow (por ejemplo runs:/<run_id>/modelo).
Si el modelo nuevo pide más campos, hay que agregarlos al formulario y a estimar_p1.
"""
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dash import Input, Output, ctx, html

RAIZ = Path(__file__).resolve().parents[1]
PREDICCION_P1 = os.environ.get("PREDICCION_P1", "api")  # "api" o "mlflow"
API_P1_URL = os.environ.get("API_P1_URL", "http://localhost:8001/api/v1/predict")
MODELO_P1 = os.environ.get("MODELO_P1", str(RAIZ / "modelos" / "knn" / "p1" / "v5" / "modelo"))
DESCRIPCION_P1 = os.environ.get(
    "MODELO_P1_DESCRIPCION",
    "red neuronal servida por la API" if PREDICCION_P1 == "api" else "k vecinos más cercanos, versión 5")
FUENTE_P1 = f"API ({API_P1_URL})" if PREDICCION_P1 == "api" else f"MLflow ({MODELO_P1})"

# Incumplimiento a 12 meses en las cosechas de 2024, el periodo de prueba
TASA_BASE_P1 = 0.00495

# Umbrales provisionales: (bajo si es menor que, alto desde). Se fijarán con la
# distribución de probabilidades del modelo en la prueba de 2024.
ESCENARIOS = {
    "Conservadora": (0.0020, 0.0075),
    "Moderada": (0.0025, 0.0100),
    "Flexible": (0.0050, 0.0200),
}
COLORES = {"Bajo": "#1e8449", "Medio": "#b9770e", "Alto": "#c0392b"}
CAMPOS = ["ev-fico", "ev-dti", "ev-ltv", "ev-monto", "ev-tasa", "ev-plazo",
          "ev-proposito", "ev-propiedad", "ev-estado"]
ESTILO = {"color": "#17324d", "marginBottom": "12px"}
NOTA_INICIAL = "Completa la solicitud y selecciona Comparar perfil para estimar la probabilidad."


@lru_cache(maxsize=1)
def cargar_modelo():
    import mlflow.pyfunc
    return mlflow.pyfunc.load_model(MODELO_P1)


def consultar_api(solicitud: dict) -> float:
    """Envía la solicitud a la ruta predict de la API y devuelve la probabilidad."""
    import requests
    respuesta = requests.post(API_P1_URL, json={"inputs": [solicitud]}, timeout=10)
    if respuesta.status_code != 200:
        raise RuntimeError(f"la API respondió {respuesta.status_code}: {respuesta.text[:300]}")
    return float(respuesta.json()["predictions"][0])


def estimar_p1(fico, dti, ltv, monto, tasa, plazo, proposito, propiedad, estado):
    """Probabilidad de llegar a 90 días de mora o más en los primeros 12 meses."""
    solicitud = {
        "credit_score": float(fico), "dti": float(dti), "oltv": float(ltv),
        "orig_upb": float(monto), "orig_rate": float(tasa), "orig_term": float(plazo),
        "purpose": str(proposito), "prop_type": str(propiedad), "state": str(estado),
    }
    if PREDICCION_P1 == "api":
        return consultar_api(solicitud)
    return float(cargar_modelo().predict(pd.DataFrame([solicitud]))[0])


def nivel(probabilidad, escenario="Moderada"):
    bajo, alto = ESCENARIOS[escenario]
    if probabilidad < bajo:
        return "Bajo"
    return "Alto" if probabilidad >= alto else "Medio"


def porcentaje(valor, decimales=2):
    return f"{100 * valor:.{decimales}f} %".replace(".", ",")


def tabla_escenarios(probabilidad=None):
    filas = []
    for nombre, (bajo, alto) in ESCENARIOS.items():
        n = nivel(probabilidad, nombre) if probabilidad is not None else "—"
        filas.append(html.Tr([
            html.Td(nombre),
            html.Td(f"bajo < {porcentaje(bajo)} · alto ≥ {porcentaje(alto)}"),
            html.Td(n, style={"fontWeight": "bold", "color": COLORES.get(n, "#64748b")}),
        ], style={"borderBottom": "1px solid #edf1f5"}))
    return html.Table([
        html.Thead(html.Tr([html.Th("Escenario"), html.Th("Umbrales"), html.Th("Nivel")])),
        html.Tbody(filas),
    ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "12px", "marginBottom": "8px"})


def registrar_callbacks_prediccion(app):

    @app.callback(
        Output("ev-probabilidad", "children"),
        Output("ev-probabilidad", "style"),
        Output("ev-nota-modelo", "children"),
        Output("ev-escenarios", "children"),
        Input("ev-comparar", "n_clicks"),
        Input("ev-ejemplo", "n_clicks"),
        Input("ev-limpiar", "n_clicks"),
        *[Input(campo, "value") for campo in CAMPOS],
        prevent_initial_call=True
    )
    def estimar(comparar, ejemplo, limpiar, fico, dti, ltv, monto, tasa, plazo,
                proposito, propiedad, estado):

        vacio = ("—", ESTILO, NOTA_INICIAL, tabla_escenarios())

        # Al editar un campo, cargar el ejemplo o limpiar, se borra el resultado,
        # igual que la comparación del perfil.
        if ctx.triggered_id != "ev-comparar":
            return vacio

        valores = [fico, dti, ltv, monto, tasa, plazo, proposito, propiedad, estado]
        if any(v is None or v == "" for v in valores):
            return vacio
        if not (300 <= fico <= 850 and 0 <= dti <= 100 and 0 <= ltv <= 200
                and monto > 0 and 0 < tasa <= 30 and 1 <= plazo <= 480):
            return vacio

        try:
            p = estimar_p1(*valores)
        except Exception as error:
            return ("No disponible", ESTILO,
                    f"No se pudo usar el modelo ({FUENTE_P1}): {error}", tabla_escenarios())

        n = nivel(p)
        veces = f"{p / TASA_BASE_P1:.1f}".replace(".", ",")
        nota = (f"Nivel {n.lower()} en el escenario moderado: {veces} veces la tasa de las "
                f"cosechas de 2024 ({porcentaje(TASA_BASE_P1, 3)}). Modelo: {DESCRIPCION_P1}, "
                f"consultado en {FUENTE_P1}.")
        return porcentaje(p), {**ESTILO, "color": COLORES[n]}, nota, tabla_escenarios(p)

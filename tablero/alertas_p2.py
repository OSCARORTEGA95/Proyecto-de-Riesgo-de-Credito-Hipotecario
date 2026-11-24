"""Alertas tempranas de la pregunta 2 en la vista Seguimiento.

Lee la lista que produce el puntaje por lotes (src/riesgo_credito/modelos/puntaje_p2.py), que
pidió a la API la probabilidad de que cada crédito activo y al día llegue a 90 días de mora o más
en los próximos 3 meses. La gerencia de cobranza elige cuántos créditos puede gestionar (1, 5 o
10 % de la cartera) y el tablero muestra la lista priorizada, filtrada por el estado de Seguimiento.
Al seleccionar un crédito, el tablero vuelve a consultar la API en vivo con sus variables.

La dirección de la API se define con API_P2_URL (por defecto http://localhost:8001/api/v1/predict_p2).
"""
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
from dash import Input, Output, State, dash_table, dcc, html
from dash.dash_table.Format import Format, Group

RAIZ = Path(__file__).resolve().parents[1]
RUTA = RAIZ / "data" / "tablero" / "alertas_p2"
API_P2_URL = os.environ.get("API_P2_URL", "http://localhost:8001/api/v1/predict_p2")

# Tasa de la pregunta 2 en los meses de 2024 (prueba del modelo): 447 de 1.141.651
TASA_BASE_P2 = 0.000392
CAPACIDADES = [(0.01, "1 % de la cartera"), (0.05, "5 % de la cartera"), (0.10, "10 % de la cartera")]
VARIABLES = ["loan_id", "period", "edad", "pct_saldo", "dif_tasa", "eltv", "atrasos_12m", "max_atraso_12m",
             "meses_desde_atraso", "indulgencia_12m", "con_atraso_previo", "diferimiento", "diferimiento_mes",
             "saldo_sin_interes", "mora_desastre", "modificado", "plan_asistencia", "credit_score", "dti", "oltv",
             "ocltv", "orig_rate", "orig_upb", "orig_term", "mi_pct", "num_borrowers", "num_units", "purpose",
             "occupancy", "channel", "first_time_homebuyer", "prop_type", "valuation_method", "state"]


def cargar():
    """Lista puntuada y su resumen; (None, {}) si todavía no se ha corrido el puntaje por lotes."""
    archivo = RUTA / "alertas.parquet"
    if not archivo.exists():
        return None, {}
    datos = pd.read_parquet(archivo).sort_values("prioridad").reset_index(drop=True)
    resumen_json = RUTA / "resumen.json"
    info = json.loads(resumen_json.read_text(encoding="utf-8")) if resumen_json.exists() else {}
    return datos, info


alertas, resumen = cargar()


# ---------- formato ----------

def numero(valor, decimales=0):
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def porcentaje(valor, decimales=1):
    return f"{numero(100 * valor, decimales)} %"


def mes(periodo):
    texto = str(periodo)
    return f"{texto[4:6]}/{texto[:4]}"


def kpi(titulo, identificador, detalle):
    return html.Div([
        html.P(titulo, style={"fontSize": "14px", "fontWeight": "600", "color": "#b8c5d6", "margin": 0}),
        html.H2(id=identificador, style={"fontSize": "28px", "fontWeight": "700", "color": "#ffffff",
                                         "margin": "12px 0"}),
        html.P(detalle, id=f"{identificador}-detalle", style={"fontSize": "12px", "color": "#94a3b8", "margin": 0}),
    ], className="mr-kpi", style={"backgroundColor": "#242930", "padding": "20px", "border": "1px solid #3a424d",
                                  "borderRadius": "10px"})


# ---------- cálculo ----------

def seleccionar(datos, capacidad, estado):
    """Alertas: los créditos con mayor probabilidad hasta la capacidad elegida, en toda la cartera
    puntuada; luego se filtran por estado. Devuelve (puntuados del estado, alertas del estado)."""
    n_alertas = math.ceil(capacidad * len(datos))
    marcadas = datos.assign(alerta=datos["prioridad"] <= n_alertas)
    if estado and estado != "Todos":
        marcadas = marcadas[marcadas["state"] == estado]
    return marcadas, marcadas[marcadas["alerta"]]


def indicadores(puntuados, en_alerta):
    total_esperado = puntuados["probabilidad"].sum()
    salida = {
        "puntuados": numero(len(puntuados)),
        "alertas": numero(len(en_alerta)),
        "prob": porcentaje(en_alerta["probabilidad"].mean(), 3) if len(en_alerta) else "—",
        "prob_detalle": (f"{numero(en_alerta['probabilidad'].mean() / TASA_BASE_P2, 1)} veces la tasa de 2024 "
                         f"({porcentaje(TASA_BASE_P2, 3)})") if len(en_alerta) else "Sin alertas",
        "concentra": porcentaje(en_alerta["probabilidad"].sum() / total_esperado) if total_esperado > 0 else "—",
    }
    if "y_p2" in puntuados and puntuados["y_p2"].notna().mean() > 0.95 and puntuados["y_p2"].sum() > 0:
        capturados = en_alerta["y_p2"].sum() / puntuados["y_p2"].sum()
        salida["concentra_detalle"] = (f"Atrasos esperados en la lista. Observados: {porcentaje(capturados)} de "
                                       f"{numero(puntuados['y_p2'].sum())} que sí llegaron a 90 días")
    else:
        salida["concentra_detalle"] = "Suma de probabilidades de la lista sobre la de toda la cartera"
    return salida


def filas_tabla(en_alerta, nombres_estados):
    tabla = en_alerta.head(500)
    return [{
        "prioridad": int(f["prioridad"]),
        "loan_id": f["loan_id"],
        "estado": nombres_estados.get(f["state"], f["state"]),
        "prob": round(100 * float(f["probabilidad"]), 4),
        "veces": round(float(f["probabilidad"]) / TASA_BASE_P2, 1),
        "saldo": a_json(f["saldo_estimado"]),
        "atrasos": a_json(f["atrasos_12m"]), "max_atraso": a_json(f["max_atraso_12m"]),
        "meses_atraso": a_json(f["meses_desde_atraso"]), "puntaje": a_json(f["credit_score"]),
        "eltv": a_json(f["eltv"]),
    } for _, f in tabla.iterrows()]


def a_json(valor):
    if valor is None or valor is pd.NA:
        return None
    if hasattr(valor, "item"):
        valor = valor.item()
    if isinstance(valor, float) and math.isnan(valor):
        return None
    return valor


def consultar_api(loan_id):
    """Vuelve a pedir a la API la probabilidad de un crédito de la lista."""
    import requests
    fila = alertas.loc[alertas["loan_id"] == loan_id, VARIABLES].iloc[0]
    credito = {c: a_json(v) for c, v in fila.items()}
    t0 = time.time()
    respuesta = requests.post(API_P2_URL, json={"inputs": [credito]}, timeout=60)
    milisegundos = 1000 * (time.time() - t0)
    if respuesta.status_code != 200:
        raise RuntimeError(f"la API respondió {respuesta.status_code}: {respuesta.text[:300]}")
    return float(respuesta.json()["predictions"][0]), milisegundos


# ---------- pantalla ----------

ESTILO_PANEL = {"backgroundColor": "white", "padding": "20px", "border": "1px solid #dce3eb",
                "borderRadius": "10px", "marginTop": "14px"}


def crear_alertas():
    titulo = html.H4("Alertas tempranas de riesgo · modelo de la pregunta 2")
    objetivo = html.P("Créditos activos y al día con mayor probabilidad de llegar a 90 o más días de mora "
                      "en los próximos tres meses, ordenados para priorizar la gestión de cobranza.")
    if alertas is None:
        return html.Div([titulo, objetivo, html.P(
            "Todavía no se ha generado el puntaje de la cartera. En el servidor, con la API encendida, corre "
            "python src/riesgo_credito/modelos/puntaje_p2.py y reconstruye el tablero (manual de instalación).",
            style={"fontSize": "13px", "color": "#b45309"})], className="mr-panel", style=ESTILO_PANEL)

    contexto = (f"Corte de las alertas: {mes(resumen.get('periodo', alertas['period'].max()))} · "
                f"{numero(len(alertas))} créditos al día puntuados ({resumen.get('porcentaje_creditos', 1)} % "
                f"de la cartera, muestra fija) · Modelo: k vecinos (knn-p2, versión 1), consultado en la API · "
                f"Filtro de estado: el de Seguimiento.")
    tabla = dash_table.DataTable(
        id="alt-tabla",
        columns=[
            {"name": "Prioridad", "id": "prioridad", "type": "numeric"},
            {"name": "Crédito", "id": "loan_id"},
            {"name": "Estado", "id": "estado"},
            {"name": "Probabilidad (%)", "id": "prob", "type": "numeric", "format": Format(precision=3)},
            {"name": "Veces la tasa base", "id": "veces", "type": "numeric", "format": Format(precision=1)},
            {"name": "Saldo estimado (USD)", "id": "saldo", "type": "numeric",
             "format": Format(group=Group.yes, precision=0)},
            {"name": "Atrasos 12 m", "id": "atrasos", "type": "numeric"},
            {"name": "Máx. atraso 12 m", "id": "max_atraso", "type": "numeric"},
            {"name": "Meses desde atraso", "id": "meses_atraso", "type": "numeric"},
            {"name": "Puntaje", "id": "puntaje", "type": "numeric"},
            {"name": "LTV actual", "id": "eltv", "type": "numeric"},
        ],
        data=[], page_size=10, sort_action="native", row_selectable="single", selected_rows=[],
        locale_format={"decimal": ",", "group": ".", "grouping": [3]},
        style_table={"overflowX": "auto"},
        style_cell={"padding": "11px", "fontFamily": "Arial", "fontSize": "12px", "textAlign": "left",
                    "color": "#e5edf7", "backgroundColor": "#242930"},
        style_header={"backgroundColor": "#303945", "color": "#ffffff", "fontWeight": "bold"},
    )
    return html.Div([
        titulo, objetivo,
        html.P(contexto, style={"fontSize": "13px", "color": "#64748b"}),
        html.Div([
            html.Label("Capacidad de gestión de cobranza", style={"fontSize": "13px", "color": "#64748b",
                                                                   "marginRight": "12px"}),
            dcc.RadioItems(id="alt-capacidad", options=[{"label": e, "value": v} for v, e in CAPACIDADES],
                           value=0.05, inline=True, inputStyle={"marginRight": "6px", "marginLeft": "14px"}),
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "margin": "8px 0 14px"}),
        html.Div([
            kpi("Créditos puntuados", "alt-puntuados", "Al día en el mes de corte"),
            kpi("Créditos en alerta", "alt-alertas", "Dentro de la capacidad elegida"),
            kpi("Probabilidad media en alerta", "alt-prob", ""),
            kpi("Riesgo que concentra la lista", "alt-concentra", ""),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "12px"}),
        html.H4("Lista priorizada", style={"marginTop": "18px"}),
        html.P("Selecciona un crédito para volver a consultar su probabilidad en la API.",
               style={"fontSize": "12px", "color": "#64748b"}),
        tabla,
        html.Div(id="alt-api", style={"fontSize": "13px", "marginTop": "12px", "color": "#17324d"}),
    ], className="mr-panel", style=ESTILO_PANEL)


def registrar_callbacks_alertas(app, nombres_estados):
    if alertas is None:
        return

    @app.callback(
        Output("alt-puntuados", "children"),
        Output("alt-alertas", "children"),
        Output("alt-prob", "children"),
        Output("alt-prob-detalle", "children"),
        Output("alt-concentra", "children"),
        Output("alt-concentra-detalle", "children"),
        Output("alt-tabla", "data"),
        Output("alt-tabla", "selected_rows"),
        Input("seg-estado", "value"),
        Input("alt-capacidad", "value"),
    )
    def actualizar_alertas(estado, capacidad):
        puntuados, en_alerta = seleccionar(alertas, capacidad, estado)
        k = indicadores(puntuados, en_alerta)
        return (k["puntuados"], k["alertas"], k["prob"], k["prob_detalle"], k["concentra"],
                k["concentra_detalle"], filas_tabla(en_alerta, nombres_estados), [])

    @app.callback(
        Output("alt-api", "children"),
        Input("alt-tabla", "selected_rows"),
        State("alt-tabla", "data"),
    )
    def consultar(seleccion, datos):
        if not seleccion or not datos or seleccion[0] >= len(datos):
            return ""
        fila = datos[seleccion[0]]
        try:
            p, ms = consultar_api(fila["loan_id"])
        except Exception as error:
            return html.Span(f"No se pudo consultar la API ({API_P2_URL}): {error}", style={"color": "#c0392b"})
        return html.Span([
            html.B(f"Crédito {fila['loan_id']}: "),
            f"la API ({API_P2_URL}) devolvió {porcentaje(p, 3)} de probabilidad de llegar a 90 días de mora "
            f"en 3 meses, {numero(p / TASA_BASE_P2, 1)} veces la tasa base, en {numero(ms)} ms.",
        ])

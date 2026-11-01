
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, dash_table, Input, Output


# ==========================================
# 1. DATOS DE SEGUIMIENTO
# ==========================================

RUTA = (
    Path(__file__).resolve().parents[1]
    / "data" / "tablero" / "serie_mensual"
)

datos_seg = pd.read_parquet(RUTA)
datos_seg["period"] = datos_seg["period"].astype(int)

meses = sorted(datos_seg["period"].unique())
estados_seg = sorted(datos_seg["state"].dropna().unique())


# ==========================================
# 2. FUNCIONES AUXILIARES
# ==========================================

def numero(valor, decimales=0):
    texto = f"{valor:,.{decimales}f}"
    return (
        texto.replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def monto_usd(valor):
    if valor >= 1e12:
        return f"USD {numero(valor / 1e12, 2)} billones"
    if valor >= 1e9:
        return f"USD {numero(valor / 1e9, 2)} mil M"
    if valor >= 1e6:
        return f"USD {numero(valor / 1e6, 2)} M"
    return f"USD {numero(valor)}"


def etiqueta_mes(periodo):
    texto = str(periodo)
    return f"{texto[4:6]}/{texto[:4]}"


def tarjeta_kpi(titulo, identificador):
    return html.Div([
        html.P(titulo, style={
            "fontSize": "13px",
            "color": "#64748b",
            "margin": "0"
        }),
        html.H2(id=identificador, style={
            "fontSize": "24px",
            "color": "#17324d",
            "margin": "12px 0"
        })
    ], style={
        "backgroundColor": "white",
        "padding": "18px",
        "border": "1px solid #dce3eb",
        "borderRadius": "10px"
    })


def tarjeta_grafica(titulo, identificador):
    return html.Div([
        html.H4(titulo, style={"color": "#17324d"}),
        dcc.Graph(
            id=identificador,
            config={"displayModeBar": False}
        )
    ], style={
        "backgroundColor": "white",
        "padding": "16px",
        "border": "1px solid #dce3eb",
        "borderRadius": "10px",
        "minWidth": 0
    })


# ==========================================
# 3. PANTALLA DE SEGUIMIENTO
# ==========================================

def crear_seguimiento(nombres_estados):

    opciones_estado = [
        {"label": "Todos", "value": "Todos"}
    ] + [
        {
            "label": nombres_estados.get(x, x),
            "value": x
        }
        for x in estados_seg
    ]

    opciones_mes = [
        {
            "label": etiqueta_mes(x),
            "value": int(x)
        }
        for x in meses
    ]

    return html.Div([

        html.H2("Seguimiento de cartera"),

        html.P(
            "Monitoreo histórico de saldos, créditos activos "
            "y niveles de mora.",
            style={"color": "#64748b"}
        ),

        # Filtros

        html.Div([
            html.Div([
                html.Label("Estado geográfico"),
                dcc.Dropdown(
                    id="seg-estado",
                    options=opciones_estado,
                    value="Todos",
                    clearable=False
                )
            ]),

            html.Div([
                html.Label("Mes de corte"),
                dcc.Dropdown(
                    id="seg-mes",
                    options=opciones_mes,
                    value=int(meses[-1]),
                    clearable=False
                )
            ])

        ], style={
            "display": "grid",
            "gridTemplateColumns": "repeat(2, minmax(180px, 1fr))",
            "gap": "14px",
            "maxWidth": "700px",
            "marginBottom": "20px"
        }),

        # Seis indicadores reales

        html.Div([
            tarjeta_kpi("Créditos activos", "seg-activos"),
            tarjeta_kpi("Saldo de cartera", "seg-saldo"),
            tarjeta_kpi("Créditos al día", "seg-al-dia"),
            tarjeta_kpi("Mora 30+ días", "seg-mora30"),
            tarjeta_kpi("Mora 60+ días", "seg-mora60"),
            tarjeta_kpi("Mora 90+ días", "seg-mora90")
        ], style={
            "display": "grid",
            "gridTemplateColumns": (
                "repeat(auto-fit, minmax(180px, 1fr))"
            ),
            "gap": "12px"
        }),

        html.P(
            "Los conteos de mora son acumulativos: "
            "90+ está incluido en 60+ y 30+.",
            style={
                "fontSize": "12px",
                "color": "#64748b"
            }
        ),

        # Gráficas históricas

        html.Div([
            tarjeta_grafica(
                "Evolución de créditos activos",
                "seg-grafico-activos"
            ),
            tarjeta_grafica(
                "Evolución de la tasa de mora 90+",
                "seg-grafico-mora"
            )
        ], style={
            "display": "grid",
            "gridTemplateColumns": (
                "repeat(auto-fit, minmax(400px, 1fr))"
            ),
            "gap": "14px",
            "marginTop": "15px"
        }),

        # Distribución geográfica

        html.Div([
            tarjeta_grafica(
                "Créditos en mora 90+ por estado",
                "seg-grafico-estados"
            )
        ], style={"marginTop": "14px"}),

        # Tabla descriptiva

        html.Div([
            html.H4("Resumen por estado geográfico"),

            dash_table.DataTable(
                id="seg-tabla",
                columns=[
                    {"name": "Estado", "id": "estado"},
                    {"name": "Créditos activos", "id": "activos"},
                    {"name": "Mora 90+", "id": "mora90"},
                    {"name": "Tasa mora 90+", "id": "tasa"}
                ],
                data=[],
                page_size=10,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "padding": "11px",
                    "fontFamily": "Arial",
                    "fontSize": "12px",
                    "textAlign": "left"
                },
                style_header={
                    "backgroundColor": "#edf2f7",
                    "fontWeight": "bold"
                }
            )

        ], style={
            "backgroundColor": "white",
            "padding": "18px",
            "border": "1px solid #dce3eb",
            "borderRadius": "10px",
            "marginTop": "14px"
        }),

        # Integración predictiva futura

        html.Div([
            html.H4("Alertas tempranas de riesgo"),

            html.P(
                "Predicción de alcanzar 90 o más días "
                "de mora en los próximos tres meses."
            ),

            html.P(
                "Pendiente de integrar el modelo predictivo "
                "y la API del equipo. Los indicadores anteriores "
                "corresponden a información histórica observada.",
                style={"color": "#64748b", "fontSize": "13px"}
            )

        ], style={
            "backgroundColor": "white",
            "padding": "20px",
            "border": "1px solid #dce3eb",
            "borderRadius": "10px",
            "marginTop": "14px"
        })

    ], style={
        "padding": "24px",
        "backgroundColor": "#f2f5fa",
        "minHeight": "100vh"
    })


# ==========================================
# 4. ACTUALIZAR SEGUIMIENTO
# ==========================================

def registrar_callbacks_seguimiento(app, nombres_estados):

    @app.callback(
        Output("seg-activos", "children"),
        Output("seg-saldo", "children"),
        Output("seg-al-dia", "children"),
        Output("seg-mora30", "children"),
        Output("seg-mora60", "children"),
        Output("seg-mora90", "children"),
        Output("seg-grafico-activos", "figure"),
        Output("seg-grafico-mora", "figure"),
        Output("seg-grafico-estados", "figure"),
        Output("seg-tabla", "data"),

        Input("seg-estado", "value"),
        Input("seg-mes", "value")
    )
    def actualizar_seguimiento(estado, mes):

        # Aplicar filtro geográfico
        tabla = datos_seg.copy()

        if estado != "Todos":
            tabla = tabla[tabla["state"] == estado]

        # Último corte seleccionado
        corte = tabla[tabla["period"] == mes]

        activos = int(corte["activos"].sum())
        saldo = corte["saldo"].sum()
        al_dia = int(corte["al_dia"].sum())
        mora30 = int(corte["mora_30"].sum())
        mora60 = int(corte["mora_60"].sum())
        mora90 = int(corte["mora_90"].sum())

        # Serie histórica hasta el mes seleccionado
        historial = (
            tabla[tabla["period"] <= mes]
            .groupby("period", as_index=False)[
                ["activos", "mora_90"]
            ]
            .sum()
            .sort_values("period")
        )

        historial["tasa_mora90"] = (
            100 * historial["mora_90"]
            / historial["activos"].replace(0, float("nan"))
        )

        fechas = [
            etiqueta_mes(x)
            for x in historial["period"]
        ]

        # Gráfico 1: créditos activos
        figura_activos = go.Figure(go.Scatter(
            x=fechas,
            y=historial["activos"],
            mode="lines",
            line=dict(color="#3478b7", width=2)
        ))

        figura_activos.update_layout(
            height=350,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=65, r=20, t=15, b=45),
            xaxis=dict(nticks=9),
            yaxis=dict(gridcolor="#edf1f5"),
            showlegend=False
        )

        # Gráfico 2: tasa observada de mora 90+
        figura_mora = go.Figure(go.Scatter(
            x=fechas,
            y=historial["tasa_mora90"],
            mode="lines",
            line=dict(color="#c47b39", width=2),
            fill="tozeroy",
            fillcolor="rgba(196,123,57,0.10)"
        ))

        figura_mora.update_layout(
            height=350,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=65, r=20, t=15, b=45),
            xaxis=dict(nticks=9),
            yaxis=dict(
                ticksuffix=" %",
                gridcolor="#edf1f5",
                rangemode="tozero"
            ),
            showlegend=False
        )

        # Resumen por estado en el mes de corte
        por_estado = (
            corte.groupby("state", as_index=False)[
                ["activos", "mora_90"]
            ]
            .sum()
        )

        por_estado["tasa"] = (
            100 * por_estado["mora_90"]
            / por_estado["activos"].replace(0, float("nan"))
        )

        por_estado = por_estado.sort_values(
            "mora_90", ascending=False
        )

        top10 = por_estado.head(10).iloc[::-1]

        figura_estados = go.Figure(go.Bar(
            x=top10["mora_90"],
            y=[
                nombres_estados.get(x, x)
                for x in top10["state"]
            ],
            orientation="h",
            marker_color="#3478b7"
        ))

        figura_estados.update_layout(
            height=400,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=150, r=25, t=15, b=35),
            xaxis=dict(gridcolor="#edf1f5"),
            showlegend=False
        )

        # Tabla descriptiva (no son predicciones)
        filas = []

        for _, fila in por_estado.iterrows():
            filas.append({
                "estado": nombres_estados.get(
                    fila["state"], fila["state"]
                ),
                "activos": numero(fila["activos"]),
                "mora90": numero(fila["mora_90"]),
                "tasa": (
                    numero(fila["tasa"], 2) + " %"
                    if pd.notna(fila["tasa"]) else "—"
                )
            })

        return (
            numero(activos),
            monto_usd(saldo),
            numero(al_dia),
            numero(mora30),
            numero(mora60),
            numero(mora90),
            figura_activos,
            figura_mora,
            figura_estados,
            filas
        )

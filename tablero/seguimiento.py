
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from dash import html, dcc, dash_table, Input, Output
from dash.dash_table.Format import Format, Group


# ==========================================
# 1. DATOS DE SEGUIMIENTO
# ==========================================

RUTA = (
    Path(__file__).resolve().parents[1]
    / "data" / "tablero" / "serie_mensual"
)

datos_seg = pd.read_parquet(RUTA)

datos_seg["period"] = datos_seg["period"].astype(int)
datos_seg["trimestre"] = datos_seg["trimestre"].astype(str)

meses = sorted(datos_seg["period"].unique())
trimestres_seg = sorted(datos_seg["trimestre"].unique())
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


def etiqueta_trimestre(trimestre):
    return trimestre.replace("Q", " Q")


def selector(titulo, identificador, opciones, valor):
    return html.Div([

        html.Label(titulo, style={
            "fontSize": "13px",
            "color": "#64748b",
            "display": "block",
            "marginBottom": "6px"
        }),

        dcc.Dropdown(
            id=identificador,
            options=opciones,
            value=valor,
            clearable=False
        )

    ], style={"minWidth": 0})


def tarjeta_kpi(titulo, identificador, detalle):
    return html.Div([

        html.P(titulo, style={
            "fontSize": "13px",
            "color": "#64748b",
            "margin": 0
        }),

        html.H2(id=identificador, style={
            "fontSize": "24px",
            "color": "#17324d",
            "margin": "12px 0"
        }),

        html.P(detalle, style={
            "fontSize": "11px",
            "color": "#94a3b8",
            "margin": 0
        })

    ], style={
        "backgroundColor": "white",
        "padding": "18px",
        "border": "1px solid #dce3eb",
        "borderRadius": "10px"
    })


def tarjeta_grafica(titulo, subtitulo, identificador):
    return html.Div([

        html.H4(titulo, style={
            "margin": "0 0 6px",
            "color": "#17324d"
        }),

        html.P(subtitulo, style={
            "fontSize": "12px",
            "color": "#64748b",
            "margin": "0 0 8px"
        }),

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


def figura_vacia(mensaje):
    figura = go.Figure()

    figura.add_annotation(
        text=mensaje,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False
    )

    figura.update_layout(
        height=350,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=30, r=30, t=30, b=30),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    return figura


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

    opciones_trimestre = [
        {
            "label": etiqueta_trimestre(x),
            "value": x
        }
        for x in trimestres_seg
    ]

    filtros = html.Div([

        selector(
            "Estado geográfico",
            "seg-estado",
            opciones_estado,
            "Todos"
        ),

        selector(
            "Originación desde",
            "seg-desde",
            opciones_trimestre,
            trimestres_seg[0]
        ),

        selector(
            "Originación hasta",
            "seg-hasta",
            opciones_trimestre,
            trimestres_seg[-1]
        ),

        selector(
            "Mes de corte",
            "seg-mes",
            opciones_mes,
            int(meses[-1])
        ),

        html.Button(
            "Restablecer",
            id="seg-restablecer",
            n_clicks=0,
            style={
                "padding": "12px",
                "backgroundColor": "white",
                "border": "1px solid #ccd6e0",
                "borderRadius": "6px",
                "cursor": "pointer",
                "alignSelf": "end"
            }
        )

    ], style={
        "display": "grid",
        "gridTemplateColumns": (
            "repeat(auto-fit, minmax(170px, 1fr))"
        ),
        "gap": "12px",
        "alignItems": "end",
        "backgroundColor": "white",
        "padding": "18px",
        "border": "1px solid #dce3eb",
        "borderRadius": "10px",
        "marginBottom": "16px"
    })

    indicadores = html.Div([

        tarjeta_kpi(
            "Créditos activos",
            "seg-activos",
            "Cartera en el mes de corte"
        ),

        tarjeta_kpi(
            "Saldo de cartera",
            "seg-saldo",
            "Saldo vigente en USD"
        ),

        tarjeta_kpi(
            "Créditos al día",
            "seg-al-dia",
            "Sin mora de 30+ días"
        ),

        tarjeta_kpi(
            "Mora 30+ días",
            "seg-mora30",
            "Créditos con atraso de 30 días o más"
        ),

        tarjeta_kpi(
            "Mora 60+ días",
            "seg-mora60",
            "Créditos con atraso de 60 días o más"
        ),

        tarjeta_kpi(
            "Mora 90+ días",
            "seg-mora90",
            "Créditos con atraso de 90 días o más"
        )

    ], style={
        "display": "grid",
        "gridTemplateColumns": (
            "repeat(auto-fit, minmax(180px, 1fr))"
        ),
        "gap": "12px"
    })

    graficas = html.Div([

        tarjeta_grafica(
            "Evolución de créditos activos",
            "Número de créditos por mes",
            "seg-grafico-activos"
        ),

        tarjeta_grafica(
            "Evolución de la tasa de mora 90+",
            "Créditos en mora 90+ / créditos activos",
            "seg-grafico-mora"
        )

    ], style={
        "display": "grid",
        "gridTemplateColumns": (
            "repeat(auto-fit, minmax(400px, 1fr))"
        ),
        "gap": "14px",
        "marginTop": "14px"
    })

    resumen_geografico = html.Div([

        html.H4("Resumen por estado geográfico"),

        dash_table.DataTable(
            id="seg-tabla",

            columns=[
                {
                    "name": "Estado",
                    "id": "estado"
                },
                {
                    "name": "Créditos activos",
                    "id": "activos",
                    "type": "numeric",
                    "format": Format(
                        group=Group.yes,
                        precision=0
                    )
                },
                {
                    "name": "Mora 90+",
                    "id": "mora90",
                    "type": "numeric",
                    "format": Format(
                        group=Group.yes,
                        precision=0
                    )
                },
                {
                    "name": "Tasa mora 90+ (%)",
                    "id": "tasa",
                    "type": "numeric",
                    "format": Format(
                        precision=2
                    )
                }
            ],

            data=[],
            page_size=10,
            sort_action="native",

            locale_format={
                "decimal": ",",
                "group": ".",
                "grouping": [3]
            },

            style_table={
                "overflowX": "auto"
            },

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
    })

    alertas = html.Div([

        html.H4("Alertas tempranas de riesgo"),

        html.P(
            "Objetivo: identificar créditos activos y al día "
            "con probabilidad de alcanzar 90 o más días "
            "de mora durante los próximos tres meses."
        ),

        html.P(
            "Pendiente de integrar el modelo predictivo "
            "y la API del equipo. Los indicadores presentados "
            "en esta pantalla son históricos y observados.",
            style={
                "fontSize": "13px",
                "color": "#64748b"
            }
        )

    ], style={
        "backgroundColor": "white",
        "padding": "20px",
        "border": "1px solid #dce3eb",
        "borderRadius": "10px",
        "marginTop": "14px"
    })

    return html.Div([

        html.H2("Seguimiento de cartera"),

        html.P(
            "Monitoreo histórico de saldos, créditos "
            "activos y niveles de mora.",
            style={"color": "#64748b"}
        ),

        filtros,

        html.P(
            id="seg-contexto",
            style={
                "fontSize": "13px",
                "color": "#64748b"
            }
        ),

        html.P(
            id="seg-mensaje",
            style={
                "fontSize": "13px",
                "color": "#b45309"
            }
        ),

        indicadores,

        html.P(
            "Los conteos de mora son acumulativos: "
            "90+ está incluido en 60+ y 30+. "
            "No deben sumarse entre sí.",
            style={
                "fontSize": "12px",
                "color": "#64748b"
            }
        ),

        graficas,

        html.Div(
            tarjeta_grafica(
                "Créditos en mora 90+ por estado",
                "10 estados con mayor número de créditos "
                "en mora 90+",
                "seg-grafico-estados"
            ),
            style={"marginTop": "14px"}
        ),

        resumen_geografico,

        alertas

    ], style={
        "padding": "24px",
        "backgroundColor": "#f2f5fa",
        "minHeight": "100vh"
    })


# ==========================================
# 4. ACTUALIZAR INDICADORES Y GRAFICAS
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
        Output("seg-contexto", "children"),
        Output("seg-mensaje", "children"),

        Input("seg-estado", "value"),
        Input("seg-desde", "value"),
        Input("seg-hasta", "value"),
        Input("seg-mes", "value")
    )
    def actualizar_seguimiento(estado, desde, hasta, mes):

        contexto = (
            f"Corte: {etiqueta_mes(mes)} · "
            f"Originación: {etiqueta_trimestre(desde)} "
            f"a {etiqueta_trimestre(hasta)} · "
            f"Estado: {nombres_estados.get(estado, estado)}"
        )

        # Validar el rango de originación
        if desde > hasta:

            vacia = figura_vacia(
                "Selecciona un rango de originación válido"
            )

            return (
                "0", "USD 0", "0", "0", "0", "0",
                vacia, vacia, vacia,
                [],
                contexto,
                "La originación inicial no puede "
                "ser posterior a la final."
            )

        # Aplicar los filtros de originación
        tabla = datos_seg[
            (datos_seg["trimestre"] >= desde) &
            (datos_seg["trimestre"] <= hasta)
        ]

        # Aplicar el filtro de estado
        if estado != "Todos":
            tabla = tabla[tabla["state"] == estado]

        # Tomar únicamente el mes de corte
        corte = tabla[tabla["period"] == mes]

        activos = int(corte["activos"].sum())

        if activos == 0:

            vacia = figura_vacia(
                "Sin información para los filtros seleccionados"
            )

            return (
                "0", "USD 0", "0", "0", "0", "0",
                vacia, vacia, vacia,
                [],
                contexto,
                "No se encontraron créditos activos "
                "para este corte y rango de originación."
            )

        # Indicadores del mes seleccionado
        saldo = corte["saldo"].sum()
        al_dia = int(corte["al_dia"].sum())
        mora30 = int(corte["mora_30"].sum())
        mora60 = int(corte["mora_60"].sum())
        mora90 = int(corte["mora_90"].sum())

        # ----------------------------------
        # Evolución mensual
        # ----------------------------------

        historial = (
            tabla[tabla["period"] <= mes]
            .groupby("period", as_index=False)[
                ["activos", "mora_90"]
            ]
            .sum()
            .sort_values("period")
        )

        historial = historial[
            historial["activos"] > 0
        ].copy()

        historial["tasa_mora90"] = (
            100 * historial["mora_90"]
            / historial["activos"]
        )

        fechas = [
            etiqueta_mes(x)
            for x in historial["period"]
        ]

        # Gráfica 1: créditos activos
        figura_activos = go.Figure(go.Scatter(
            x=fechas,
            y=historial["activos"],
            mode="lines",
            line=dict(
                color="#3478b7",
                width=2
            ),
            hovertemplate=(
                "%{x}<br>"
                "Créditos activos: %{y:,.0f}"
                "<extra></extra>"
            )
        ))

        figura_activos.update_layout(
            height=350,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=75, r=20, t=15, b=50),
            xaxis=dict(nticks=9),
            yaxis=dict(
                gridcolor="#edf1f5",
                rangemode="tozero"
            ),
            showlegend=False
        )

        # Gráfica 2: tasa de mora 90+
        figura_mora = go.Figure(go.Scatter(
            x=fechas,
            y=historial["tasa_mora90"],
            mode="lines",
            line=dict(
                color="#c47b39",
                width=2
            ),
            fill="tozeroy",
            fillcolor="rgba(196,123,57,0.10)",
            hovertemplate=(
                "%{x}<br>"
                "Tasa mora 90+: %{y:.2f} %"
                "<extra></extra>"
            )
        ))

        figura_mora.update_layout(
            height=350,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=65, r=20, t=15, b=50),
            xaxis=dict(nticks=9),
            yaxis=dict(
                ticksuffix=" %",
                gridcolor="#edf1f5",
                rangemode="tozero"
            ),
            showlegend=False
        )

        # ----------------------------------
        # Resumen por estado
        # ----------------------------------

        por_estado = (
            corte.groupby("state", as_index=False)[
                ["activos", "mora_90"]
            ]
            .sum()
        )

        por_estado["tasa"] = (
            100 * por_estado["mora_90"]
            / por_estado["activos"].replace(
                0, float("nan")
            )
        )

        por_estado = por_estado.sort_values(
            "mora_90",
            ascending=False
        )

        # Gráfica 3: diez estados con más mora
        top10 = (
            por_estado[por_estado["mora_90"] > 0]
            .head(10)
            .iloc[::-1]
        )

        if top10.empty:

            figura_estados = figura_vacia(
                "No se registran créditos en mora 90+"
            )

        else:

            figura_estados = go.Figure(go.Bar(
                x=top10["mora_90"],
                y=[
                    nombres_estados.get(x, x)
                    for x in top10["state"]
                ],
                orientation="h",
                marker_color="#3478b7",
                text=[
                    numero(x)
                    for x in top10["mora_90"]
                ],
                textposition="outside",
                cliponaxis=False
            ))

            figura_estados.update_layout(
                height=400,
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=150, r=85, t=15, b=35),
                xaxis=dict(
                    gridcolor="#edf1f5",
                    rangemode="tozero"
                ),
                showlegend=False
            )

        # Tabla: valores numéricos para ordenar
        filas = []

        for _, fila in por_estado.iterrows():

            tasa = (
                round(float(fila["tasa"]), 2)
                if pd.notna(fila["tasa"])
                else None
            )

            filas.append({
                "estado": nombres_estados.get(
                    fila["state"],
                    fila["state"]
                ),
                "activos": int(fila["activos"]),
                "mora90": int(fila["mora_90"]),
                "tasa": tasa
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
            filas,
            contexto,
            ""
        )

    # ======================================
    # 5. BOTON RESTABLECER
    # ======================================

    @app.callback(
        Output("seg-estado", "value"),
        Output("seg-desde", "value"),
        Output("seg-hasta", "value"),
        Output("seg-mes", "value"),

        Input("seg-restablecer", "n_clicks"),
        prevent_initial_call=True
    )
    def restablecer_seguimiento(n_clicks):

        return (
            "Todos",
            trimestres_seg[0],
            trimestres_seg[-1],
            int(meses[-1])
        )


from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html


# ==========================================
# 1. DATOS DE ORIGINACION
# ==========================================

RUTA = (
    Path(__file__).resolve().parents[1]
    / "data" / "tablero" / "originacion"
)

datos = pd.read_parquet(RUTA)
datos["trimestre"] = datos["trimestre"].astype(str)

trimestres = sorted(datos["trimestre"].unique())
estados = sorted(datos["state"].dropna().unique())

rangos_fico = [
    "<620", "620-679", "680-719",
    "720-759", "760-799", ">=800", "Sin dato"
]

rangos_ltv = [
    "<=60", "61-70", "71-80",
    "81-90", "91-95", ">95", "Sin dato"
]

propositos = {
    "P": "Compra",
    "N": "Refinanciación sin retiro",
    "C": "Refinanciación con retiro"
}

nombres_estados = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "Distrito de Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawái",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Luisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Míchigan",
    "MN": "Minnesota",
    "MS": "Misisipi",
    "MO": "Misuri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "Nuevo Hampshire",
    "NJ": "Nueva Jersey",
    "NM": "Nuevo México",
    "NY": "Nueva York",
    "NC": "Carolina del Norte",
    "ND": "Dakota del Norte",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregón",
    "PA": "Pensilvania",
    "RI": "Rhode Island",
    "SC": "Carolina del Sur",
    "SD": "Dakota del Sur",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "Virginia Occidental",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "PR": "Puerto Rico",
    "GU": "Guam",
    "VI": "Islas Vírgenes"
}


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


def promedio(tabla, suma, cantidad, decimales=2, porcentaje=False):
    n = tabla[cantidad].sum()

    if n == 0:
        return "—"

    resultado = numero(
        tabla[suma].sum() / n,
        decimales
    )

    return resultado + " %" if porcentaje else resultado


def filtrar_datos(estado, desde, hasta, proposito, fico):
    if desde > hasta:
        return datos.iloc[0:0]

    tabla = datos[
        (datos["trimestre"] >= desde) &
        (datos["trimestre"] <= hasta)
    ]

    if estado != "Todos":
        tabla = tabla[tabla["state"] == estado]

    if proposito != "Todos":
        tabla = tabla[tabla["purpose"] == proposito]

    if fico != "Todos":
        tabla = tabla[tabla["rango_puntaje"] == fico]

    return tabla


# ==========================================
# 3. COMPONENTES VISUALES
# ==========================================

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
            "fontSize": "25px",
            "color": "#17324d",
            "margin": "12px 0"
        }),
        html.P(detalle, style={
            "fontSize": "12px",
            "color": "#94a3b8",
            "margin": 0
        })
    ], style={
        "backgroundColor": "white",
        "padding": "18px",
        "border": "1px solid #dce3eb",
        "borderRadius": "10px"
    })


def tarjeta_grafica(titulo, subtitulo, identificador, nota=None):
    contenido = [
        html.H4(titulo, style={
            "color": "#17324d",
            "margin": "0 0 6px"
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
    ]

    if nota:
        contenido.append(
            html.P(nota, style={
                "fontSize": "11px",
                "color": "#64748b"
            })
        )

    return html.Div(contenido, style={
        "backgroundColor": "white",
        "padding": "15px",
        "border": "1px solid #dce3eb",
        "borderRadius": "10px",
        "minWidth": 0
    })


def figura_vacia(mensaje="Sin datos para los filtros seleccionados"):
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
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white"
    )

    return figura


def grafica_barras(tabla, columna, orden):
    conteos = tabla.groupby(columna)["creditos"].sum()

    categorias = [x for x in orden if x in conteos.index]
    categorias += [
        x for x in conteos.index
        if x not in categorias
    ]

    valores = [conteos[x] for x in categorias]

    figura = go.Figure(go.Bar(
        x=categorias,
        y=valores,
        marker_color="#3478b7",
        text=[numero(x) for x in valores],
        textposition="outside",
        textfont=dict(size=10),
        cliponaxis=False
    ))

    figura.update_layout(
        height=350,
        margin=dict(l=45, r=15, t=35, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=11),
        showlegend=False,
        xaxis=dict(tickangle=-25),
        yaxis=dict(
            gridcolor="#edf1f5",
            rangemode="tozero"
        )
    )

    return figura


# ==========================================
# 4. PANTALLA PANORAMA
# ==========================================

def crear_panorama():

    opciones_estado = [
        {"label": "Todos", "value": "Todos"}
    ] + [
        {
            "label": nombres_estados.get(x, x),
            "value": x
        }
        for x in estados
    ]

    opciones_trimestre = [
        {
            "label": x.replace("Q", " Q"),
            "value": x
        }
        for x in trimestres
    ]

    opciones_proposito = [
        {"label": "Todos", "value": "Todos"}
    ] + [
        {"label": nombre, "value": codigo}
        for codigo, nombre in propositos.items()
    ]

    opciones_fico = [
        {"label": "Todos", "value": "Todos"}
    ] + [
        {"label": x, "value": x}
        for x in rangos_fico
    ]

    filtros = html.Div([

        selector(
            "Estado", "filtro-estado",
            opciones_estado, "Todos"
        ),

        selector(
            "Originación desde", "filtro-desde",
            opciones_trimestre, trimestres[0]
        ),

        selector(
            "Hasta", "filtro-hasta",
            opciones_trimestre, trimestres[-1]
        ),

        html.Button(
            "Restablecer",
            id="restablecer",
            n_clicks=0,
            style={
                "padding": "12px",
                "backgroundColor": "white",
                "border": "1px solid #ccd6e0",
                "borderRadius": "6px",
                "cursor": "pointer"
            }
        )

    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(170px, 1fr))",
        "gap": "12px",
        "alignItems": "end",
        "backgroundColor": "white",
        "padding": "16px 24px",
        "borderBottom": "1px solid #dce3eb"
    })

    encabezado = html.Div([

        html.Div([
            html.H2("Panorama de cartera"),
            html.P(id="subtitulo")
        ]),

        html.Div([

            selector(
                "Propósito", "filtro-proposito",
                opciones_proposito, "Todos"
            ),

            selector(
                "Rango FICO", "filtro-fico",
                opciones_fico, "Todos"
            )

        ], style={
            "display": "grid",
            "gridTemplateColumns": (
                "repeat(2, minmax(150px, 1fr))"
            ),
            "gap": "12px",
            "maxWidth": "450px",
            "width": "100%"
        })

    ], style={
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "gap": "20px",
        "flexWrap": "wrap"
    })

    indicadores = [
        ("Créditos originados", "kpi-creditos", "Total de créditos"),
        ("Monto total originado", "kpi-monto", "Monto original de la cartera"),
        ("FICO promedio", "kpi-fico", "Puntaje crediticio original"),
        ("LTV promedio", "kpi-ltv", "Relación préstamo / valor"),
        ("DTI promedio", "kpi-dti", "Relación deuda / ingreso")
    ]

    tarjetas = html.Div([
        tarjeta_kpi(titulo, identificador, detalle)
        for titulo, identificador, detalle in indicadores
    ], style={
        "display": "grid",
        "gridTemplateColumns": (
            "repeat(auto-fit, minmax(180px, 1fr))"
        ),
        "gap": "12px"
    })

    graficas_distribucion = html.Div([

        tarjeta_grafica(
            "Distribución por propósito",
            "Número de créditos",
            "grafico-proposito"
        ),

        tarjeta_grafica(
            "Distribución por rangos de FICO",
            "Número de créditos",
            "grafico-fico"
        ),

        tarjeta_grafica(
            "Distribución por rangos de LTV",
            "Número de créditos",
            "grafico-ltv"
        )

    ], style={
        "display": "grid",
        "gridTemplateColumns": (
            "repeat(auto-fit, minmax(310px, 1fr))"
        ),
        "gap": "12px",
        "marginTop": "14px"
    })

    graficas_historicas = html.Div([

        html.Div(
            tarjeta_grafica(
                "Créditos por estado geográfico",
                "10 estados con más originaciones",
                "grafico-estados"
            ),
            style={
                "flex": "2 1 360px",
                "minWidth": 0
            }
        ),

        html.Div(
            tarjeta_grafica(
                "Evolución histórica del incumplimiento por cohorte",
                "% de créditos con 90+ días de mora "
                "durante sus primeros 12 meses",
                "grafico-cohortes",
                "Se muestran cohortes hasta 2025 Q1. "
                "Las cohortes 2019 Q2–2020 Q1 estuvieron "
                "afectadas por la pandemia; las más recientes "
                "no tienen observación completa de 12 meses."
            ),
            style={
                "flex": "3 1 500px",
                "minWidth": 0
            }
        )

    ], style={
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "12px",
        "marginTop": "14px"
    })

    contenido = html.Div([
        encabezado,
        html.P(id="mensaje-filtros", style={
            "color": "#c0392b"
        }),
        tarjetas,
        graficas_distribucion,
        graficas_historicas
    ], style={"padding": "24px"})

    return html.Div([filtros, contenido])


# ==========================================
# 5. ACTUALIZAR PANORAMA
# ==========================================

def registrar_callbacks_panorama(app):

    @app.callback(
        Output("kpi-creditos", "children"),
        Output("kpi-monto", "children"),
        Output("kpi-fico", "children"),
        Output("kpi-ltv", "children"),
        Output("kpi-dti", "children"),
        Output("subtitulo", "children"),
        Output("mensaje-filtros", "children"),
        Output("grafico-proposito", "figure"),
        Output("grafico-fico", "figure"),
        Output("grafico-ltv", "figure"),
        Output("grafico-estados", "figure"),
        Output("grafico-cohortes", "figure"),

        Input("filtro-estado", "value"),
        Input("filtro-desde", "value"),
        Input("filtro-hasta", "value"),
        Input("filtro-proposito", "value"),
        Input("filtro-fico", "value")
    )
    def actualizar_panorama(estado, desde, hasta, proposito, fico):

        tabla = filtrar_datos(
            estado, desde, hasta, proposito, fico
        )

        subtitulo = (
            f"Originaciones · "
            f"{desde.replace('Q', ' Q')} – "
            f"{hasta.replace('Q', ' Q')}"
        )

        if tabla.empty or tabla["creditos"].sum() == 0:

            mensaje = (
                "La fecha inicial no puede superar la final."
                if desde > hasta
                else "No hay créditos para estos filtros."
            )

            vacias = [figura_vacia() for _ in range(5)]

            return (
                "0", "USD 0", "—", "—", "—",
                subtitulo, mensaje, *vacias
            )

        indicadores = (
            numero(int(tabla["creditos"].sum())),
            monto_usd(tabla["monto_total"].sum()),
            promedio(tabla, "suma_puntaje", "n_puntaje", 1),
            promedio(
                tabla, "suma_ltv", "n_ltv",
                2, porcentaje=True
            ),
            promedio(
                tabla, "suma_dti", "n_dti",
                2, porcentaje=True
            )
        )

        # Gráfica 1: propósito

        conteos = tabla.groupby("purpose")["creditos"].sum()

        codigos = [
            c for c in propositos
            if c in conteos.index and conteos[c] > 0
        ]

        colores = {
            "P": "#17324d",
            "N": "#3478b7",
            "C": "#9cb5cb"
        }

        figura_proposito = go.Figure(go.Pie(
            labels=[propositos[x] for x in codigos],
            values=[conteos[x] for x in codigos],
            hole=0.65,
            marker_colors=[colores[x] for x in codigos]
        ))

        figura_proposito.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="white",
            font=dict(size=11),
            annotations=[
                dict(
                    text=numero(int(conteos.sum())),
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14)
                )
            ]
        )

        # Gráficas 2 y 3: FICO y LTV

        figura_fico = grafica_barras(
            tabla, "rango_puntaje", rangos_fico
        )

        figura_ltv = grafica_barras(
            tabla, "rango_ltv", rangos_ltv
        )

        # Gráfica 4: créditos por estado

        ranking = (
            tabla.groupby("state")["creditos"]
            .sum()
            .nlargest(10)
            .sort_values()
        )

        figura_estados = go.Figure(go.Bar(
            x=ranking.values,
            y=[
                nombres_estados.get(x, x)
                for x in ranking.index
            ],
            orientation="h",
            marker_color="#3478b7",
            text=[numero(x) for x in ranking.values],
            textposition="outside",
            cliponaxis=False
        ))

        figura_estados.update_layout(
            height=420,
            margin=dict(l=140, r=90, t=15, b=35),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(
                gridcolor="#edf1f5",
                rangemode="tozero"
            ),
            yaxis=dict(title=""),
            showlegend=False,
            font=dict(size=11)
        )

        # Gráfica 5: incumplimiento por cohorte

        cohortes = (
            tabla[tabla["trimestre"] <= "2025Q1"]
            .groupby("trimestre", as_index=False)[
                ["p1_positivos", "p1_etiquetables"]
            ]
            .sum()
            .sort_values("trimestre")
        )

        cohortes = cohortes[
            cohortes["p1_etiquetables"] > 0
        ].copy()

        if cohortes.empty:

            figura_cohortes = figura_vacia(
                "Sin cohortes con seguimiento suficiente"
            )

        else:

            cohortes["tasa"] = (
                100 * cohortes["p1_positivos"]
                / cohortes["p1_etiquetables"]
            )

            figura_cohortes = go.Figure(go.Scatter(
                x=cohortes["trimestre"].str.replace(
                    "Q", " Q", regex=False
                ),
                y=cohortes["tasa"],
                mode="lines+markers",
                line=dict(
                    color="#3478b7",
                    width=2
                ),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(52,120,183,0.10)",
                hovertemplate=(
                    "%{x}<br>"
                    "Incumplimiento: %{y:.2f} %"
                    "<extra></extra>"
                )
            ))

            figura_cohortes.update_layout(
                height=420,
                margin=dict(l=55, r=15, t=20, b=60),
                paper_bgcolor="white",
                plot_bgcolor="white",
                xaxis=dict(
                    title="Trimestre de originación",
                    tickangle=-45,
                    nticks=10,
                    showgrid=False
                ),
                yaxis=dict(
                    title="Tasa de incumplimiento",
                    ticksuffix=" %",
                    gridcolor="#edf1f5",
                    rangemode="tozero"
                ),
                showlegend=False,
                font=dict(size=11)
            )

        return (
            *indicadores,
            subtitulo,
            "",
            figura_proposito,
            figura_fico,
            figura_ltv,
            figura_estados,
            figura_cohortes
        )

    # Botón Restablecer

    @app.callback(
        Output("filtro-estado", "value"),
        Output("filtro-desde", "value"),
        Output("filtro-hasta", "value"),
        Output("filtro-proposito", "value"),
        Output("filtro-fico", "value"),

        Input("restablecer", "n_clicks"),
        prevent_initial_call=True
    )
    def restablecer_filtros(n_clicks):
        return (
            "Todos",
            trimestres[0],
            trimestres[-1],
            "Todos",
            "Todos"
        )

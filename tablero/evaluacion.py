
from dash import html, dcc, Input, Output, State, ctx


# ==========================================
# 1. ESTILOS
# ==========================================

TARJETA = {
    "backgroundColor": "white",
    "padding": "22px",
    "border": "1px solid #dce3eb",
    "borderRadius": "10px"
}

CAMPO = {
    "width": "100%",
    "padding": "11px",
    "border": "1px solid #ccd6e0",
    "borderRadius": "6px",
    "boxSizing": "border-box",
    "fontSize": "14px"
}


def numero(valor, decimales=1):
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def campo_numerico(titulo, identificador, minimo, maximo, paso=1):
    return html.Div([
        html.Label(titulo, style={
            "fontSize": "13px",
            "color": "#64748b",
            "display": "block",
            "marginBottom": "6px"
        }),
        dcc.Input(
            id=identificador,
            type="number",
            min=minimo,
            max=maximo,
            step=paso,
            style=CAMPO
        )
    ])


def campo_lista(titulo, identificador, opciones):
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
            placeholder="Seleccionar",
            clearable=True
        )
    ])


# ==========================================
# 2. DISEÑO DE EVALUACIÓN
# ==========================================

def crear_evaluacion(estados, nombres_estados):

    opciones_proposito = [
        {"label": "Compra", "value": "P"},
        {"label": "Refinanciación sin retiro", "value": "N"},
        {"label": "Refinanciación con retiro", "value": "C"}
    ]

    opciones_propiedad = [
        {"label": "Vivienda unifamiliar", "value": "SF"},
        {"label": "Unidad en proyecto planificado", "value": "PU"},
        {"label": "Condominio", "value": "CO"},
        {"label": "Vivienda prefabricada", "value": "MH"},
        {"label": "Cooperativa", "value": "CP"}
    ]

    opciones_estado = [
        {
            "label": nombres_estados.get(x, x),
            "value": x
        }
        for x in estados
    ]

    formulario = html.Div([

        html.Div([
            html.Div([
                html.H4("Datos de la solicitud", style={"margin": "0"}),
                html.P(
                    "Características del crédito al momento de la originación",
                    style={"color": "#64748b", "fontSize": "12px"}
                )
            ]),

            html.Button(
                "Cargar caso de ejemplo",
                id="ev-ejemplo",
                n_clicks=0,
                style={
                    "padding": "10px",
                    "backgroundColor": "white",
                    "border": "1px solid #ccd6e0",
                    "borderRadius": "6px",
                    "cursor": "pointer"
                }
            )
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "gap": "12px",
            "flexWrap": "wrap"
        }),

        html.Div([

            campo_numerico("Puntaje FICO", "ev-fico", 300, 850),
            campo_numerico("DTI (%)", "ev-dti", 0, 100, 0.1),

            campo_numerico("LTV (%)", "ev-ltv", 0, 200, 0.1),
            campo_numerico("Monto del préstamo (USD)", "ev-monto", 1, None),

            campo_numerico("Tasa de interés (%)", "ev-tasa", 0, 30, 0.001),
            campo_numerico("Plazo (meses)", "ev-plazo", 1, 480),

            campo_lista(
                "Propósito",
                "ev-proposito",
                opciones_proposito
            ),

            campo_lista(
                "Tipo de propiedad",
                "ev-propiedad",
                opciones_propiedad
            ),

            campo_lista(
                "Estado geográfico",
                "ev-estado",
                opciones_estado
            )

        ], style={
            "display": "grid",
            "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
            "gap": "16px",
            "marginTop": "18px"
        }),

        html.Div([

            html.Button(
                "Limpiar",
                id="ev-limpiar",
                n_clicks=0,
                style={
                    "padding": "12px 20px",
                    "backgroundColor": "white",
                    "border": "1px solid #ccd6e0",
                    "borderRadius": "6px",
                    "cursor": "pointer"
                }
            ),

            html.Button(
                "Comparar perfil",
                id="ev-comparar",
                n_clicks=0,
                style={
                    "padding": "12px 20px",
                    "backgroundColor": "#17324d",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "6px",
                    "cursor": "pointer"
                }
            )

        ], style={
            "display": "flex",
            "justifyContent": "flex-end",
            "gap": "10px",
            "marginTop": "24px"
        }),

        html.Div(id="ev-mensaje", style={
            "marginTop": "12px",
            "color": "#b45309",
            "fontSize": "13px"
        })

    ], style=TARJETA)

    resultado_predictivo = html.Div([

        html.H4("Resultado predictivo", style={"marginTop": "0"}),

        html.P(
            "Probabilidad de alcanzar 90 o más días de mora "
            "durante los primeros 12 meses",
            style={"color": "#64748b", "fontSize": "13px"}
        ),

        html.H2("No disponible", style={
            "color": "#17324d",
            "marginBottom": "12px"
        }),

        html.P(
            "La estimación estará disponible cuando se integre "
            "el modelo predictivo desarrollado por el equipo.",
            style={"fontSize": "13px", "color": "#64748b"}
        ),

        html.Hr(),

        html.H4("Escenarios de tolerancia al riesgo"),

        html.P(
            "Conservadora · Moderada · Flexible",
            style={"color": "#64748b"}
        ),

        html.P(
            "Los umbrales se aplicarán cuando exista una "
            "probabilidad estimada. Por ahora no se clasifica el riesgo.",
            style={"fontSize": "12px", "color": "#64748b"}
        )

    ], style=TARJETA)

    perfil = html.Div([

        html.H4("Perfil frente a la cartera", style={
            "marginTop": "0"
        }),

        html.P(
            "Comparación descriptiva con las originaciones de 2026 Q1",
            style={"fontSize": "12px", "color": "#64748b"}
        ),

        html.Div(
            "Completa la solicitud y selecciona Comparar perfil.",
            id="ev-perfil",
            style={"fontSize": "13px", "marginTop": "15px"}
        )

    ], style=TARJETA)

    return html.Div([

        html.H2("Evaluación de solicitudes"),

        html.P(
            "¿Cuál es la probabilidad de que el crédito alcance "
            "90 o más días de mora durante sus primeros 12 meses?",
            style={"color": "#64748b"}
        ),

        html.Div([

            formulario,

            html.Div([
                resultado_predictivo,
                perfil
            ], style={
                "display": "grid",
                "gap": "14px",
                "alignContent": "start"
            })

        ], style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(360px, 1fr))",
            "gap": "14px",
            "marginTop": "20px"
        })

    ], style={
        "padding": "24px",
        "backgroundColor": "#f2f5fa",
        "minHeight": "100vh"
    })


# ==========================================
# 3. INTERACCIONES DEL FORMULARIO
# ==========================================

def registrar_callbacks_evaluacion(app, datos):

    # Cargar el ejemplo o limpiar el formulario

    @app.callback(
        Output("ev-fico", "value"),
        Output("ev-dti", "value"),
        Output("ev-ltv", "value"),
        Output("ev-monto", "value"),
        Output("ev-tasa", "value"),
        Output("ev-plazo", "value"),
        Output("ev-proposito", "value"),
        Output("ev-propiedad", "value"),
        Output("ev-estado", "value"),

        Input("ev-ejemplo", "n_clicks"),
        Input("ev-limpiar", "n_clicks"),
        prevent_initial_call=True
    )
    def cargar_o_limpiar(ejemplo, limpiar):

        if ctx.triggered_id == "ev-ejemplo":
            return (
                712, 41, 88, 412000, 6.375,
                360, "P", "SF", "FL"
            )

        return (None,) * 9

    # Comparar la solicitud con la cartera real

    @app.callback(
        Output("ev-mensaje", "children"),
        Output("ev-perfil", "children"),

        Input("ev-comparar", "n_clicks"),

        State("ev-fico", "value"),
        State("ev-dti", "value"),
        State("ev-ltv", "value"),
        State("ev-monto", "value"),
        State("ev-tasa", "value"),
        State("ev-plazo", "value"),
        State("ev-proposito", "value"),
        State("ev-propiedad", "value"),
        State("ev-estado", "value"),

        prevent_initial_call=True
    )
    def comparar_perfil(
        n, fico, dti, ltv, monto, tasa, plazo,
        proposito, propiedad, estado
    ):

        valores = [
            fico, dti, ltv, monto, tasa, plazo,
            proposito, propiedad, estado
        ]

        if any(x is None or x == "" for x in valores):
            return (
                "Completa todos los campos antes de comparar.",
                "No se ha realizado la comparación."
            )

        if not (
            300 <= fico <= 850
            and 0 <= dti <= 100
            and 0 <= ltv <= 200
            and monto > 0
            and 0 < tasa <= 30
            and 1 <= plazo <= 480
        ):
            return (
                "Revisa los valores: hay campos fuera del rango permitido.",
                "No se ha realizado la comparación."
            )

        # Promedios de originaciones reales 2026 Q1

        referencia = datos[
            datos["trimestre"] == "2026Q1"
        ]

        if referencia.empty:
            return (
                "No se encontraron datos de referencia para 2026 Q1.",
                "Comparación no disponible."
            )

        def media(columna_suma, columna_n):
            n = referencia[columna_n].sum()
            return referencia[columna_suma].sum() / n if n else None

        referencias = [
            ("Puntaje FICO", fico, media("suma_puntaje", "n_puntaje"), "", 1),
            ("LTV", ltv, media("suma_ltv", "n_ltv"), " %", 1),
            ("DTI", dti, media("suma_dti", "n_dti"), " %", 1),
            ("Tasa de interés", tasa, media("suma_tasa", "n_tasa"), " %", 3),
            (
                "Monto del préstamo",
                monto,
                referencia["monto_total"].sum()
                / referencia["creditos"].sum(),
                " USD",
                0
            )
        ]

        filas = []

        for nombre, solicitud, cartera, unidad, decimales in referencias:

            if cartera is None:
                continue

            diferencia = solicitud - cartera

            filas.append(html.Tr([

                html.Td(nombre, style={"padding": "10px"}),

                html.Td(
                    numero(solicitud, decimales) + unidad,
                    style={"padding": "10px", "fontWeight": "bold"}
                ),

                html.Td(
                    numero(cartera, decimales) + unidad,
                    style={"padding": "10px"}
                ),

                html.Td(
                    ("+" if diferencia > 0 else "")
                    + numero(diferencia, decimales)
                    + unidad,
                    style={"padding": "10px"}
                )

            ], style={"borderBottom": "1px solid #edf1f5"}))

        tabla = html.Table([

            html.Thead(html.Tr([
                html.Th("Variable"),
                html.Th("Solicitud"),
                html.Th("Cartera"),
                html.Th("Diferencia")
            ])),

            html.Tbody(filas)

        ], style={
            "width": "100%",
            "fontSize": "12px",
            "borderCollapse": "collapse"
        })

        return (
            "Comparación descriptiva realizada. "
            "Este resultado no es una predicción de riesgo.",
            tabla
        )

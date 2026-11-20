from dash import html, dcc, Input, Output, ctx

TARJETA = {
    "backgroundColor": "#242930", "padding": "24px",
    "border": "1px solid #3b4450", "borderRadius": "12px"
}
TEXTO = {"color": "#b8c5d6", "fontSize": "14px", "lineHeight": "1.6"}
CAMPO = {
    "width": "100%", "height": "44px", "padding": "10px 12px",
    "backgroundColor": "#ffffff", "color": "#17212f",
    "border": "1px solid #bac5d3", "borderRadius": "7px",
    "boxSizing": "border-box", "fontSize": "16px", "fontWeight": "500"
}
MENSAJE_INICIAL = "Completa la solicitud y selecciona Comparar perfil."


def numero(valor, decimales=1):
    return f"{valor:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formatear_valor(valor, unidad, decimales):
    if unidad == "USD":
        return "USD " + numero(valor, decimales)
    if unidad == "FICO":
        return numero(valor, decimales)
    return numero(valor, decimales) + " %"


def formatear_diferencia(valor, unidad, decimales):
    signo = "+" if valor > 0 else "−" if valor < 0 else ""
    if unidad == "USD":
        return signo + "USD " + numero(abs(valor), decimales)
    if unidad == "FICO":
        return signo + numero(abs(valor), decimales) + " puntos"
    return signo + numero(abs(valor), decimales) + " pp"


def campo_numerico(titulo, identificador, minimo, maximo, paso=1):
    return html.Div([
        html.Label(titulo, style={"fontSize": "14px", "color": "#cbd5e1", "display": "block", "marginBottom": "8px"}),
        dcc.Input(
            id=identificador, type="number", min=minimo, max=maximo, step=paso,
            className="mr-input-numerico", style=CAMPO
        )
    ])


def campo_lista(titulo, identificador, opciones):
    return html.Div([
        html.Label(titulo, style={"fontSize": "14px", "color": "#cbd5e1", "display": "block", "marginBottom": "8px"}),
        dcc.Dropdown(id=identificador, options=opciones, placeholder="Seleccionar", clearable=True)
    ])


def tarjeta_comparacion(nombre, solicitud, cartera, unidad, decimales, criterio):
    diferencia = float(solicitud) - float(cartera)
    tolerancia = 1e-9
    if abs(diferencia) < tolerancia:
        flecha, color, lectura = "→", "#e2e8f0", "Igual al promedio"
    else:
        flecha = "↑" if diferencia > 0 else "↓"
        if criterio == "alto_mejor":
            color = "#34d399" if diferencia > 0 else "#fb7185"
        elif criterio == "bajo_mejor":
            color = "#fb7185" if diferencia > 0 else "#34d399"
        else:
            color = "#93c5fd"  # Diferencia informativa, no evaluación de riesgo.
        lectura = "Frente al promedio histórico"

    return html.Div([
        html.P(nombre, style={"fontSize": "13px", "color": "#b8c5d6", "margin": "0 0 8px"}),
        html.Div(formatear_valor(solicitud, unidad, decimales), style={
            "color": "#ffffff", "fontSize": "24px", "fontWeight": "700", "overflowWrap": "anywhere"
        }),
        html.Div(f"{flecha} {formatear_diferencia(diferencia, unidad, decimales)}", style={
            "color": color, "fontSize": "15px", "fontWeight": "700", "marginTop": "10px"
        }),
        html.P(f"Promedio: {formatear_valor(cartera, unidad, decimales)}", style={
            "fontSize": "12px", "color": "#aeb9c8", "margin": "8px 0 0"
        }),
        html.P(lectura, style={"fontSize": "11px", "color": "#94a3b8", "margin": "4px 0 0"})
    ], style={
        "backgroundColor": "#1b222b", "border": "1px solid #3b4450",
        "borderRadius": "10px", "padding": "18px", "minWidth": 0
    })


def crear_evaluacion(estados, nombres_estados):
    opciones_proposito = [
        {"label": "Compra", "value": "P"},
        {"label": "Refinanciación sin retiro", "value": "N"},
        {"label": "Refinanciación con retiro", "value": "C"},
    ]
    opciones_propiedad = [
        {"label": "Vivienda unifamiliar", "value": "SF"},
        {"label": "Unidad en proyecto planificado", "value": "PU"},
        {"label": "Condominio", "value": "CO"},
        {"label": "Vivienda prefabricada", "value": "MH"},
        {"label": "Cooperativa", "value": "CP"},
    ]
    opciones_estado = [{"label": nombres_estados.get(x, x), "value": x} for x in estados]

    formulario = html.Div([
        html.Div([
            html.Div([
                html.H4("Datos de la solicitud", style={"color": "#ffffff", "fontSize": "19px", "margin": 0}),
                html.P("Características del crédito al momento de la originación", style=TEXTO),
            ]),
            html.Button("Cargar caso de ejemplo", id="ev-ejemplo", n_clicks=0, style={
                "padding": "10px 14px", "backgroundColor": "#31445b", "color": "#ffffff",
                "border": "1px solid #51657d", "borderRadius": "7px", "cursor": "pointer"
            })
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "12px", "flexWrap": "wrap"}),
        html.Div([
            campo_numerico("Puntaje FICO", "ev-fico", 300, 850),
            campo_numerico("DTI (%)", "ev-dti", 0, 100, 0.1),
            campo_numerico("LTV (%)", "ev-ltv", 0, 200, 0.1),
            campo_numerico("Monto del préstamo (USD)", "ev-monto", 1, None),
            campo_numerico("Tasa de interés (%)", "ev-tasa", 0, 30, 0.001),
            campo_numerico("Plazo (meses)", "ev-plazo", 1, 480),
            campo_lista("Propósito", "ev-proposito", opciones_proposito),
            campo_lista("Tipo de propiedad", "ev-propiedad", opciones_propiedad),
            campo_lista("Estado geográfico", "ev-estado", opciones_estado),
        ], className="mr-evaluacion-campos", style={
            "display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
            "gap": "16px", "marginTop": "18px"
        }),
        html.Div([
            html.Button("Limpiar", id="ev-limpiar", n_clicks=0, style={
                "padding": "12px 20px", "backgroundColor": "#303945", "color": "#ffffff",
                "border": "1px solid #596579", "borderRadius": "7px", "cursor": "pointer"
            }),
            html.Button("Comparar perfil", id="ev-comparar", n_clicks=0, style={
                "padding": "12px 20px", "backgroundColor": "#206bc4", "color": "#ffffff",
                "border": "none", "borderRadius": "7px", "cursor": "pointer", "fontWeight": "700"
            })
        ], style={"display": "flex", "justifyContent": "flex-end", "gap": "10px", "marginTop": "24px"}),
        html.Div(id="ev-mensaje", style={"marginTop": "12px", "color": "#fbbf24", "fontSize": "14px"})
    ], className="mr-panel", style=TARJETA)

    resultado_predictivo = html.Div([
        html.H4("Resultado predictivo", style={"color": "#ffffff", "fontSize": "19px", "marginTop": 0}),
        html.P("Probabilidad de alcanzar 90 o más días de mora durante los primeros 12 meses", style=TEXTO),
        html.H2("—", id="ev-probabilidad", style={
            "color": "#ffffff", "fontSize": "32px", "fontWeight": "700", "marginBottom": "12px"
        }),
        html.P(MENSAJE_INICIAL, id="ev-nota-modelo", style=TEXTO),
        html.Hr(style={"borderColor": "#445161"}),
        html.H4("Escenarios de tolerancia al riesgo", style={"color": "#ffffff", "fontSize": "17px"}),
        html.P("Conservadora · Moderada · Flexible", style=TEXTO),
        html.Div(id="ev-escenarios"),
        html.P("Los escenarios cambian los umbrales de clasificación, no la probabilidad estimada.", style=TEXTO)
    ], className="mr-panel", style=TARJETA)

    perfil = html.Div([
        html.H4("Perfil frente a la cartera", style={"color": "#ffffff", "fontSize": "19px", "marginTop": 0}),
        html.P("Comparación con los promedios de originaciones de 2026 Q1", style=TEXTO),
        html.Div(MENSAJE_INICIAL, id="ev-perfil", style={"marginTop": "15px", "overflowX": "auto"})
    ], className="mr-panel", style=TARJETA)

    return html.Div([
        html.H2("Evaluación de solicitudes", style={"color": "#ffffff", "fontSize": "26px"}),
        html.P("¿Cuál es la probabilidad de que el crédito alcance 90 o más días de mora durante sus primeros 12 meses?", style=TEXTO),
        html.Div([
            formulario,
            html.Div([resultado_predictivo, perfil], style={
                "display": "grid", "gap": "14px", "alignContent": "start"
            })
        ], style={
            "display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(360px, 1fr))",
            "gap": "14px", "marginTop": "20px"
        })
    ], className="mr-screen mr-evaluacion", style={
        "padding": "24px", "backgroundColor": "#171b21", "minHeight": "100vh"
    })


def registrar_callbacks_evaluacion(app, datos):
    @app.callback(
        Output("ev-fico", "value"), Output("ev-dti", "value"),
        Output("ev-ltv", "value"), Output("ev-monto", "value"),
        Output("ev-tasa", "value"), Output("ev-plazo", "value"),
        Output("ev-proposito", "value"), Output("ev-propiedad", "value"),
        Output("ev-estado", "value"),
        Input("ev-ejemplo", "n_clicks"), Input("ev-limpiar", "n_clicks"),
        prevent_initial_call=True
    )
    def cargar_o_limpiar(ejemplo, limpiar):
        if ctx.triggered_id == "ev-ejemplo":
            return (712, 41, 88, 412000, 6.375, 360, "P", "SF", "FL")
        return (None,) * 9

    @app.callback(
        Output("ev-mensaje", "children"), Output("ev-perfil", "children"),
        Input("ev-comparar", "n_clicks"), Input("ev-ejemplo", "n_clicks"),
        Input("ev-limpiar", "n_clicks"),
        Input("ev-fico", "value"), Input("ev-dti", "value"),
        Input("ev-ltv", "value"), Input("ev-monto", "value"),
        Input("ev-tasa", "value"), Input("ev-plazo", "value"),
        Input("ev-proposito", "value"), Input("ev-propiedad", "value"),
        Input("ev-estado", "value"),
        prevent_initial_call=True
    )
    def comparar_perfil(comparar, ejemplo, limpiar, fico, dti, ltv, monto,
                        tasa, plazo, proposito, propiedad, estado):
        if ctx.triggered_id != "ev-comparar":
            return "", MENSAJE_INICIAL
        valores = [fico, dti, ltv, monto, tasa, plazo, proposito, propiedad, estado]
        if any(x is None or x == "" for x in valores):
            return "Completa todos los campos antes de comparar.", MENSAJE_INICIAL
        try:
            valido = (
                300 <= fico <= 850 and 0 <= dti <= 100 and
                0 <= ltv <= 200 and monto > 0 and 0 < tasa <= 30 and
                1 <= plazo <= 480 and plazo == int(plazo)
            )
        except (TypeError, ValueError):
            valido = False
        if not valido:
            return "Revisa los valores: hay campos fuera del rango permitido.", MENSAJE_INICIAL

        referencia = datos[datos["trimestre"] == "2026Q1"]
        if referencia.empty:
            return "No se encontraron originaciones de 2026 Q1.", "Comparación no disponible."

        def media(suma, conteo):
            cantidad = referencia[conteo].sum()
            return float(referencia[suma].sum() / cantidad) if cantidad > 0 else None

        creditos = referencia["creditos"].sum()
        monto_promedio = float(referencia["monto_total"].sum() / creditos) if creditos > 0 else None
        variables = [
            ("Puntaje FICO", fico, media("suma_puntaje", "n_puntaje"), "FICO", 1, "alto_mejor"),
            ("LTV", ltv, media("suma_ltv", "n_ltv"), "%", 1, "bajo_mejor"),
            ("DTI", dti, media("suma_dti", "n_dti"), "%", 1, "bajo_mejor"),
            ("Tasa de interés", tasa, media("suma_tasa", "n_tasa"), "%", 3, "neutro"),
            ("Monto del préstamo", monto, monto_promedio, "USD", 0, "neutro"),
        ]
        tarjetas = [
            tarjeta_comparacion(nombre, solicitud, cartera, unidad, decimales, criterio)
            for nombre, solicitud, cartera, unidad, decimales, criterio in variables
            if cartera is not None
        ]
        if not tarjetas:
            return "Sin promedios disponibles para comparar.", "Comparación no disponible."
        contenido = html.Div([
            html.Div(tarjetas, style={
                "display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(185px, 1fr))",
                "gap": "12px"
            }),
            html.P(
                "Las flechas muestran la diferencia frente al promedio de 2026 Q1. "
                "Rojo o verde indican una lectura orientativa para FICO, LTV y DTI; "
                "azul es informativo. No sustituyen la predicción del modelo.",
                style={"fontSize": "12px", "color": "#aeb9c8", "marginTop": "16px"}
            )
        ])
        return "Comparación actualizada correctamente.", contenido


from dash import Dash, dcc, html

# Aplicación principal
app = Dash(__name__)
app.title = "Mortgage Risk Analytics"

# Tarjetas para los indicadores
def tarjeta_kpi(titulo):
    return html.Div([
        html.P(titulo, style={"color": "#64748b", "fontSize": "13px"}),
        html.H2("—", style={"color": "#17324d", "margin": "10px 0"}),
        html.P("Pendiente de conectar datos",
               style={"color": "#94a3b8", "fontSize": "12px"})
    ], style={
        "backgroundColor": "white",
        "padding": "18px",
        "borderRadius": "10px",
        "border": "1px solid #dce3eb"
    })

# Pantalla Panorama
panorama = html.Div([
    html.H2("Panorama de cartera"),
    html.P("Originaciones de créditos hipotecarios Freddie Mac"),

    html.Div([
        tarjeta_kpi("Créditos originados"),
        tarjeta_kpi("Monto total originado"),
        tarjeta_kpi("FICO promedio"),
        tarjeta_kpi("LTV promedio"),
        tarjeta_kpi("DTI promedio")
    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
        "gap": "12px"
    })

], style={"padding": "24px"})

# Estructura de las tres pantallas
app.layout = html.Div([
    html.Div("MR  |  Mortgage Risk Analytics", style={
        "backgroundColor": "#122f49",
        "color": "white",
        "padding": "20px",
        "fontSize": "20px",
        "fontWeight": "bold"
    }),

    dcc.Tabs([
        dcc.Tab(label="Panorama", children=panorama),

        dcc.Tab(label="Evaluación", children=html.Div([
            html.H2("Evaluación de solicitudes"),
            html.P("Pendiente de integrar el modelo predictivo.")
        ], style={"padding": "24px"})),

        dcc.Tab(label="Seguimiento", children=html.Div([
            html.H2("Monitoreo y alertas"),
            html.P("Pendiente de incorporar los indicadores de seguimiento.")
        ], style={"padding": "24px"}))
    ])

], style={
    "backgroundColor": "#f2f5fa",
    "minHeight": "100vh",
    "fontFamily": "Arial, sans-serif"
})

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True)

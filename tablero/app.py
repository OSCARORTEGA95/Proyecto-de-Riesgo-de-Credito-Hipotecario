
from dash import Dash, dcc, html
from pathlib import Path

# Importar las tres pantallas
from panorama import (
    crear_panorama,
    registrar_callbacks_panorama,
    datos,
    estados,
    nombres_estados
)

from evaluacion import (
    crear_evaluacion,
    registrar_callbacks_evaluacion
)

from seguimiento import (
    crear_seguimiento,
    registrar_callbacks_seguimiento
)

from prediccion import registrar_callbacks_prediccion


# ==========================================
# 1. APLICACION PRINCIPAL
# ==========================================

app = Dash(
    __name__,
    assets_folder=str(Path(__file__).resolve().parent / "assets"),
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/@tabler/core@1.7.0/dist/css/tabler.min.css"
    ]
)
app.title = "Mortgage Risk Analytics"

# Servidor para despliegue posterior
server = app.server



# ==========================================
# 2. NAVEGACION DEL TABLERO
# ==========================================

estilo_menu = {
    "padding": "18px 22px",
    "textAlign": "left",
    "fontSize": "14px",
    "fontWeight": "500",
    "color": "#52677f",
    "backgroundColor": "#ffffff",
    "border": "none",
    "borderLeft": "4px solid transparent",
}

estilo_menu_activo = {
    **estilo_menu,
    "backgroundColor": "#eaf2fc",
    "color": "#206bc4",
    "fontWeight": "700",
    "borderLeft": "4px solid #206bc4",
}

app.layout = html.Div(
    [
        # Encabezado principal
        html.Div(
            [
                html.Div(
                    "MR",
                    style={
                        "backgroundColor": "#ffffff",
                        "color": "#17324d",
                        "padding": "10px 13px",
                        "borderRadius": "10px",
                        "fontSize": "18px",
                        "fontWeight": "700",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            "Mortgage Risk Analytics",
                            style={
                                "fontSize": "22px",
                                "fontWeight": "700",
                            },
                        ),
                        html.Div(
                            "Plataforma de análisis de riesgo hipotecario",
                            style={
                                "fontSize": "12px",
                                "color": "#cbd5e1",
                            },
                        ),
                    ]
                ),
            ],
            style={
                "backgroundColor": "#122f49",
                "color": "#ffffff",
                "padding": "18px 28px",
                "display": "flex",
                "alignItems": "center",
                "gap": "16px",
            },
        ),

        # Navegación lateral
        
dcc.Tabs(
    [
        dcc.Tab(
            label="Panorama",
            children=crear_panorama(),
        ),
        dcc.Tab(
            label="Evaluación",
            children=crear_evaluacion(
                estados,
                nombres_estados
            ),
        ),
        dcc.Tab(
            label="Seguimiento",
            children=crear_seguimiento(
                nombres_estados
            ),
        ),
    ],
    vertical=False,
    style={
        "width": "100%",
        "backgroundColor": "#ffffff",
    },
    colors={
        "border": "#e3e9f2",
        "primary": "#206bc4",
        "background": "#ffffff",
    },
)

    ], className="mr-theme", style={
        "backgroundColor": "#f3f6fb",
        "minHeight": "100vh",
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "margin": 0,
    },
)



# ==========================================
# 3. ACTIVAR FUNCIONES DE CADA MODULO
# ==========================================

registrar_callbacks_panorama(app)

registrar_callbacks_evaluacion(
    app,
    datos
)

registrar_callbacks_seguimiento(
    app,
    nombres_estados
)

registrar_callbacks_prediccion(app)


# ==========================================
# 4. EJECUTAR EL TABLERO
# ==========================================

# En el EC2 el puerto 8050 lo usa MLflow: allí se ejecuta con
#   TABLERO_HOST=0.0.0.0 TABLERO_PUERTO=8060 TABLERO_DEBUG=0 python tablero/app.py
if __name__ == "__main__":
    import os

    app.run(
        host=os.environ.get("TABLERO_HOST", "127.0.0.1"),
        port=int(os.environ.get("TABLERO_PUERTO", "8050")),
        debug=os.environ.get("TABLERO_DEBUG", "1") == "1",
        dev_tools_ui=False,
    )

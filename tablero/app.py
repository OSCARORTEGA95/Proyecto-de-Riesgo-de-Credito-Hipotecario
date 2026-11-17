
from dash import Dash, dcc, html

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

app = Dash(__name__)
app.title = "Mortgage Risk Analytics"

# Servidor para despliegue posterior
server = app.server


# ==========================================
# 2. NAVEGACION DEL TABLERO
# ==========================================

app.layout = html.Div([

    html.Div(
        "MR  |  Mortgage Risk Analytics",
        style={
            "backgroundColor": "#122f49",
            "color": "white",
            "padding": "20px",
            "fontSize": "20px",
            "fontWeight": "bold"
        }
    ),

    dcc.Tabs([

        # Modulo 1: Panorama
        dcc.Tab(
            label="Panorama",
            children=crear_panorama()
        ),

        # Modulo 2: Evaluacion
        dcc.Tab(
            label="Evaluación",
            children=crear_evaluacion(
                estados,
                nombres_estados
            )
        ),

        # Modulo 3: Seguimiento
        dcc.Tab(
            label="Seguimiento",
            children=crear_seguimiento(
                nombres_estados
            )
        )

    ])

], style={
    "backgroundColor": "#f2f5fa",
    "minHeight": "100vh",
    "fontFamily": "Arial, sans-serif",
    "margin": 0
})


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

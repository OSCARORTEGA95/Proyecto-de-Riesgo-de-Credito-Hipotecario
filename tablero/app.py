
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


# ==========================================
# 4. EJECUTAR EL TABLERO
# ==========================================

if __name__ == "__main__":
    app.run(debug=True, dev_tools_ui=False)

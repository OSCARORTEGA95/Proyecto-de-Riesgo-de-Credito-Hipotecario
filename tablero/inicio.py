
from dash import html


def tarjeta_modulo(numero, titulo, descripcion, detalle):
    return html.Div(
        [
            html.Div(numero, style={
                "fontSize": "13px",
                "fontWeight": "700",
                "color": "#7db8f0",
                "marginBottom": "14px",
            }),
            html.H3(titulo, style={
                "fontSize": "20px",
                "fontWeight": "700",
                "color": "#ffffff",
                "marginBottom": "10px",
            }),
            html.P(descripcion, style={
                "fontSize": "15px",
                "color": "#cbd5e1",
                "lineHeight": "1.7",
            }),
            html.P(detalle, style={
                "fontSize": "13px",
                "color": "#91a5bb",
                "marginTop": "18px",
            }),
        ],
        style={
            "backgroundColor": "#242930",
            "border": "1px solid #3c4654",
            "borderRadius": "12px",
            "padding": "26px",
            "flex": "1 1 260px",
            "minWidth": "0",
        },
    )


def crear_inicio():
    return html.Div(
        [
            # Presentación principal
            html.Div(
                [
                    html.P(
                        "PLATAFORMA DE ANÁLISIS DE RIESGO HIPOTECARIO",
                        style={
                            "fontSize": "13px",
                            "fontWeight": "700",
                            "letterSpacing": "1.5px",
                            "color": "#79b8f5",
                            "marginBottom": "18px",
                        },
                    ),
                    html.H1(
                        "Información para decisiones crediticias más informadas",
                        style={
                            "fontSize": "clamp(28px, 3vw, 44px)",
                            "fontWeight": "750",
                            "color": "#ffffff",
                            "lineHeight": "1.2",
                            "maxWidth": "850px",
                            "marginBottom": "22px",
                        },
                    ),
                    html.P(
                        "Mortgage Risk Analytics integra información histórica "
                        "de créditos hipotecarios y herramientas analíticas "
                        "para explorar originaciones, evaluar solicitudes "
                        "y monitorear el comportamiento de la cartera.",
                        style={
                            "fontSize": "17px",
                            "color": "#cbd5e1",
                            "lineHeight": "1.8",
                            "maxWidth": "850px",
                        },
                    ),
                    html.Div(
                        [
                            html.Span("Análisis histórico"),
                            html.Span("Evaluación predictiva"),
                            html.Span("Seguimiento de cartera"),
                        ],
                        className="mr-inicio-etiquetas",
                        style={
                            "display": "flex",
                            "flexWrap": "wrap",
                            "gap": "14px",
                            "marginTop": "28px",
                        },
                    ),
                ],
                style={
                    "background": (
                        "linear-gradient(120deg, #122f49 0%, "
                        "#1d3045 55%, #242930 100%)"
                    ),
                    "border": "1px solid #39506a",
                    "borderRadius": "16px",
                    "padding": "clamp(28px, 4vw, 58px)",
                    "marginBottom": "26px",
                },
            ),

            html.H2(
                "Explora nuestras herramientas",
                style={
                    "fontSize": "25px",
                    "color": "#ffffff",
                    "fontWeight": "700",
                    "marginBottom": "20px",
                },
            ),

            # Módulos disponibles
            html.Div(
                [
                    tarjeta_modulo(
                        "01 / PANORAMA",
                        "Conoce la originación hipotecaria",
                        "Consulta indicadores de créditos originados, "
                        "montos, puntajes FICO y características "
                        "financieras de las solicitudes.",
                        "Utiliza filtros por período, estado y características del crédito.",
                    ),
                    tarjeta_modulo(
                        "02 / EVALUACIÓN",
                        "Analiza el riesgo de una solicitud",
                        "Ingresa los datos de un crédito y consulta "
                        "la probabilidad estimada de alcanzar 90 "
                        "o más días de mora durante sus primeros 12 meses.",
                        "La predicción requiere que el servicio de la API esté disponible.",
                    ),
                    tarjeta_modulo(
                        "03 / SEGUIMIENTO",
                        "Monitorea la evolución de la cartera",
                        "Explora créditos activos, saldos y niveles "
                        "de mora de 30, 60 y 90 días, con información "
                        "histórica mensual.",
                        "Compara períodos y analiza el comportamiento de la cartera.",
                    ),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "18px",
                    "marginBottom": "26px",
                },
            ),

            # Información metodológica
            html.Div(
                [
                    html.H3(
                        "¿Cómo utilizar la plataforma?",
                        style={
                            "fontSize": "20px",
                            "color": "#ffffff",
                            "fontWeight": "700",
                            "marginBottom": "14px",
                        },
                    ),
                    html.P(
                        "Selecciona una de las pestañas superiores para "
                        "acceder al módulo que deseas consultar. "
                        "En Panorama y Seguimiento podrás aplicar filtros "
                        "a los indicadores históricos. En Evaluación "
                        "podrás ingresar una solicitud y consultar el "
                        "resultado del modelo predictivo.",
                        style={
                            "fontSize": "15px",
                            "color": "#cbd5e1",
                            "lineHeight": "1.8",
                        },
                    ),
                    html.P(
                        "Los indicadores históricos describen la cartera "
                        "observada. Las probabilidades del modelo son "
                        "estimaciones analíticas y no constituyen, por sí "
                        "solas, una decisión de aprobación o rechazo.",
                        style={
                            "fontSize": "14px",
                            "color": "#91a5bb",
                            "lineHeight": "1.7",
                            "marginTop": "14px",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#242930",
                    "border": "1px solid #3c4654",
                    "borderRadius": "12px",
                    "padding": "28px",
                },
            ),
        ],
        style={
            "backgroundColor": "#171b21",
            "padding": "30px 24px 45px",
            "minHeight": "calc(100vh - 180px)",
        },
    )

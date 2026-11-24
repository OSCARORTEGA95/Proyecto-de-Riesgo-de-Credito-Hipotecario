# Árbol de decisión — Riesgo hipotecario P1

## Objetivo

Desarrollar un modelo de árbol de decisión para estimar el riesgo de que un crédito hipotecario alcance 90 o más días de mora durante sus primeros 12 meses.

## Datos utilizados

Se utilizaron archivos Parquet preparados para la pregunta P1 y descargados mediante DVC desde el almacenamiento S3 del proyecto.

La variable objetivo es `y_p1`:

- `0`: no alcanza 90 o más días de mora.
- `1`: alcanza 90 o más días de mora.

Las variables explicativas incluyen FICO, DTI, LTV, tasa, monto, plazo, propósito, ocupación y características de la propiedad.

## Archivos desarrollados

| Archivo | Función |
|---|---|
| `arbol_decision_p1.py` | Entrenamiento inicial y validación. |
| `comparar_umbrales_p1.py` | Evaluación de diferentes umbrales. |
| `ajustar_arbol_p1.py` | Comparación de tres configuraciones del árbol. |
| `comparar_periodos_p1.py` | Comparación de períodos de entrenamiento desde 2019. |
| `evaluar_arbol_final_p1.py` | Entrenamiento con 2022–2023 y prueba con 2024. |
| `registrar_mlflow_p1.py` | Registro de resultados históricos en MLflow. |
| `entrenar_mlflow_p1.py` | Entrenamiento directo con seguimiento en MLflow. |

## Metodología

Primero se entrenaron tres configuraciones del árbol utilizando 2022 y se validaron con 2023. Se seleccionó el árbol flexible por su mayor Average Precision.

Después se volvió a entrenar con 2022–2023 y se evaluó con 2024.

También se compararon períodos históricos desde 2019, con y sin las cohortes críticas asociadas a la pandemia. Esta comparación utilizó muestras de hasta 150.000 créditos por trimestre y se validó con 2023.

## Resultados

| Indicador | Árbol flexible — Prueba 2024 |
|---|---:|
| ROC-AUC | 0,81603 |
| Average Precision | 0,02430 |
| Precision, umbral 2 % | 2,72 % |
| Recall, umbral 2 % | 38,29 % |
| Alertas generadas | 72.875 |

Estos resultados corresponden a una configuración candidata, no necesariamente al modelo seleccionado por el equipo.

## MLflow

Se registraron ocho ejecuciones en el experimento `Riesgo_Hipotecario_P1_Arbol_Ana`, utilizando MLflow instalado en AWS EC2.

Siete ejecuciones corresponden al registro de resultados históricos y una corresponde a un entrenamiento directo con seguimiento MLflow.

## Estado

El código y las evaluaciones están desarrollados. La integración con la API y el tablero queda pendiente de la selección del modelo del equipo.
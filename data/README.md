# Datos del proyecto

Los datos de Freddie Mac no se pueden redistribuir, así que no están en Git. Se versionan con DVC en el bucket privado `s3://riesgodecreditohipotecario` (remoto `aws-remote`). En Git solo quedan los punteros (`*.dvc`, `dvc.lock`), las métricas del ETL y este archivo.

## Cómo descargar

Con las credenciales del Learner Lab configuradas, desde la raíz del repositorio:

```bash
dvc pull --glob "data/modelado/p1/*"            # tabla de la pregunta 1, todos los trimestres
dvc pull "data/modelado/p1/trimestre=2019Q1"    # un solo trimestre
dvc pull --glob "data/limpios/originacion/*"
dvc pull --glob "data/limpios/desempeno/*"      # la más pesada
```

Las tablas se leen con pandas: `pd.read_parquet("data/modelado/p1")` lee todos los trimestres descargados y agrega la columna `trimestre`.

## Cómo regenerar

`dvc repro` ejecuta el ETL (`src/riesgo_credito/procesamiento/etl.py`) para cada trimestre de `params.yaml` cuyo zip, código o parámetros hayan cambiado. Requiere Java 17 o superior y `pyspark==4.0.1`.

## Capas

```
data/crudos/historical_data_AAAATn.zip     zip trimestral de Freddie Mac (DVC)
data/limpios/originacion/trimestre=AAAATn  un crédito por fila
data/limpios/desempeno/trimestre=AAAATn    un crédito por mes
data/modelado/p1/trimestre=AAAATn          pregunta 1: variables de la solicitud y etiqueta
data/metricas/etl/AAAATn.json              conteos de cada ejecución (en Git)
```

## data/limpios/originacion

Columnas del archivo de originación con sus tipos. Los códigos de no disponible (por ejemplo 9999 en el puntaje o 999 en deuda-ingreso) son nulos. Un vacío en `msa` significa zona no metropolitana y en `special_eligibility` que no aplica.

Se descartan `ppm_flag`, `amort_type`, `io_flag` y `vantage_score`, que no traen información, y `valuation_method`, no disponible en el 94 % de los créditos.

| Columna | Tipo | Descripción |
|---|---|---|
| loan_id | texto | Identificador del crédito |
| credit_score | entero | Puntaje crediticio |
| first_payment_date | entero | Mes del primer pago, AAAAMM |
| first_time_homebuyer | texto | Primera vivienda: Y, N |
| maturity_date | entero | Mes de vencimiento, AAAAMM |
| msa | texto | Área metropolitana |
| mi_pct | entero | Porcentaje de seguro hipotecario |
| num_units | entero | Unidades del inmueble |
| occupancy | texto | P residencia principal, I inversión, S segunda vivienda |
| ocltv | entero | Préstamo-valor combinada (%) |
| dti | entero | Deuda-ingreso (%) |
| orig_upb | decimal | Monto original (USD) |
| oltv | entero | Préstamo-valor (%) |
| orig_rate | decimal | Tasa de interés original (%) |
| channel | texto | R minorista, C corresponsal, B corredor |
| state | texto | Estado |
| prop_type | texto | Tipo de inmueble (SF, PU, CO, MH, CP) |
| zip3 | texto | Tres primeros dígitos del código postal |
| purpose | texto | P compra, C refinanciación con retiro, N sin retiro |
| orig_term | entero | Plazo original (meses) |
| num_borrowers | entero | Número de deudores |
| seller | texto | Originador |
| super_conforming | texto | Crédito de monto alto: Y |
| pre_harp_id, harp | texto | Programa HARP |
| special_eligibility | texto | Programa de elegibilidad especial |
| defecto | booleano | Salió por código 96 (defecto de suscripción o administración); se excluye del modelado |

## data/limpios/desempeno

Columnas del archivo de desempeño con sus tipos. Se descartan las posteriores al cierre (recuperaciones, gastos, pérdida y fechas de liquidación), que no pueden ser variables de un modelo y cuyo análisis está fuera del alcance, `ddlpi`, las constantes `step_mod_flag` y `cramdown_costs`, y `loan_age`, que se reinicia al modificar el crédito.

| Columna | Tipo | Descripción |
|---|---|---|
| loan_id | texto | Identificador del crédito |
| period | entero | Mes del registro, AAAAMM |
| current_upb | decimal | Saldo (USD) |
| dlq_status | texto | Meses de atraso (00, 01, ...) o RA (adjudicación del inmueble) |
| remaining_months | entero | Meses restantes |
| mod_flag | texto | Y modificado en el mes, P modificado antes |
| zero_balance_code | entero | Motivo de salida: 1 prepago, 2 remate, 3 venta corta, 9 venta del inmueble adjudicado, 15 venta del crédito, 16 titularización, 96 defecto |
| current_rate | decimal | Tasa actual (%) |
| non_interest_upb | decimal | Saldo que no genera intereses |
| cum_mod_cost, current_mod_cost | decimal | Costo de la modificación |
| payment_deferral | texto | Diferimiento de pagos: P, C |
| eltv | entero | Préstamo-valor estimado actual (%) |
| dlq_disaster | texto | Mora por desastre: Y |
| assistance_plan | texto | F indulgencia, R plan de repago, T periodo de prueba |
| interest_bearing_upb | decimal | Saldo que genera intereses |
| mi_cancel | texto | Cancelación del seguro: Y, N |
| servicer | texto | Administrador |
| mes | entero | Índice del mes (año * 12 + mes), para restar fechas |
| edad | entero | Meses desde el primer pago; 1 es el mes del primer pago y los meses previos quedan en cero o negativo |
| dlq_num | entero | Meses de atraso; nulo cuando es RA |
| es_ra | booleano | En adjudicación del inmueble |
| modificado | booleano | El crédito ya fue modificado (Y o P) |
| en_pandemia | booleano | Mes entre 202003 y 202112 |
| incumple | booleano | Evento de incumplimiento en el mes: 90 días o más, RA o salida por 2, 3 o 9 |

## data/modelado/p1

Una fila por crédito, sin los créditos con código 96. Solo variables disponibles al momento de la solicitud, que son también las que recibe la API.

| Columna | Descripción |
|---|---|
| loan_id | Identificador del crédito |
| first_payment_date | Mes del primer pago, para la partición temporal |
| credit_score, dti, oltv, ocltv, orig_rate, orig_upb, orig_term, mi_pct, purpose, occupancy, channel, first_time_homebuyer, num_borrowers, num_units, prop_type, state | Variables de la solicitud (ver originación) |
| y_p1 | 1 si el primer incumplimiento ocurre en los primeros 12 meses, 0 si no; nulo si el crédito sigue activo sin incumplir y aún no cumple 12 meses |

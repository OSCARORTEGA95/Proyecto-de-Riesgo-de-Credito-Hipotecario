# Datos del proyecto

Los datos de Freddie Mac no se pueden redistribuir, así que no están en Git. Se versionan con DVC en el bucket privado `s3://riesgodecreditohipotecario` (remoto `aws-remote`). En Git solo quedan los punteros (`*.dvc`, `dvc.lock`), las métricas del ETL y este archivo.

## Cómo descargar

Con las credenciales del Learner Lab configuradas, desde la raíz del repositorio:

```bash
dvc pull --glob "data/modelado/p1/*"            # tabla de la pregunta 1, todos los trimestres
dvc pull "data/modelado/p1/trimestre=2019Q1"    # un solo trimestre
dvc pull --glob "data/limpios/originacion/*"
dvc pull --glob "data/modelado/p2/*"            # tabla de la pregunta 2, pesada
dvc pull --glob "data/tablero/*/*"              # todas las tablas del tablero, livianas
dvc pull --glob "data/limpios/desempeno/*"      # la más pesada
```

Las tablas se leen con pandas: `pd.read_parquet("data/modelado/p1")` lee todos los trimestres descargados y agrega la columna `trimestre`.

P2 tiene millones de filas por trimestre y no cabe completa en la memoria de una VM de 8 GB. Para entrenar conviene leer solo una muestra, por ejemplo todos los positivos y los créditos con `grupo_muestra < 10`:

```python
import pyarrow.dataset as ds
d = ds.dataset("data/modelado/p2", partitioning="hive")
filtro = (ds.field("dlq_num") == 0) & ds.field("y_p2").is_valid() & ((ds.field("y_p2") == 1) | (ds.field("grupo_muestra") < 10))
p2 = d.to_table(filter=filtro).to_pandas()
```

## Cómo regenerar

`dvc repro` ejecuta el ETL (`src/riesgo_credito/procesamiento/etl.py`) para cada trimestre de `params.yaml` cuyo zip, código o parámetros hayan cambiado. Requiere Java 17 o superior y `pyspark==4.0.1`.

## Capas

```
data/crudos/historical_data_AAAATn.zip     zip trimestral de Freddie Mac (DVC)
data/limpios/originacion/trimestre=AAAATn  un crédito por fila
data/limpios/desempeno/trimestre=AAAATn    un crédito por mes
data/modelado/p1/trimestre=AAAATn          pregunta 1: variables de la solicitud y etiqueta
data/modelado/p2/trimestre=AAAATn          pregunta 2: variables mensuales y etiquetas
data/tablero/originacion/trimestre=AAAATn  conteos y sumas para Panorama y Evaluación
data/tablero/serie_mensual/trimestre=AAAATn  mora y asistencia por mes y estado
data/tablero/cartera_activa/trimestre=AAAATn  créditos al día en la fecha de corte, para Seguimiento
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

## data/modelado/p2

Una fila por crédito y mes en que el crédito está activo y al día o con 30 días de atraso, sin los créditos con código 96. Las variables usan solo información hasta ese mes. Las variables de la solicitud no se repiten: se unen desde P1 por `loan_id`.

La etiqueta de la carta es `y_p2` en las filas con `dlq_num = 0`. Las variantes son `y_p2` con `dlq_num` 0 o 1 (desde al día o 30 días) y `y_p2_60` con `dlq_num = 0` (llegada a 60 días o más).

| Columna | Descripción |
|---|---|
| loan_id, period | Crédito y mes, AAAAMM |
| edad | Meses desde el primer pago |
| dlq_num | Atraso en el mes: 0 al día, 1 treinta días |
| current_upb | Saldo (USD) |
| pct_saldo | Saldo / monto original |
| current_rate | Tasa actual (%) |
| dif_tasa | Tasa actual − tasa original |
| eltv | Préstamo-valor estimado actual (%) |
| atrasos_12m | Meses con atraso en los 12 meses anteriores |
| max_atraso_12m | Mayor atraso en los 12 meses anteriores |
| meses_desde_atraso | Meses desde el último atraso; nulo si nunca se ha atrasado |
| con_atraso_previo | Se ha atrasado alguna vez antes del mes |
| plan_asistencia | Plan en el mes: F indulgencia, R plan de repago, T periodo de prueba; nulo sin plan |
| indulgencia_12m | Meses en indulgencia en los 12 meses anteriores |
| diferimiento | Tiene diferimiento de pagos |
| saldo_sin_interes | Tiene saldo que no genera intereses |
| mora_desastre | Atraso atribuido a un desastre |
| modificado | El crédito ya fue modificado |
| en_pandemia | Mes entre 202003 y 202112 |
| y_p2 | 1 si llega a 90 días o más (o RA, o salida por 2, 3 o 9) en los 3 meses siguientes; nulo si esos meses no se alcanzan a observar |
| y_p2_60 | Igual, con 60 días o más |
| grupo_muestra | Número de 0 a 99 fijo por crédito, para tomar muestras reproducibles |

## data/tablero

Guardan conteos y sumas, no tasas ni promedios, para poder sumar cualquier combinación de filtros y dividir después. Por ejemplo, el puntaje promedio es `suma_puntaje / n_puntaje` y el incumplimiento a 12 meses por cohorte es `p1_positivos / p1_etiquetables` agrupando por `trimestre`.

**originacion**: un grupo por `state`, `purpose`, `rango_puntaje` (<620, 620-679, 680-719, 720-759, 760-799, >=800, Sin dato) y `rango_ltv` (<=60, 61-70, 71-80, 81-90, 91-95, >95). Columnas: `creditos`, `monto_total`, `suma_puntaje` y `n_puntaje`, `suma_tasa` y `n_tasa`, `suma_ltv` y `n_ltv`, `suma_dti` y `n_dti`, `p1_etiquetables`, `p1_positivos`. Incluye los créditos con código 96, porque cuentan como originaciones.

**serie_mensual**: un grupo por `period` y `state`, solo créditos activos. Columnas: `activos`, `saldo`, `al_dia`, `mora_30`, `mora_60`, `mora_90` (incluye RA), `saldo_mora_90`, `indulgencia`, `plan_repago`, `periodo_prueba`, `mora_desastre`, `diferimiento`.

**cartera_activa**: los créditos al día en la fecha de corte del trimestre, con las columnas de P2 (sin etiquetas), `first_payment_date` y las variables de la solicitud. Es lo que la vista Seguimiento envía a la API y lo que muestra en el detalle del crédito. Para otra fecha de corte se filtra P2 por `period`.

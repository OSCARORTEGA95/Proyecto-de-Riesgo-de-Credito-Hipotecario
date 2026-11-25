# Riesgo de crédito hipotecario con datos masivos

Prototipo funcional que estima la probabilidad de incumplimiento de cada solicitud de crédito y
alerta qué créditos activos llegarán a 90 días de mora en los próximos 3 meses. Usa el historial
mensual de créditos de Freddie Mac, procesado con Spark, y se despliega como un servicio de
inferencia y un tablero de control en contenedores Docker.

Proyecto del curso Despliegue de soluciones analíticas, Maestría en Inteligencia Analítica de
Datos, Universidad de Los Andes. Estado: **en construcción** (octubre a noviembre de 2026).

## Preguntas de negocio

1. **¿Qué clientes aprobar?** Al momento de la solicitud, ¿cuál es la probabilidad de que el
   crédito incumpla en sus primeros 12 meses?
2. **¿Qué clientes se van a atrasar?** Entre los créditos activos y al día, ¿cuáles llegarán a
   90 días de mora en los próximos 3 meses?

## Solución

| Componente | Herramienta | Qué hace |
|---|---|---|
| Procesamiento por lotes | PySpark | Lee los archivos crudos, limpia, une originación con desempeño y genera tablas en Parquet |
| Versionado | Git y Data Version Control | Código en Git; datos procesados y modelos en un almacenamiento remoto privado |
| Experimentos | MLflow | Registra parámetros, métricas y artefactos de las cuatro versiones de modelo |
| Modelos | Python | Versión 1, aprendizaje automático clásico; versión 2, árbol de decisión; versión 3, aprendizaje profundo; versión 4, red neuronal recurrente |
| Servicio de inferencia | Interfaz de programación de aplicaciones | Recibe una solicitud o un crédito y devuelve la probabilidad estimada |
| Tablero | Por definir | Evaluación de solicitudes, efecto del umbral, alerta temprana de mora y panorama de la cartera |
| Despliegue | Docker | Contenedores para el servicio, el tablero y MLflow |

Las versiones 1 a 3 responden la pregunta 1. La versión 4 responde la pregunta 2 con la secuencia
mensual de pagos de cada crédito.

## Datos

[Single-Family Loan-Level Dataset de Freddie Mac](https://freddiemac.com/research/datasets/sf-loanlevel-dataset),
créditos originados entre 2015 y 2020 con su desempeño hasta la fecha más reciente publicada.

| Archivo | Contenido |
|---|---|
| Originación | Una fila por crédito, 31 variables: puntaje crediticio, relación deuda-ingreso, relación préstamo-valor, monto, plazo, propósito, ocupación, estado y canal |
| Desempeño mensual | Una fila por crédito por mes, 35 variables: saldo, meses de atraso, edad del crédito, modificaciones, asistencia al deudor y código de cierre |

**Licencia:** los datos son gratuitos con registro, pero **no se pueden redistribuir**. Por eso
no se suben a Git ni a un almacenamiento público. Cada integrante los descarga con su propia
cuenta, y Data Version Control apunta a un almacenamiento remoto privado del equipo.

**Particularidades de los datos:**

- **Códigos especiales para "no disponible":** 9999 en el puntaje crediticio y 999 en la relación deuda-ingreso.
- **Texto en una columna numérica:** el estado de atraso del primer mes viene como "XX".
- **Vacíos con distintos significados:** según la variable, un vacío puede significar "no participa", "no aplica" o "no ha ocurrido".
- **Columnas constantes o vacías:** se eliminan en la canalización.

El diccionario completo está en [docs/datos](docs/datos).

## Estructura del repositorio

```
├── data/                     Se gestiona con Data Version Control (dvc.yaml, dvc.lock), no con Git
│   ├── crudos/               Archivos originales de Freddie Mac
│   ├── limpios/, desempeno/  Tablas en Parquet generadas por Spark
│   ├── modelado/p1, p2/      Tablas de las preguntas 1 y 2
│   └── tablero/              Tablas agregadas que lee el tablero
├── config/                   Parámetros de cada familia de modelos
├── src/riesgo_credito/
│   ├── procesamiento/        Canalización con PySpark y etiquetas
│   └── modelos/              k vecinos, logística, naive Bayes, árbol, red densa y red recurrente
├── modelos/                  Versiones exportadas y resumen de métricas
├── package-src/              Paquete del modelo desplegado (modelo_riesgo, .whl)
├── api/                      API de inferencia (FastAPI) y su Dockerfile
├── tablero/                  Tablero en Dash y su Dockerfile
├── docker-compose.yml        Levanta la API y el tablero
├── docs/manuales/            Manual de usuario, manual de instalación y guías de despliegue
├── notebooks/                Exploración, canalización y modelos
└── tests/                    Pruebas
```

## Cómo empezar

Requisitos: Docker con el complemento Compose, Git, Python 3.10 o superior y Data Version Control
con acceso al almacenamiento remoto del equipo (S3). El manual de instalación completo está en
[docs/manuales](docs/manuales).

```bash
git clone https://github.com/OSCARORTEGA95/Proyecto-de-Riesgo-de-Credito-Hipotecario.git
cd Proyecto-de-Riesgo-de-Credito-Hipotecario
python3 -m venv .venv && source .venv/bin/activate
pip install "dvc[s3]" tox build
dvc pull data/tablero/originacion data/tablero/serie_mensual data/modelado/p1 data/modelado/p2
cd package-src && tox run -e train && tox run -e p2 && python3 -m build && cd ..   # los dos modelos en un .whl
rm -f api/model-pkg/*.whl && cp package-src/dist/*.whl api/model-pkg/
sudo docker compose up -d --build api                  # API en el puerto 8001
python src/riesgo_credito/modelos/puntaje_p2.py        # alertas de la pregunta 2, puntuadas por la API
sudo docker compose up -d --build                      # tablero en el puerto 8060
```

La API sirve las dos preguntas: `POST /api/v1/predict` (probabilidad de incumplir en los primeros 12
meses, red neuronal) y `POST /api/v1/predict_p2` (probabilidad de que un crédito al día llegue a 90 días
de mora en 3 meses, k vecinos). La vista Evaluación del tablero usa la primera y Seguimiento, la segunda.

Si no tienes acceso al almacenamiento remoto, descarga los trimestres desde la página de Freddie
Mac con tu cuenta, guárdalos en `data/crudos/` y corre la canalización con `dvc repro`.

## Cómo trabajamos

- **Ramas:** `main` siempre estable; cada integrante trabaja en ramas propias y entra por
  solicitud de integración revisada por otro miembro.
- **Commits:** frecuentes y descriptivos, al menos dos por persona por semana; la nota es
  individual y se sostiene con los aportes visibles en el repositorio.
- **Partición de datos:** siempre temporal, nunca aleatoria, para evitar fuga de información.

## Equipo

| Integrante | Rol |
|---|---|
| Oscar David Ortega Villa | Líder: pregunta de negocio, modelos, MLflow y servicio de inferencia |
| Ana Isabel Fernández Duque | Analista de datos: exploración, maqueta, tablero y manual de usuario |
| Juan Sebastián Valdés Fuentes | Ingeniero de datos: canalización con Spark, Data Version Control, contenedores y despliegue |

## Entregas

| Entrega | Fecha | Contenido |
|---|---|---|
| 1 | Domingo 25 de octubre de 2026 | Problema, pregunta de negocio, datos, repositorios, exploración y maqueta |
| 2 | Domingo 8 de noviembre de 2026 | Modelos, experimentos en MLflow y tablero |
| 3 | Martes 24 de noviembre de 2026 | Modelos desplegados, contenedores, manuales y video |

Video de la entrega final: https://youtu.be/30XSaNiUy90

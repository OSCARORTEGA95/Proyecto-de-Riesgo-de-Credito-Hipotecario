
# Mortgage Risk Analytics - Tablero Dash

## Descripción

Tablero interactivo para el análisis y monitoreo del riesgo de crédito hipotecario, desarrollado con Python, Dash y Plotly a partir de datos de Freddie Mac.

El tablero está organizado en tres módulos:

- **Panorama:** indicadores de originación, distribuciones por FICO, LTV y propósito, distribución geográfica y evolución histórica del incumplimiento.
- **Evaluación:** formulario de solicitudes y comparación descriptiva con los promedios de originaciones de 2026 Q1.
- **Seguimiento:** indicadores de cartera activa, saldos, mora observada y evolución histórica.

Las funcionalidades predictivas de Evaluación y Seguimiento están pendientes de integración con los modelos y la API del equipo. No se generan probabilidades simuladas.

## Estructura

- `app.py`: aplicación principal y navegación.
- `panorama.py`: indicadores y gráficas de originación.
- `evaluacion.py`: formulario y comparación de solicitudes.
- `seguimiento.py`: monitoreo histórico de cartera.
- `requirements.txt`: dependencias de Python para el tablero.

## Instalación

Ejecutar los comandos desde la carpeta principal del repositorio.

Crear y activar un entorno virtual en Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```powershell
python -m pip install -r tablero/requirements.txt
```

Para descargar los datos con DVC, instalar adicionalmente:

```powershell
python -m pip install "dvc[s3]"
```

## Datos

Los datos son preparados por el proceso ETL del equipo y administrados mediante DVC con almacenamiento remoto en Amazon S3.

El tablero utiliza:

- `data/tablero/originacion`: información agregada de originación.
- `data/tablero/serie_mensual`: información agregada de seguimiento mensual.

Las particiones utilizadas corresponden a los trimestres de 2019 Q1 a 2026 Q1.

Para descargar solamente los datos requeridos por el tablero, ejecutar en PowerShell:

```powershell
$rutas = foreach ($anio in 2019..2026) {
    foreach ($q in 1..4) {
        if ($anio -lt 2026 -or $q -eq 1) {
            "data/tablero/originacion/trimestre=${anio}Q$q"
            "data/tablero/serie_mensual/trimestre=${anio}Q$q"
        }
    }
}

dvc pull $rutas
```

Se requiere acceso autorizado al bucket S3 y credenciales de AWS configuradas localmente.

No se deben guardar credenciales ni archivos de datos privados en GitHub.

## Ejecución

Desde la carpeta principal del repositorio:

```powershell
python tablero/app.py
```

Abrir en el navegador:

http://127.0.0.1:8050/

## Consideraciones

- Los promedios de Panorama se calculan utilizando las sumas y los conteos de los datos agregados.
- Evaluación compara una solicitud con los promedios reales de originaciones de 2026 Q1.
- Seguimiento utiliza datos históricos observados y permite filtrar por estado y mes.
- Las probabilidades predictivas y las alertas tempranas requieren integrar los modelos desarrollados por el equipo.
- Para el despliegue en Docker se deberán adaptar las rutas de datos y la configuración de ejecución al entorno del contenedor.

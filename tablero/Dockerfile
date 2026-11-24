FROM python:3.12-slim

# Crear usuario que ejecuta el tablero
RUN adduser --disabled-password --gecos '' dash-user

# Instalar compiladores C/C++ y build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Definir directorio de trabajo
WORKDIR /opt/riesgo

# Instalar dependencias (gunicorn sirve el tablero en el contenedor)
ADD ./tablero/requirements.txt /opt/riesgo/tablero/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /opt/riesgo/tablero/requirements.txt "gunicorn>=26.0.0,<27.0.0"

# Copiar el tablero y las tablas que lee: originacion y serie_mensual (descargadas con DVC)
# y alertas_p2 (lista puntuada por la API con src/riesgo_credito/modelos/puntaje_p2.py)
ADD ./tablero /opt/riesgo/tablero/
ADD ./data/tablero /opt/riesgo/data/tablero/

# Hacer ejecutable el script de arranque
RUN chmod +x /opt/riesgo/tablero/run.sh
# Cambiar propiedad de la carpeta a dash-user
RUN chown -R dash-user:dash-user ./

USER dash-user
WORKDIR /opt/riesgo/tablero
# Puerto a exponer para el tablero
EXPOSE 8060

# Comando a ejecutar al correr el contenedor
CMD ["bash", "./run.sh"]

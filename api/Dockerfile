FROM python:3.12-slim

# Crear usuario que ejecuta la API
RUN adduser --disabled-password --gecos '' api-user

# Instalar compiladores C/C++ y build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Definir directorio de trabajo
WORKDIR /opt/api

# Instalar dependencias, incluido el paquete del modelo (model-pkg/*.whl)
ADD ./requirements.txt /opt/api/requirements.txt
ADD ./model-pkg /opt/api/model-pkg/
RUN pip install --upgrade pip
RUN pip install -r /opt/api/requirements.txt

# Copiar el código de la API
ADD ./app /opt/api/app/
ADD ./run.sh /opt/api/run.sh

# Hacer ejecutable el script de arranque
RUN chmod +x /opt/api/run.sh
# Cambiar propiedad de la carpeta a api-user
RUN chown -R api-user:api-user ./

USER api-user
# Puerto a exponer para la API
EXPOSE 8001

# Comando a ejecutar al correr el contenedor
CMD ["bash", "./run.sh"]

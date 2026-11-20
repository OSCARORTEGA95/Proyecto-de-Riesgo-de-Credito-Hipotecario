# Redes recurrentes y redespliegue con Docker

Todo corre en el EC2 del proyecto (t3.medium, IP elástica 107.22.35.29). MLflow sigue en el
puerto 8050; la API (8001) y el tablero (8060) pasan a correr en contenedores.

## 1. Traer el código

```bash
cd ~/Riesgo_de_credito_hipotecario
git fetch origin
git checkout modelo/red-recurrente
git pull
```

La rama `modelo/red-recurrente` parte de `despliegue/api-tablero`, así que trae la red densa,
el paquete, la API, el tablero y los Dockerfile de la contenerización.

## 2. Datos

```bash
source ~/env-dvc/bin/activate
df -h ~                                   # espacio libre en disco
dvc pull --glob "data/modelado/p1/*"
dvc pull --glob "data/modelado/p2/*"      # tabla mensual: pesa bastante más que la de p1
```

Si el disco no alcanza para toda la tabla de la pregunta 2, basta con los trimestres de
originación hasta 2024Q4 (`dvc pull data/modelado/p2/trimestre=2024Q4`, etc.).

## 3. Entrenar las redes recurrentes

Con la API y el tablero detenidos, para dejarle la memoria al entrenamiento:

```bash
export MLFLOW_TRACKING_URI=http://localhost:8050
nohup env PYTHONPATH=src python -m riesgo_credito.modelos.rnn --pregunta p2 > rnn_p2.log 2>&1 &
tail -f rnn_p2.log
```

Cuando termine (`Terminó en X minutos`), la pregunta 1:

```bash
nohup env PYTHONPATH=src python -m riesgo_credito.modelos.rnn --pregunta p1 --sin-referencia > rnn_p1.log 2>&1 &
```

En MLflow quedan los experimentos `rnn-p2` (referencia y 4 versiones) y `rnn-p1` (2 versiones),
con las curvas por época de cada ejecución.

| Pregunta | Secuencia que lee la red | Se compara con |
|---|---|---|
| 2 | Los últimos 12 meses del crédito: edad, saldo, tasa, LTV, atrasos, días de mora, banderas | k vecinos (`knn-p2`) |
| 1 | Las columnas de la solicitud, una por paso (la solicitud no tiene historia) | red densa (`red-p1`) |

## 4. Comparar y elegir

```bash
PYTHONPATH=src python -m riesgo_credito.modelos.rnn --accion comparar
```

Imprime, por pregunta, todas las versiones ordenadas por AUC de validación y el ganador, y
guarda la tabla en `modelos/comparacion_rnn.csv`. El ganador se elige por validación, no por
prueba, igual que en las demás comparaciones del proyecto.

- **Pregunta 1:** si gana la red densa, la API sigue sirviendo el paquete actual. Si gana una
  red recurrente, hay que llevarla al paquete (`package-src`) antes de reconstruir la imagen.
- **Pregunta 2:** hoy la API y el tablero solo sirven la pregunta 1. Servir la pregunta 2 pide
  una ruta nueva en la API que reciba los 12 meses del crédito.

## 5. Desplegar la API y el tablero en contenedores

Si la API o el tablero todavía corren fuera de Docker, detenerlos:

```bash
pkill -f "app/main.py"; pkill -f "tablero/app.py"; pkill -f "tox run -e run"
```

Paquete del modelo (si `api/model-pkg/` ya tiene el `.whl` vigente, se puede saltar):

```bash
cd package-src && tox run -e train && tox run -e test_package && python3 -m build && cd ..
cp package-src/dist/modelo_riesgo_hipotecario-0.0.1-py3-none-any.whl api/model-pkg/
```

Imágenes y contenedores, en una red de Docker para que el tablero encuentre la API por nombre:

```bash
dvc pull --glob "data/tablero/*/*"
sudo docker build -t riesgo-api api/
sudo docker build -t riesgo-tablero -f tablero/Dockerfile .
sudo docker network create riesgo
sudo docker run -d --name api --network riesgo -p 8001:8001 -e PORT=8001 --restart unless-stopped riesgo-api
sudo docker run -d --name tablero --network riesgo -p 8060:8060 -e PORT=8060 \
     -e API_P1_URL=http://api:8001/api/v1/predict --restart unless-stopped riesgo-tablero
sudo docker ps
```

`--restart unless-stopped` vuelve a levantar los dos contenedores cuando la instancia se
reinicia. Prueba: `http://107.22.35.29:8001/docs` y `http://107.22.35.29:8060`, vista Evaluación.

Para ver los registros: `sudo docker logs -f api` o `sudo docker logs -f tablero`.
Para actualizar después de un cambio: `sudo docker rm -f api tablero`, reconstruir y volver a correr.

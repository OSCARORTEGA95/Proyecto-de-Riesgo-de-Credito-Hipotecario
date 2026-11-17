# Despliegue: red neuronal, paquete, API y tablero en el EC2

Todo corre en la misma máquina EC2 (t3.medium, Ubuntu), cada servicio en su puerto:

| Servicio | Puerto | Cómo se levanta |
|---|---|---|
| MLflow | 8050 | ya está corriendo |
| API (FastAPI + uvicorn) | 8001 | `tox run -e run` en `api/` |
| Tablero (Dash) | 8060 | `python tablero/app.py` con variables de entorno |

Abre los puertos 8001 y 8060 en el grupo de seguridad: EC2 → Security Groups → grupo de la
instancia → Edit inbound rules → Add rule → Custom TCP, puerto, Anywhere IPv4 → Save rules.

```
data/modelado/p1 (DVC) ──► package-src: tox train ──► .whl ──► api/model-pkg ──► API :8001 ◄── tablero :8060
                                  │
                                  └──► MLflow :8050 (experimento red-p1)
```

## 0. Preparar la máquina (una sola vez)

```bash
sudo apt update
sudo apt install -y python3-pip python3.12-venv zip unzip
python3 -m venv ~/env-despliegue
source ~/env-despliegue/bin/activate
pip install tox build "dvc[s3]"
git clone https://github.com/OSCARORTEGA95/Proyecto-de-Riesgo-de-Credito-Hipotecario.git
cd Proyecto-de-Riesgo-de-Credito-Hipotecario
git checkout despliegue/api-tablero
```

Configura las credenciales del Learner Lab (AWS Details → AWS CLI) en `~/.aws/credentials`
y descarga la tabla de la pregunta 1:

```bash
dvc pull --glob "data/modelado/p1/*"
```

## 1. Comparar las 8 redes en MLflow

```bash
pip install -r tablero/requirements.txt pyyaml   # mlflow, scikit-learn, pandas, pyarrow, dash
export MLFLOW_TRACKING_URI=http://localhost:8050
PYTHONPATH=src python -m riesgo_credito.modelos.red
```

En `http://<IP>:8050`, experimento `red-p1`: selecciona las ejecuciones y usa Compare.
Cada ejecución tiene las curvas por época en la pestaña de métricas.

## 2. Empaquetar el modelo (Taller 5)

Copia en `package-src/modelo_riesgo/config.yml` las capas y el dropout de la mejor red.

```bash
cd package-src
export MLFLOW_TRACKING_URI=http://localhost:8050   # registra también el modelo del paquete
tox run -e train          # entrena y guarda modelo_riesgo/trained_models/modelo_red_p1_v0.0.1.pkl
tox run -e test_package   # pruebas con el modelo entrenado
python3 -m build          # crea dist/*.tar.gz y dist/*.whl
cd ../package-test
cp ../package-src/dist/modelo_riesgo_hipotecario-0.0.1-py3-none-any.whl .
pip install modelo_riesgo_hipotecario-0.0.1-py3-none-any.whl
pip freeze | grep modelo
python3 test-package.py
```

Nueva versión (Taller 5, parte 4): quita una variable de `variables_numericas` o
`variables_categoricas` en `config.yml`, cambia `modelo_riesgo/VERSION` a `0.0.2` y repite.

## 3. Levantar la API (Taller 6)

```bash
cd ../api
cp ../package-src/dist/modelo_riesgo_hipotecario-0.0.1-py3-none-any.whl model-pkg/
tox run -e test_app       # instala la API con el modelo y corre las pruebas
tox run -e run            # API en http://<IP>:8001 (documentación en /docs)
```

Para dejarla corriendo al cerrar la terminal: `nohup tox run -e run > api.log 2>&1 &`.

Con la versión 0.0.2 del paquete: copia el nuevo `.whl` a `model-pkg/`, cambia la última
línea de `requirements.txt` y usa `tox run -r -e test_app` (la `-r` recrea el ambiente).

En `/docs`, ruta `predict`, "Try it out": si borras del ejemplo una variable que el modelo
usa, la API responde 400 con el nombre de la variable; si borras una que no usa (o agregas
`channel`), responde normalmente.

## 4. Levantar el tablero conectado a la API

En otra terminal, desde la raíz del repositorio:

```bash
source ~/env-despliegue/bin/activate
dvc pull --glob "data/tablero/*/*"
export API_P1_URL=http://localhost:8001/api/v1/predict
TABLERO_HOST=0.0.0.0 TABLERO_PUERTO=8060 TABLERO_DEBUG=0 nohup python tablero/app.py > tablero.log 2>&1 &
```

Abre `http://<IP>:8060`, vista Evaluación, "Caso de ejemplo" y "Comparar perfil": la nota
bajo la probabilidad dice que el modelo es la red neuronal consultada en la API.
Para volver al modelo KNN de MLflow sin API: `PREDICCION_P1=mlflow`.

## Si algo falla

- `tox run -e train` no encuentra datos: falta `dvc pull --glob "data/modelado/p1/*"`.
- El tablero muestra "No disponible": la API no está corriendo o `API_P1_URL` está mal.
- La IP pública cambia cada vez que se reinicia el laboratorio: usa la nueva en el navegador.
- Para ver qué hay en cada puerto: `sudo ss -ltnp | grep -E '8001|8050|8060'`.

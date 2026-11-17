from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel

import modelo_riesgo

PACKAGE_ROOT = Path(modelo_riesgo.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent  # carpeta package-src
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class ConfigRed(BaseModel):
    capas: List[int]
    dropout: float
    epocas_max: int
    paciencia: int
    tam_lote: int
    tasa_aprendizaje: float
    l2: float


class Config(BaseModel):
    package_name: str
    pipeline_save_file: str
    ruta_datos: str
    etiqueta: str
    seleccion: List[str]
    validacion: List[str]
    entrenamiento: List[str]
    prueba: List[str]
    max_validacion: int
    proporcion_negativos: int
    semilla: int
    variables_numericas: List[str]
    variables_categoricas: List[str]
    red: ConfigRed
    experimento_mlflow: str

    @property
    def variables(self) -> List[str]:
        return self.variables_numericas + self.variables_categoricas


def cargar_config(ruta: Path = CONFIG_FILE_PATH) -> Config:
    with open(ruta, encoding="utf-8") as f:
        return Config(**yaml.safe_load(f))


config = cargar_config()

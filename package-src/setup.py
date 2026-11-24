#!/usr/bin/env python
from pathlib import Path

from setuptools import find_packages, setup

NAME = "modelo-riesgo-hipotecario"
DESCRIPTION = ("Riesgo de crédito hipotecario: red neuronal de la pregunta 1 (incumplir en los primeros 12 meses) "
               "y k vecinos de la pregunta 2 (llegar a 90 días de mora en los próximos 3 meses).")
URL = "https://github.com/OSCARORTEGA95/Proyecto-de-Riesgo-de-Credito-Hipotecario"
AUTHOR = "Equipo Riesgo de Crédito Hipotecario - MIAD"
REQUIRES_PYTHON = ">=3.10.0"

ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
PACKAGE_DIR = ROOT_DIR / "modelo_riesgo"

about = {"__version__": (PACKAGE_DIR / "VERSION").read_text().strip()}


def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return [linea.strip() for linea in fd.read().splitlines() if linea.strip() and not linea.startswith("#")]


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"modelo_riesgo": ["VERSION", "config.yml", "trained_models/*.pkl", "trained_models/*.joblib",
                                    "trained_models/*.json"]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
)

"""Modelo de riesgo de crédito hipotecario, pregunta 1: probabilidad de incumplir en los primeros 12 meses."""
import logging
from pathlib import Path

with open(Path(__file__).resolve().parent / "VERSION") as version_file:
    __version__ = version_file.read().strip()

logging.getLogger(__name__).addHandler(logging.NullHandler())

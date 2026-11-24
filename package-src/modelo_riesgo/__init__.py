"""Modelos de riesgo de crédito hipotecario.

Pregunta 1 (predict.py): probabilidad de incumplir en los primeros 12 meses, red neuronal.
Pregunta 2 (p2.py): probabilidad de que un crédito al día llegue a 90 días de mora en 3 meses, k vecinos.
"""
import logging
from pathlib import Path

with open(Path(__file__).resolve().parent / "VERSION") as version_file:
    __version__ = version_file.read().strip()

logging.getLogger(__name__).addHandler(logging.NullHandler())

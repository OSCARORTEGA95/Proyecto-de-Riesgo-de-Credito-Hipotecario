import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def sample_input_data():
    """Cinco solicitudes con los nueve campos del formulario (una sin puntaje, que el modelo imputa)."""
    return pd.DataFrame(json.loads((Path(__file__).parent / "datos_prueba.json").read_text(encoding="utf-8")))

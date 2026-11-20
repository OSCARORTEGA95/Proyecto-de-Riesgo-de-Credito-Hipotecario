from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("http://localhost:8001/api/v1/health")
    assert response.status_code == 200
    cuerpo = response.json()
    assert cuerpo["api_version"] and cuerpo["model_version"]


def test_make_prediction(client: TestClient, solicitudes: list) -> None:
    response = client.post("http://localhost:8001/api/v1/predict", json={"inputs": solicitudes})
    assert response.status_code == 200
    resultado = response.json()
    assert resultado["errors"] is None
    assert len(resultado["predictions"]) == 2
    assert all(0.0 <= p <= 1.0 for p in resultado["predictions"])


def test_falta_variable_usada(client: TestClient, solicitudes: list) -> None:
    sin_puntaje = [{k: v for k, v in s.items() if k != "credit_score"} for s in solicitudes]
    response = client.post("http://localhost:8001/api/v1/predict", json={"inputs": sin_puntaje})
    assert response.status_code == 400
    assert "credit_score" in response.json()["detail"]["faltan_variables"]


def test_variable_no_usada(client: TestClient, solicitudes: list) -> None:
    con_canal = [{**s, "channel": "R"} for s in solicitudes]
    response = client.post("http://localhost:8001/api/v1/predict", json={"inputs": con_canal})
    assert response.status_code == 200


def test_raiz_html(client: TestClient) -> None:
    response = client.get("http://localhost:8001/")
    assert response.status_code == 200 and "/docs" in response.text

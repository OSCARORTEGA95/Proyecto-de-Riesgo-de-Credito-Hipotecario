import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Any  # noqa: E402

from fastapi import APIRouter, FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from loguru import logger  # noqa: E402

from app.api import api_router  # noqa: E402
from app.config import settings, setup_app_logging  # noqa: E402

# Registros de la API
setup_app_logging(config=settings)

app = FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")

root_router = APIRouter()


# Ruta raíz: una página HTML con el enlace a la documentación interactiva
@root_router.get("/")
def index(request: Request) -> Any:
    """Respuesta básica en HTML."""
    body = (
        "<html>"
        "<body style='padding: 10px; font-family: Arial, sans-serif;'>"
        "<h1>API de riesgo de crédito hipotecario</h1>"
        "<p>Probabilidad de incumplir en los primeros 12 meses de un crédito, "
        "estimada con la red neuronal empaquetada.</p>"
        "<div>"
        "Consulte la documentación y pruebe la API: <a href='/docs'>aquí</a>"
        "</div>"
        "</body>"
        "</html>"
    )
    return HTMLResponse(content=body)


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

# Acceso CORS desde los orígenes de la configuración
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


if __name__ == "__main__":
    # Solo para desarrollo y pruebas. En producción se usaría un servidor detrás de un proxy.
    logger.warning("Running in development mode. Do not run like this in production.")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")

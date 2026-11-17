import logging
import sys
from types import FrameType
from typing import List, cast

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO  # los registros pueden ser INFO, DEBUG, WARNING, etc.


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"

    # Orígenes con acceso CORS. El tablero llama a la API desde su servidor de Python,
    # así que esto solo importa si un navegador consulta la API directamente.
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:8050",
        "http://localhost:8060",
        "http://localhost:8001",
        "https://localhost:3000",
    ]

    PROJECT_NAME: str = "API de riesgo de crédito hipotecario"

    logging: LoggingSettings = LoggingSettings()

    model_config = SettingsConfigDict(case_sensitive=True)


class InterceptHandler(logging.Handler):
    """Envía a loguru los registros del módulo logging (por ejemplo, los de uvicorn)."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_app_logging(config: Settings) -> None:
    """Prepara los registros de la API y de uvicorn."""
    LOGGERS = ("uvicorn.asgi", "uvicorn.access")
    logging.getLogger().handlers = [InterceptHandler()]
    for logger_name in LOGGERS:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(level=config.logging.LOGGING_LEVEL)]

    logger.configure(handlers=[{"sink": sys.stderr, "level": config.logging.LOGGING_LEVEL}])


settings = Settings()

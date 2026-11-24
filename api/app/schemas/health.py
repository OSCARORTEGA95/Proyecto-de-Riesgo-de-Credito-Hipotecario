from typing import Any, Optional

from pydantic import BaseModel


class Health(BaseModel):
    name: str
    api_version: str
    model_version: str
    model_p1: str
    model_p2: Optional[Any] = None

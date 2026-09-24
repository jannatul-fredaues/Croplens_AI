"""
Response models for the API. Keeping these separate from main.py means the
shape of a response can be reused, imported by tests, and documented cleanly
in the auto-generated /docs page.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model_version: str = Field(examples=["v1"])
    model_loaded: bool


class ClassScore(BaseModel):
    label: str = Field(examples=["onion"])
    confidence: float = Field(ge=0.0, le=1.0, examples=[0.94])


class CareInfo(BaseModel):
    """Static crop-care data. Fields are optional since not every class
    (or a not-yet-filled crop_info.json entry) will have all of them."""

    season: str | None = None
    water: str | None = None
    harvest: str | None = None
    npk: str | None = None
    source: str | None = None


class PredictResponse(BaseModel):
    label: str = Field(examples=["onion"])
    confidence: float = Field(ge=0.0, le=1.0, examples=[0.94])
    low_confidence: bool
    top_k: list[ClassScore]
    care: CareInfo | None = None
    model_version: str


class ErrorResponse(BaseModel):
    detail: str

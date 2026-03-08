"""Pydantic-схемы для REST API v2."""
from __future__ import annotations
from typing import Optional, Union
from pydantic import BaseModel, Field
from ..core.enums import Case, Number
from ..service import EntityType

class InflectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500,
                      examples=["красивый большой дом", "Иванов Иван Иванович"])
    target_case: Case = Field(..., examples=["gent", "datv"])
    entity_type: EntityType = Field(EntityType.AUTO)
    target_number: Optional[Number] = None
    gender: Optional[str] = Field(None, examples=["male", "female"])
    context: Optional[str] = Field(None, max_length=500)

class InflectResponse(BaseModel):
    original: str
    result: str
    confidence: float
    engine: str
    target_case: str
    target_case_readable: str
    details: list = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0

class BatchItem(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    target_case: str
    entity_type: str = "auto"
    target_number: Optional[str] = None
    gender: Optional[str] = None
    context: Optional[str] = None

class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(..., min_length=1, max_length=1000)

class BatchResponse(BaseModel):
    results: list[InflectResponse]
    total_elapsed_ms: float

class ParadigmRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=100)

class HealthResponse(BaseModel):
    status: str
    version: str
    engines: dict
    metrics: dict
    gpu_components: Union[dict, str] = "none"
    phrase_cache_size: int

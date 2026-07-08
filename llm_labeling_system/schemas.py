from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class LabelIn(BaseModel):
    seq: int
    label: str
    confidence: float = 1.0
    use_for_training: Optional[bool] = None
    notes: str = ""


class SuggestIn(BaseModel):
    seq: int
    mock: bool = False
    model: Optional[str] = None


class ScanIn(BaseModel):
    mock: bool = False
    model: Optional[str] = None


class AISettingsIn(BaseModel):
    # None = leave unchanged; "" = clear the stored value (fall back to env/default).
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None


class AITestIn(BaseModel):
    # None = use the stored/env value.
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None


class ProgressOut(BaseModel):
    total: int
    labeled: int
    remaining: int


class SessionOut(BaseModel):
    id: int
    project: str
    source_name: str
    row_count: int
    window_size: int
    stride: int
    window_count: int
    created_at: int
    progress: ProgressOut

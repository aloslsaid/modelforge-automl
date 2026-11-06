from pydantic import BaseModel, Field
from typing import Literal

class PrepareRequest(BaseModel):
    target_column: str = Field(min_length=1, max_length=200)
    model_type: Literal["REGRESSION", "CLASSIFICATION"]

class PrepareResponse(BaseModel):
    status: str
    original_filename: str
    cleaned_filename: str
    target_column: str
    model_type: str
    rows: int
    columns: int
    ai_provider: str
    cleaning_summary: list[str]
    download_url: str

from pathlib import Path
from enum import Enum
from pydantic import BaseModel


class FaceLivenessInput(BaseModel):
    camera_image_b64: str


class FaceLivenessStatus(str, Enum):
    true = 'True'
    false = 'False'


class FaceLivenessOutput(BaseModel):
    status: FaceLivenessStatus

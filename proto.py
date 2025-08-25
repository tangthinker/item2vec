from typing import List
from pydantic import BaseModel

class TrainReq(BaseModel):
    data: List[str]

class GenVecReq(BaseModel):
    item: str

class GenVecResp(BaseModel):
    vec: List[float]
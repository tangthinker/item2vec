from typing import List
from pydantic import BaseModel

class TrainReq(BaseModel):
    data: List[List[str]]

class GenVecReq(BaseModel):
    item: str

class GenVecResp(BaseModel):
    vec: List[float]

class SimilarityReq(BaseModel):
    item1: str
    item2: str

class SimilarityResp(BaseModel):
    similarity: float
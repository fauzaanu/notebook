from typing import List

from pydantic import BaseModel


class RetrievedDocs(BaseModel):
    relevant_texts: List[str]


class FinalResponse(BaseModel):
    answer: str

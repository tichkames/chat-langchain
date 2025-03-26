from typing import Literal, Any
from pydantic import BaseModel

class Event(BaseModel):
    """Generic event structure."""

    event: str = "data"
    data: dict


class EndEvent(BaseModel):
    """Event representing the end of a stream."""

    event: Literal["end"] = "end"

def default_serialization(obj: Any) -> Any:
    """
    Default serialization for LangChain objects.
    Converts BaseModel instances to dictionaries.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
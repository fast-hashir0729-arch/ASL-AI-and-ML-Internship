"""Request and response shapes (the API contract)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import MAX_BATCH_SIZE

EXAMPLE_PASSENGER = {
    "pclass": 1,
    "sex": "female",
    "age": 29,
    "sibsp": 0,
    "parch": 0,
    "fare": 211.34,
    "embarked": "S",
}


class Passenger(BaseModel):
    """One passenger. strict=True means "3" (a string) is rejected for a number."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={"example": EXAMPLE_PASSENGER},
    )

    pclass: int = Field(ge=1, le=3, description="Ticket class: 1, 2 or 3")
    sex: Literal["male", "female"]
    age: float = Field(ge=0, le=100, description="Age in years")
    sibsp: int = Field(ge=0, le=10, description="Siblings/spouses aboard")
    parch: int = Field(ge=0, le=10, description="Parents/children aboard")
    fare: float = Field(ge=0, le=600, description="Ticket price")
    embarked: Literal["S", "C", "Q"] = Field(description="Port of embarkation")


class BatchRequest(BaseModel):
    """A list of passengers to predict in one call."""

    model_config = ConfigDict(extra="forbid")

    records: list[Passenger] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    """The model output for one passenger."""

    prediction: int = Field(description="1 = survived, 0 = did not survive")
    label: str
    probability_survived: float
    confidence: float = Field(description="Probability of the predicted class")


class PredictionResponse(Prediction):
    """Single prediction plus model information."""

    model_name: str
    model_version: str


class BatchResponse(BaseModel):
    """Many predictions plus model information."""

    model_name: str
    model_version: str
    count: int
    predictions: list[Prediction]

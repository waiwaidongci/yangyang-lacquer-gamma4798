from pydantic import BaseModel, ConfigDict, Field


class CalibrationRuleIn(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    meter: str = Field(min_length=1, max_length=64)
    starts: str = Field(min_length=10, max_length=10)
    ends: str = Field(min_length=10, max_length=10)
    lower: float = Field(ge=-30, le=30, allow_inf_nan=False)
    upper: float = Field(ge=-30, le=30, allow_inf_nan=False)


class CalibrationUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    rules: list[CalibrationRuleIn] = Field(max_length=200)


class CalibrationView(BaseModel):
    version: int
    rules: list[CalibrationRuleIn]

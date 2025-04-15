import operator
from pydantic import BaseModel, Field
from typing import List, Tuple, Union
from typing_extensions import TypedDict, Annotated

class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    ui_steps: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

class Plan(BaseModel):
    steps: List[str] = Field(description="different steps to follow, should be in sorted order")
    ui_steps: List[str] = Field(description="steps to show in UI, should be in sorted order and no more than 5 words")

class Response(BaseModel):
    response: str = Field(description="Final response should be precise and too the point.")
 
class Act(BaseModel):
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )
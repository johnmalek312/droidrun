from typing import List
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Event
from llama_index.core.tools import FunctionTool


class InputEvent(Event):
    input: list[ChatMessage]

class ExecutePlan(Event):
    plan: List[str] 

class TaskControllerEvent(Event):
    task: str ## change placeholder


class TaskCompletedEvent(Event):
    task_id: str

class TaskFailedEvent(Event):
    task_description: str
    reason: str

class RunTaskEvent(Event):
    goal: str
    tools: list[FunctionTool] = []

class UpdateFailTaskEvent(Event):
    




class TaskResultEvent(Event):
    task_id: str
    result: bool
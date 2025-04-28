from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event
from typing import Optional


class InputEvent(Event):
    input: list[ChatMessage]

class ModelOutputEvent(Event):
    thoughts: Optional[str] = None
    code: Optional[str] = None  

class ExecutionEvent(Event):
    code: str
    globals: dict[str, str] = {}
    locals: dict[str, str] = {}

class ExecutionResultEvent(Event):
    output: str
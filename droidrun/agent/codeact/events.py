from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event


class InputEvent(Event):
    input: list[ChatMessage]

class ModelOutputEvent(Event):
    thoughts: str = ""
    code: str = ""

class ExecutionEvent(Event):
    code: str
    globals: dict[str, str] = {}
    locals: dict[str, str] = {}

class ExecutionResultEvent(Event):
    output: str
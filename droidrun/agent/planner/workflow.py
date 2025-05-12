from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from .events import *
import base64
import json
import logging
import re
from enum import Enum
import time
from typing import Awaitable, Callable, List, Optional, Dict, Any, Tuple, TYPE_CHECKING, Union
import inspect
# LlamaIndex imports for LLM interaction and types
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ImageBlock, TextBlock
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Context, step
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms.llm import LLM
from ..utils.executer import SimpleCodeExecutor
from ..utils.chat_utils import add_ui_text_block, add_screenshot_image_block, add_screenshot, message_copy
from .task_manager import TaskManager

from llama_index.core import set_global_handler
set_global_handler("arize_phoenix")
# Load environment variables (for API key)
from dotenv import load_dotenv
load_dotenv()

if TYPE_CHECKING:
    from ...tools import Tools

logger = logging.getLogger("droidrun")
logging.basicConfig(level=logging.INFO)

DEFAULT_PLANNER_SYSTEM_PROMPT = """You are an Android Task Planner. Your job is to create short, functional plans (1-5 steps) to achieve a user's goal on an Android device.

**Inputs You Receive:**
1.  **User's Overall Goal.**
2.  **Current Device State:**
    *   A **screenshot** of the current screen.
    *   **JSON data** of visible UI elements.

**Your Task:**
Given the goal and current state, devise the **next 1-5 functional steps**. Focus on what to achieve, not how. Planning fewer steps at a time improves accuracy, as the state can change.

**Step Format:**
Each step must be a functional goal. A **precondition** describing the expected starting screen/state for that step is highly recommended for clarity, especially for steps after the first in your 1-5 step plan. Each task string can start with "Precondition: ... Goal: ...". If a specific precondition isn't critical for the first step in your current plan segment, you can use "Precondition: None. Goal: ..." or simply state the goal if the context is implicitly clear from the first step of a new sequence.

**Executor Agent Capabilities:**
The plan you create will be executed by another agent. This executor can:
*   `swipe(direction: str, distance_percentage: int)`
*   `input_text(text: str, element_hint: Optional[str] = None)`
*   `press_key(keycode: int)` (Common: 3=HOME, 4=BACK)
*   `tap_by_coordinates(x: int, y: int)` (This is a fallback; prefer functional goals)
*   `start_app(package_name: str)`
*   `list_packages()`
*   (The executor will use the UI JSON to find elements for your functional goals like "Tap 'Settings button'" or "Enter text into 'Username field'").

**Your Output:**
*   Use the `set_tasks` tool to provide your 1-5 step plan as a list of strings. **If this is the *initial* set of tasks you are providing for the current overall goal, you MUST also call `start_agent()` immediately after your `set_tasks` call to begin execution.**
*   **After your planned steps are executed (triggered by `start_agent()` or subsequent iterations), you will be invoked again with the new device state.** You will then:
    1.  Assess if the **overall user goal** is complete.
    2.  If complete, call the `complete_goal(message: str)` tool.
    3.  If not complete, generate the next 1-5 steps using `set_tasks` (without calling `start_agent()` again).

**Key Rules:**
*   **Functional Goals ONLY:** (e.g., "Navigate to Wi-Fi settings", "Enter 'MyPassword' into the password field").
*   **NO Low-Level Actions:** Do NOT specify swipes, taps on coordinates, or element IDs in your plan.
*   **Short Plans (1-5 steps):** Plan only the immediate next actions.
*   **Use Tools:** Your response *must* be a Python code block calling `set_tasks` (potentially followed by `start_agent()` on the first call) or `complete_goal`.

**Available Planning Tools:**
{tools_description}
*   `set_tasks(tasks: List[str])`: Defines the sequence of tasks. Each element in the list is a string representing a single task.
*   `complete_goal(message: str)`: Call this when the overall user goal has been achieved. The message can summarize the completion.
*   `start_agent()`: Call this **only once**, immediately after the *first* `set_tasks` call for an overall goal, to initiate the execution of the plan.

---

**Example Interaction Flow:**

**User Goal:** Turn on Wi-Fi.

**(Round 1) Planner Input:**
*   Goal: "Turn on Wi-Fi"
*   Current State: Screenshot of Home screen, UI JSON.

**Planner Thought Process (Round 1):**
Need to open settings first, then go to Network settings. This is the first plan.
1. Task 1: "Precondition: None. Goal: Open the Settings app."
2. Task 2: "Precondition: Settings main screen is open. Goal: Navigate to 'Network & internet' settings."

**Planner Output (Round 1):**
```python
set_tasks(tasks=[
    "Precondition: None. Goal: Open the Settings app.",
    "Precondition: Settings main screen is open. Goal: Navigate to 'Network & internet' settings."
])
start_agent() # Called because this is the first set of tasks for this goal
```

**(After Executor (triggered by `start_agent()`) performs these steps...)**

**(Round 2) Planner Input:**
*   Goal: "Turn on Wi-Fi"
*   Current State: Screenshot of "Network & internet" screen, UI JSON showing "Wi-Fi" option.

**Planner Thought Process (Round 2):**
Now on "Network & internet". Need to tap Wi-Fi, then enable it. This is a subsequent plan.
1. Task 1: "Precondition: 'Network & internet' screen is open. Goal: Tap the 'Wi-Fi' option."
2. Task 2: "Precondition: Wi-Fi settings screen is open. Goal: Enable the Wi-Fi toggle if it's off."

**Planner Output (Round 2):**
```python
set_tasks(tasks=[
    "Precondition: 'Network & internet' screen is open. Goal: Tap the 'Wi-Fi' option.",
    "Precondition: Wi-Fi settings screen is open. Goal: Enable the Wi-Fi toggle if it's off."
])
# No start_agent() call here
```

**(After Executor performs these steps...)**

**(Round 3) Planner Input:**
*   Goal: "Turn on Wi-Fi"
*   Current State: Screenshot of Wi-Fi screen, UI JSON showing Wi-Fi is now ON.

**Planner Thought Process (Round 3):**
Wi-Fi is on. Goal achieved.

**Planner Output (Round 3):**
```python
complete_goal(message="Wi-Fi has been successfully enabled.")
```"""
DEFAULT_PLANNER_USER_PROMPT = """Goal: {goal}"""

DEFAULT_PLANNER_TASK_FAILED_PROMPT = """
PLANNING UPDATE: The execution of a task failed.

Failed Task Description: "{task_description}"
Reported Reason: {reason}

The previous plan has been stopped. I have attached a screenshot representing the device's **current state** immediately after the failure. Please analyze this visual information.

Original Goal: {goal}

Instruction: Based **only** on the provided screenshot showing the current state and the reason for the previous failure ('{reason}'), generate a NEW plan starting from this observed state to achieve the original goal: '{goal}'.
"""

class PlannerAgent(Workflow):
    def __init__(self, goal: str, llm: LLM, agent: Workflow, tools_instance: 'Tools', executer = None, system_prompt = None, user_prompt = None, max_retries = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.goal = goal
        self.task_manager = TaskManager()
        self.tools = [self.task_manager.set_tasks, self.task_manager.add_task, self.task_manager.get_all_tasks, self.task_manager.clear_tasks, self.task_manager.complete_goal, self.task_manager.start_agent]
        self.tools_description = self.parse_tool_descriptions()
        if not executer:
            self.executer = SimpleCodeExecutor(loop=None, globals={}, locals={}, tools=self.tools, use_same_scope=True)
        else:
            self.executer = executer
        self.system_prompt = system_prompt or DEFAULT_PLANNER_SYSTEM_PROMPT.format(tools_description=self.tools_description)
        self.user_prompt = user_prompt or DEFAULT_PLANNER_USER_PROMPT.format(goal=goal)
        self.system_message = ChatMessage(role="system", content=self.system_prompt)
        self.user_message = ChatMessage(role="user", content=self.user_prompt)
        self.memory = None
        self.agent = agent
        self.tools_instance = tools_instance

        self.max_retries = max_retries # Number of retries for a failed task

        self.current_retry = 0 # Current retry count

        # TODO: Implement self.steps_counter and self.code_exec_counter and self.max_steps
        self.steps_counter = 0 # Steps counter
        self.code_exec_counter = 0 # Code execution counter
        self.max_steps = 10000

    def _extract_code_and_thought(self, response_text: str) -> Tuple[Optional[str], str]:
        """
        Extracts code from Markdown blocks (```python ... ```) and the surrounding text (thought),
        handling indented code blocks.

        Returns:
            Tuple[Optional[code_string], thought_string]
        """
        logger.debug("✂️ Extracting code and thought from response...")
        code_pattern = r"^\s*```python\s*\n(.*?)\n^\s*```\s*?$" # Added ^\s*, re.MULTILINE, and made closing fence match more robust
        # Use re.DOTALL to make '.' match newlines and re.MULTILINE to make '^' match start of lines
        code_matches = list(re.finditer(code_pattern, response_text, re.DOTALL | re.MULTILINE))

        if not code_matches:
            # No code found, the entire response is thought
            logger.debug("  - No code block found. Entire response is thought.")
            return None, response_text.strip()

        extracted_code_parts = []
        for match in code_matches:
             # group(1) is the (.*?) part - the actual code content
             code_content = match.group(1)
             extracted_code_parts.append(code_content) # Keep original indentation for now
             logger.debug(f"  - Matched code block:\n---\n{code_content}\n---")

        extracted_code = "\n\n".join(extracted_code_parts)
        logger.debug(f"  - Combined extracted code:\n```python\n{extracted_code}\n```")


        # Extract thought text (text before the first code block, between blocks, and after the last)
        thought_parts = []
        last_end = 0
        for match in code_matches:
            # Use span(0) to get the start/end of the *entire* match (including fences and indentation)
            start, end = match.span(0)
            thought_parts.append(response_text[last_end:start])
            last_end = end
        thought_parts.append(response_text[last_end:]) # Text after the last block

        thought_text = "".join(thought_parts).strip()
        # Avoid overly long debug messages for thought
        thought_preview = (thought_text[:100] + '...') if len(thought_text) > 100 else thought_text
        logger.debug(f"  - Extracted thought: {thought_preview}")

        return extracted_code, thought_text
    
    def parse_tool_descriptions(self) -> str:
        """Parses the available tools and their descriptions for the system prompt."""
        logger.info("🛠️ Parsing tool descriptions for Planner Agent...")
        # self.available_tools is a list of functions, we need to get their docstrings, names, and signatures and display them as `def name(args) -> return_type:\n"""docstring"""    ...\n`
        tool_descriptions = []
        for tool in self.tools:
            assert callable(tool), f"Tool {tool} is not callable."
            tool_name = tool.__name__
            tool_signature = inspect.signature(tool)
            tool_docstring = tool.__doc__ or "No description available."
            # Format the function signature and docstring
            formatted_signature = f"def {tool_name}{tool_signature}:\n    \"\"\"{tool_docstring}\"\"\"\n..."
            tool_descriptions.append(formatted_signature)
            logger.debug(f"  - Parsed tool: {tool_name}")
        # Join all tool descriptions into a single string
        descriptions = "\n".join(tool_descriptions)
        logger.info(f"🔩 Found {len(tool_descriptions)} tools.")
        return descriptions
    
    @step
    async def prepare_chat(self, ev: StartEvent, ctx: Context) -> InputEvent:
        logger.info("💬 Preparing chat for planner agent...")
        await ctx.set("step", "generate_plan")
        self.memory: ChatMemoryBuffer = await ctx.get(
            "memory", default=ChatMemoryBuffer.from_defaults(llm=self.llm)
        )
        user_input = ev.get("input", default=None)
        assert len(self.memory.get_all()) > 0 or user_input or self.user_prompt, "Memory input, user prompt or user input cannot be empty."
        # Add user input to memory
        logger.info("  - Adding goal to memory.")
        if user_input:
            await self.memory.aput(ChatMessage(role="user", content=user_input))
        elif self.user_prompt:
            # Add user prompt to memory
            await self.memory.aput(ChatMessage(role="user", content=self.user_prompt))
        # Update context
        await ctx.set("memory", self.memory)
        input_messages = self.memory.get_all()
        return InputEvent(input=input_messages)
    
    @step
    async def handle_llm_input(self, ev: InputEvent, ctx: Context) -> Union[StopEvent, ModelResponseEvent]:
        """Handle LLM input."""
        # Get chat history from event
        chat_history = ev.input
        assert len(chat_history) > 0, "Chat history cannot be empty."

        self.steps_counter += 1
        logger.info(f"🧠 Step {self.steps_counter}/{self.max_steps}: Calling LLM...")
        if self.steps_counter > self.max_steps:
            logger.warning(f"🚫 Max steps ({self.max_steps}) reached. Stopping execution.")
            return StopEvent(result={'finished':True, 'message':"Max steps reached. Stopping execution.", 'steps': self.steps_counter, 'code_executions': self.code_exec_counter}) # Return final message and steps
        # Get LLM response
        response = await self._get_llm_response(chat_history)
        # Add response to memory
        await self.memory.aput(response.message)
        return ModelResponseEvent(response=response.message.content)
    
    @step
    async def handle_llm_output(self, ev: ModelResponseEvent, ctx: Context) -> Union[StopEvent, ExecutePlan]:
        """Handle LLM output."""
        response = ev.response
        if response:
            logger.info("🤖 LLM response received.")
        logger.info("🤖 LLM output received.")
        planner_step = await ctx.get("step", default=None)
        code, thoughts = self._extract_code_and_thought(response)
        logger.info(f"  - Thoughts: {'Yes' if thoughts else 'No'}, Code: {'Yes' if code else 'No'}")   
        if code:
            # Execute code if present
            logger.info(f"Response: {response}")
            result = await self.executer.execute(code)
            logger.info(f"  - Code executed successfully. Result: {result}")
            # Add result to memory
            await self.memory.aput(ChatMessage(role="user", content=f"Execution Result:\n```\n{result}\n```"))
                    
        if self.task_manager.start_execution:
            self.task_manager.start_execution = False
            logger.info("🚀 Starting task execution.")
            return ExecutePlan()
        elif self.task_manager.task_completed:
            logger.info("✅ Task execution completed.")
            return StopEvent(result={'finished':True, 'message':"Task execution completed.", 'steps': self.steps_counter, 'code_executions': self.code_exec_counter})
        else:
            await self.memory.aput(ChatMessage(role="user", content=f"Neither complete(message: str) or start_agent() was called. If you want to execute the plan, please call start_agent() or complete(message: str) if you are done."))
            logger.info("🚫 Neither complete() nor start_agent() was called. Waiting for next input.")
            return InputEvent(input=self.memory.get_all())
    @step
    async def execute_plan(self, ev: ExecutePlan, ctx: Context) -> Union[ExecutePlan, TaskFailedEvent]:
        """Execute the first pending task from the plan."""
        pending_tasks = self.task_manager.get_pending_tasks()
        
        if not pending_tasks:
            logger.info("Handing back to planner agent.")
            return InputEvent(input=self.memory.get_all())
        pending_task = pending_tasks[0]
        pending_task["status"] = self.task_manager.STATUS_ATTEMPTING
        self.task_manager.save_to_file()
        logger.info(f"🚀 Executing task: {pending_task['description']}")
        goal = pending_task['description']
        task_result = await self.agent.run(input=goal)
        self.agent.memory.reset()
        if task_result["success"]:
            self.current_retry = 0
            logger.info(f"✅ Task {pending_task['description']} completed successfully.")
            # Add task result to memory
            await self.memory.aput(ChatMessage(role="user", content=f"Task: {pending_task["description"]}\n Task Result: Successfully completed."))
            # Mark task as completed
            pending_task["status"] = self.task_manager.STATUS_COMPLETED
            self.task_manager.save_to_file()
            return ExecutePlan()
        else:
            logger.error(f"❌ Task {pending_task['description']} failed.")
            # Add task result to memory
            # await self.memory.aput(ChatMessage(role="user", content=f"Task: {pending_task["description"]}\n Task Result: Failed."))
            # Mark task as failed
            pending_task["status"] = self.task_manager.STATUS_FAILED
            self.task_manager.save_to_file()
            return TaskFailedEvent(task_description=pending_task["description"], reason=task_result["reason"])
        
    @step
    async def reevalue_failure(self, ev: TaskFailedEvent, ctx: Context) -> Union[StopEvent, ExecutePlan]:
        """Reevaluate the failure and decide whether to retry or stop."""
        await ctx.set("step", "failed_task")
        if self.current_retry < self.max_retries:
            logger.info(f"🚫 Task failed. Reevaluating...")
            #await self.memory.aput(ChatMessage(role="user", content=f"Task: {ev.task_description}\nStatus: Failed.\nReason: {ev.reason}."))
            self.task_manager.clear_tasks()
            await self.memory.aput(ChatMessage(role="user", content=DEFAULT_PLANNER_TASK_FAILED_PROMPT.format(goal=self.goal, task_description=ev.task_description, reason=ev.reason)))
            return InputEvent(input=self.memory.get_all())

        else:
            logger.error(f"🚫 Task failed. Max retries reached. Stopping execution.")
            return StopEvent(result={'finished':True, 'message':f"Task failed after {self.max_retries} retries.", 'steps': self.steps_counter, 'code_executions': self.code_exec_counter})


    async def _get_llm_response(self, chat_history: List[ChatMessage]) -> ChatResponse:
        """Get streaming response from LLM."""
        logger.debug(f"  - Sending {len(chat_history)} messages to LLM.")
        # Combine system prompt with chat history
        chat_history = await add_screenshot_image_block(self.tools_instance, chat_history)
        chat_history = await add_ui_text_block(self.tools_instance, chat_history)
        
        messages_to_send = [self.system_message] + chat_history 
        messages_to_send = [message_copy(msg) for msg in messages_to_send]

        response = await self.llm.achat(
            messages=messages_to_send
        )
        assert hasattr(response, "message"), f"LLM response does not have a message attribute.\nResponse: {response}"
        logger.debug("  - Received response from LLM.")
        return response
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from .events import (

    InputEvent,
    ExecutePlan,
    TaskControllerEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    RunTaskEvent,
    TaskResultEvent,
)
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

if TYPE_CHECKING:
    from ...tools import Tools

logger = logging.getLogger("droidrun")
logging.basicConfig(level=logging.INFO)

DEFAULT_PLANNER_SYSTEM_PROMPT = """You are an expert Task Planner Agent. Your purpose is to break down a complex user goal into a sequence of **functional, contextual steps** for controlling an Android device. You do **NOT** specify low-level interactions (like swipes, scrolls, specific coordinates) or execute tasks yourself. You create the plan using specific Planning Tools.

The plan you create will be executed by another agent capable of achieving these functional goals using low-level actions like tapping elements, swiping, inputting text, pressing keys (like HOME, BACK), starting applications, etc. Your awareness of these capabilities ensures your planned steps are achievable, but you should **not** include those low-level actions in your plan tasks.

You will be given a high-level goal. Your task is to devise a step-by-step plan where each step describes a functional goal to achieve within the UI, building upon the context of the previous step.

## Core Objective:
Create a structured plan using the provided Planning Tools, consisting of a sequence of **contextual, functional goals** for the executor agent.

## Characteristics of a Good Plan:
*   **Functional Goal Commands:** Each task describes *what* functional outcome to achieve in the UI (e.g., "Open the Settings app", "Navigate to the WiFi settings screen", "Enter 'password' into the password field", "Select the 'Save' option"). It focuses on the purpose, not the exact physical interaction.
*   **Contextual Phrasing:** After the first task, each subsequent task **MUST** start by stating the assumed current context/screen, followed by the goal. (e.g., "You are on the main Settings screen. Tap the 'Network & internet' option.").
*   **Achievable Goals:** Each functional goal must be something theoretically achievable by the executor using its capabilities (tap, type, swipe, key press, app start, visual check). You should not plan goals that require capabilities the executor doesn't have.
*   **Logical Sequence:** Tasks follow the natural workflow required to achieve the overall user goal.
*   **Appropriate Granularity:** Focus on distinct functional steps or screen transitions. Avoid breaking down single actions (like typing a word) into multiple tasks unless absolutely necessary for clarity. Combine related settings (like hour, minute, AM/PM) into one functional goal if done on the same screen area.
*   **Uses Planning Tools:** The final plan is generated *only* by calling the provided Planning Tools (`set_tasks`, `add_task`) in a ```python ... ``` block.

## Characteristics of a Bad Plan (Avoid These):
*   **Micro-interactions:** Listing low-level actions like "Swipe up", "Scroll down", "Tap coordinates (123, 456)", "Press keycode 66 (ENTER)". The executor handles *how* to achieve the functional goal.
*   **Lack of Context:** Tasks that don't state the assumed starting screen/state (e.g., just saying "Tap 'Save'" without context).
*   **Vague Instructions:** Tasks like "Manage settings", "Check status", "Configure the network". What specific functional goal needs to be achieved?
*   **Includes Execution Logic:** Specifying element IDs, widget types, or specific algorithms for finding elements.
*   **Non-Actionable Goals:** "Think about the next step", "Decide the best option".
*   **General Python Code:** Trying to *perform* actions using general Python.
*   **Passive Phrasing:** "The network should be selected" instead of "Select the 'MyNetwork' WiFi network".

## How to Respond:
1.  **Think Step-by-Step:** Analyze the goal, envisioning the sequence of screens and functional changes needed. Outline your thought process, focusing on defining contextual, functional goals achievable by the executor.
2.  **Generate Code:** Output Python code wrapped in ```python ... ``` tags. This code **MUST** use the provided Planning Tools (`set_tasks` or `add_task`) to define the plan, listing the contextual, functional goal tasks. Prefer `set_tasks` for the initial plan.
3.  **Tool Usage:** You have access ONLY to the Planning Tools below. Your code output *must* only call these.

## Available Planning Tools:
{tools_description}

**Important:**
*   Focus exclusively on **planning contextual, functional goals**. Plan the *what* (functional outcome), letting the executor determine the *how* (specific taps, swipes).
*   Your primary output should be the code block calling the planning tool(s).

## Response Format Example:

**Goal:** Set an alarm for 7:00 AM tomorrow morning on the Android device and return to the home screen.

**Your Thought Process:**
Okay, the goal requires setting an alarm functionally and returning home. I need a sequence of contextual goals:
1.  Goal: Start the Clock app. (Context: Assumed Home Screen or App Drawer) -> Command: Start the 'Clock' application.
2.  Goal: Go to the Alarm section. (Context: Clock app open) -> Command: You are in the Clock app main screen. Navigate to the 'Alarm' section.
3.  Goal: Initiate adding a new alarm. (Context: Alarm section) -> Command: You are in the Alarm section. Initiate adding a new alarm (e.g., tap '+' or 'Add').
4.  Goal: Set the desired time (7:00 AM). (Context: New alarm screen) -> Command: You are on the new alarm screen. Set the alarm time to 7:00 AM.
5.  Goal: Set the desired day (Tomorrow). (Context: New alarm screen, time set) -> Command: The time is set to 7:00 AM on the new alarm screen. Set the alarm day to 'Tomorrow'.
6.  Goal: Save the alarm. (Context: New alarm screen, details set) -> Command: The alarm details (7:00 AM, Tomorrow) are configured. Save the new alarm.
7.  Goal: Verify the alarm exists. (Context: Alarm list screen) -> Command: You are back in the Alarm list screen. Verify the 7:00 AM alarm is listed and enabled.
8.  Goal: Return to Home. (Context: Clock app / Alarm list) -> Command: Alarm is verified in the list. Return to the device's home screen.

This sequence uses functional goals (Start app, Navigate, Set time, Set day, Save, Verify, Return home) with context. The executor will handle the specific taps/swipes needed for each. I'll use `set_tasks`.

```python
# Using the set_tasks planning tool to define the contextual, functional plan.
plan_string = \"\"\"Start the 'Clock' application.
You are in the Clock app main screen. Navigate to the 'Alarm' section.
You are in the Alarm section. Initiate adding a new alarm (e.g., tap '+' or 'Add').
You are on the new alarm screen. Set the alarm time to 7:00 AM.
The time is set to 7:00 AM on the new alarm screen. Set the alarm day to 'Tomorrow'.
The alarm details (7:00 AM, Tomorrow) are configured. Save the new alarm.
You are back in the Alarm list screen. Verify the 7:00 AM alarm is listed and enabled.
Alarm is verified in the list. Return to the device's home screen.\"\"\"

# Call the tool to set these tasks
set_tasks(tasks=plan_string)
```

**(Self-Correction during thought process):** My previous plan might have included "Press HOME key". That's too low-level for this planning style. The correct functional goal is "Return to the device's home screen". Similarly, instead of "Set hour to 7", "Set minutes to 00", "Select AM", the functional goal is "Set the alarm time to 7:00 AM", letting the executor handle the specific interactions on the time picker. This plan provides clearer functional objectives with necessary context for the executor.

Remember: Your output is the **plan itself**, generated by calling the planning tools (primarily `set_tasks`) within ```python ... ``` blocks. Tasks **MUST** be **contextual, functional goals**, not low-level interactions. Adhere strictly to the "Good Plan" characteristics.
"""

DEFAULT_PLANNER_USER_PROMPT = """Goal: {goal}"""

class PlannerAgent(Workflow):
    def __init__(self, goal: str, llm: LLM, agent: Workflow, tools_instance: 'Tools', executer = None, system_prompt = None, user_prompt = None,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.goal = goal
        self.task_manager = TaskManager()
        self.tools = [self.task_manager.set_tasks, self.task_manager.add_task, self.task_manager.get_all_tasks, self.task_manager.clear_tasks]
        self.tools_description = self.parse_tool_descriptions()
        if not executer:
            self.executer = SimpleCodeExecutor(loop=self.loop, globals={}, locals={}, tools=self.tools, use_same_scope=True)
        else:
            self.executer = executer
        self.system_prompt = system_prompt or DEFAULT_PLANNER_SYSTEM_PROMPT.format(tools_description=self.tools_description)
        self.user_prompt = user_prompt or DEFAULT_PLANNER_USER_PROMPT.format(goal=goal)
        self.system_message = ChatMessage(role="system", content=self.system_prompt)
        self.user_message = ChatMessage(role="user", content=self.user_prompt)
        self.memory = None
        self.agent = agent
        self.tools_instance = tools_instance

    
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
    async def prepare_chat(self, ev: StartEvent, ctx: Context):
        logger.info("💬 Preparing chat for planner agent...")
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
            await self.memory.aput(self.user_prompt)
        # Update context
        await ctx.set("memory", self.memory)
        input_messages = self.memory.get()
        return InputEvent(input=input_messages)
    
    @step
    async def generate_plan(self, ev: InputEvent, ctx: Context) -> Union[StopEvent, ExecutePlan]:
        """Generate a plan using the LLM."""
        logger.info("📝 Generating plan...")
        # Get chat history from event
        chat_history = ev.input
        assert len(chat_history) > 0, "Chat history cannot be empty."
        # Get LLM response
        response = await self._get_llm_response(chat_history)
        # Add response to memory
        await self.memory.aput(response.message)
        logger.info("🤖 LLM response received.")
        code, thoughts = self._extract_code_and_thought(response.message.content)
        logger.info(f"  - Thoughts: {'Yes' if thoughts else 'No'}, Code: {'Yes' if code else 'No'}")
        if code:
            # Execute code if present
            logger.info("  - Executing code...")
            try:
                result = await self.executer.execute(code)
                logger.info(f"  - Code executed successfully. Result: {result}")
                # Add result to memory
                await self.memory.aput(ChatMessage(role="user", content=f"Execution Result:\n```\n{result}\n```"))
                if self.task_manager.get_pending_tasks() > 0:
                    return ExecutePlan(plan=self.task_manager.tasks)
                return StopEvent(result={'finished':True, 'message':f"No proper plan was generated. Tasks: {self.task_manager.tasks}", 'steps': 0, 'code_executions': 0}) # Return final message and steps
            except Exception as e:
                logger.error(f"🚫 Code execution failed: {e}")
                return StopEvent(result={'finished':True, 'message':f"Code execution failed: {e}", 'steps': 0, 'code_executions': 0})
        return StopEvent(result={'finished':True, 'message':"No code to execute.", 'steps': 0, 'code_executions': 0}) # Return final message and steps
    
    @step
    async def handle_llm_input(self, ev: InputEvent, ctx: Context) -> Union[StopEvent, ExecutePlan]:
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
        logger.info("🤖 LLM response received.")
        code, thoughts = self._extract_code_and_thought(response.message.content)
        logger.info(f"  - Thoughts: {'Yes' if thoughts else 'No'}, Code: {'Yes' if code else 'No'}")
        if code:
            # Execute code if present
            logger.info("  - Executing code...")
            try:
                result = await self.executer.execute(code)
                logger.info(f"  - Code executed successfully. Result: {result}")
                # Add result to memory
                await self.memory.aput(ChatMessage(role="user", content=f"Execution Result:\n```\n{result}\n```"))
                if self.task_manager.get_pending_tasks() > 0:
                    return ExecutePlan(plan=self.task_manager.tasks)
                return StopEvent(result={'finished':True, 'message':"Execution finished.", 'steps': self.steps_counter, 'code_executions': self.code_exec_counter}) # Return final message and steps
            except Exception as e:
                logger.error(f"🚫 Code execution failed: {e}")
                return TaskFailedEvent(task_id=self.steps_counter)
        return ModelOutputEvent(thoughts=thoughts, code=code)

    @step
    async def execute_plan(self, ev: ExecutePlan, ctx: Context) -> Union[TaskCompletedEvent, TaskFailedEvent]:
        """Execute the first pending task from the plan."""
        pending_task = self.task_manager.get_first_pending_task()[0]
        assert pending_task, "No pending tasks to execute."
        pending_task["status"] = self.task_manager.STATUS_ATTEMPTING
        logger.info(f"🚀 Executing task: {pending_task['description']}")
        task_result = (await self.agent.run(input=pending_task['description']))["result"]
        memory: ChatMemoryBuffer = self.agent.memory.reset()
        if task_result["success"]:
            logger.info(f"✅ Task {pending_task['description']} completed successfully.")
            # Add task result to memory
            await memory.aput(ChatMessage(role="user", content=f"Task: {pending_task["description"]}\n Task Result: Successfully completed."))
            # Mark task as completed
            pending_task["status"] = self.task_manager.STATUS_COMPLETED
            return ExecutePlan(plan=self.task_manager.tasks)
        else:
            logger.error(f"❌ Task {pending_task['description']} failed.")
            # Add task result to memory
            await memory.aput(ChatMessage(role="user", content=f"Task: {pending_task["description"]}\n Task Result: Failed."))
            # Mark task as failed
            pending_task["status"] = self.task_manager.STATUS_FAILED
            return TaskFailedEvent(task_id=pending_task["id"])
        
    @step
    async def reevalue_failure(self, ev: TaskFailedEvent, ctx: Context) -> Union[StopEvent, ExecutePlan]:
        """Reevaluate the failure and decide whether to retry or stop."""
        logger.info(f"🚫 Task failed. Reevaluating...")
        await self.memory.aput(ChatMessage(role="user", content=f"Task: {ev.task_description}\nStatus: Failed.\nReason: {ev.reason}."))
        self.task_manager.clear_tasks()
        await self.memory.aput(ChatMessage(role="user", content=f"All tasks list has been cleared. However the state of the phone is still the same, so all previous completed tasks and the failed task remain in effect. From this point replan."))

        


    
    




    async def _get_llm_response(self, chat_history: List[ChatMessage]) -> ChatResponse:
        """Get streaming response from LLM."""
        logger.debug(f"  - Sending {len(chat_history)} messages to LLM.")
        # Combine system prompt with chat history
        chat_history = await add_screenshot_image_block(self.tools, chat_history)
        chat_history = await add_ui_text_block(self.tools, chat_history)
        
        messages_to_send = [self.system_prompt] + chat_history 
        response = await self.llm.achat(
            messages=messages_to_send
        )
        assert hasattr(response, "message"), f"LLM response does not have a message attribute.\nResponse: {response}"
        logger.debug("  - Received response from LLM.")
        return response
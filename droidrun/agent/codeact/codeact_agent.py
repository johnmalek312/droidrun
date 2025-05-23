import base64
import json
import logging
import re
import inspect
from enum import Enum
import time
from typing import Awaitable, Callable, List, Optional, Dict, Any, Tuple, TYPE_CHECKING, Union

# LlamaIndex imports for LLM interaction and types
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ImageBlock, TextBlock
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Context, step
from llama_index.core.memory import ChatMemoryBuffer
from .events import FinalizeEvent, InputEvent, ModelOutputEvent, ExecutionEvent, ExecutionResultEvent
from llama_index.core import set_global_handler
set_global_handler("arize_phoenix")
# Load environment variables (for API key)
from dotenv import load_dotenv
load_dotenv()

if TYPE_CHECKING:
    from ...tools import Tools

logger = logging.getLogger("droidrun")
logging.basicConfig(level=logging.INFO)

# Simple token estimator (very rough approximation)
def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in a string.
    
    This is a very rough approximation based on the rule of thumb that
    1 token is approximately 4 characters for English text.
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // 4 + 1  # Add 1 to be safe


def message_copy(message: ChatMessage, deep = True) -> ChatMessage:
    if deep:
        copied_message = message.model_copy()
        copied_message.blocks = [block.model_copy () for block in message.blocks]

        return copied_message
    copied_message = message.model_copy()

    # Create a new, independent list containing the same block references
    copied_message.blocks = list(message.blocks) # or original_message.blocks[:]

    return copied_message


# --- Agent Definition ---

DEFAULT_CODE_ACT_SYSTEM_PROMPT = """You are a helpful AI assistant that can write and execute Python code to solve problems.

You will be given a task to perform. You should output:
- Python code wrapped in ``` tags that provides the solution to the task, or a step towards the solution. Any output you want to extract from the code should be printed to the console.
- Text to be shown directly to the user, if you want to ask for more information or provide the final answer.
- If the previous code execution can be used to respond to the user, then respond directly (typically you want to avoid mentioning anything related to the code execution in your response).
- If you output thoughts without code or empty code, the program will assume you are done and will stop execution, so if you want to provide empty code with thoughts, just write time.sleep(1) in a code block to avoid this.

## Response Format:
Example of proper code format:
To calculate the area of a circle, I need to use the formula: area = pi * radius^2. I will write a function to do this.
```python
import math

def calculate_area(radius):
    return math.pi * radius**2

# Calculate the area for radius = 5
area = calculate_area(5)
print(f"The area of the circle is {area:.2f} square units")
```

Another example (with for loop):
To calculate the sum of numbers from 1 to 10, I will use a for loop.
```python
sum = 0
for i in range(1, 11):
    sum += i
print(f"The sum of numbers from 1 to 10 is {sum}")
```

In addition to the Python Standard Library and any functions you have already written, you can use the following functions:
{tool_descriptions}

Most functions return a value, inorder to see the result of the function, you MUST print it.
Some functions may be bound instance methods, so if you encounter error you might think they need an extra parameter (self) but they don't.
You'll receive a screenshot showing the current screen and its UI elements to help you complete the task. However, screenshots won’t be saved in the chat history. So, make sure to describe what you see and explain the key parts of your plan in your thoughts, as those will be saved and used to assist you in future steps.

## Final Answer Guidelines:
- When providing a final answer, focus on directly answering the user's question
- Avoid referencing the code you generated unless specifically asked
- Present the results clearly and concisely as if you computed them directly
- If relevant, you can briefly mention general methods used, but don't include code snippets in the final answer
- Structure your response like you're directly answering the user's query, not explaining how you solved it

Reminder: Always place your Python code between ```...``` tags when you want to run code. 

You MUST ALWAYS to include your reasoning and thought process outside of the code block. You MUST DOUBLE CHECK that TASK IS COMPLETE with a SCREENSHOT.
"""

DEFAULT_CODE_ACT_USER_PROMPT = """**Current Request:**
Goal: {goal}

**What is your reasoning and the next step to address this request?** Explain your plan first, then provide code in ```python ... ``` tags if needed."""

DEFAULT_NO_THOUGHTS_PROMPT = """Your previous response provided code without explaining your reasoning first. Remember to always describe your thought process and plan *before* providing the code block.

The code you provided will be executed below.

Now, describe the next step you will take to address the original goal: {goal}"""

# --- Updated System Prompt ---
# Guides the LLM towards a Thought -> Code -> Observation cycle
class CodeActAgent(Workflow):
    """
    An agent that uses a ReAct-like cycle (Thought -> Code -> Observation)
    to solve problems requiring code execution. It extracts code from
    Markdown blocks and uses specific step types for tracking.
    """
    def __init__(
        self,
        goal: str,
        llm: LLM,
        code_execute_fn: Callable[[str], Awaitable[Dict[str, Any]]],
        tools: 'Tools',
        available_tools: List = [],
        max_steps: int = 10, # Default max steps
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        always_screenshot: bool = True,
        always_ui: bool = False,
        *args,
        **kwargs
    ):
        # assert instead of if
        assert llm, "llm must be provided."
        assert code_execute_fn, "code_execute_fn must be provided"
        assert goal, "goal must be provided."
        assert max_steps > 0, "max_steps must be greater than 0."
        super().__init__(*args, **kwargs)

        self.llm = llm
        self.code_execute_fn = code_execute_fn
        self.available_tools = available_tools or []
        self.tools = tools
        self.max_steps = max_steps
        self.goal = goal
        self.tool_descriptions = self.parse_tool_descriptions() # Parse tool descriptions once at initialization
        self.system_prompt = ChatMessage(role="system", content=PromptTemplate(system_prompt or DEFAULT_CODE_ACT_SYSTEM_PROMPT).format(tool_descriptions=self.tool_descriptions))
        self.user_prompt = ChatMessage(role="user", content=PromptTemplate(user_prompt or DEFAULT_CODE_ACT_USER_PROMPT).format(goal=goal))
        self.no_thoughts_prompt = ChatMessage(role="user", content=PromptTemplate(DEFAULT_NO_THOUGHTS_PROMPT).format(goal=goal))
        self.memory = None 
        self.steps_counter = 0 # Initialize step counter
        self.code_exec_counter = 0 # Initialize execution counter
        self.always_screenshot = always_screenshot # Flag to send screenshot with every prompt
        self.always_ui = always_ui # Flag to always send UI elements
        logger.info("✅ CodeActAgent initialized successfully.")

    def parse_tool_descriptions(self) -> str:
        """Parses the available tools and their descriptions for the system prompt."""
        logger.info("🛠️ Parsing tool descriptions...")
        # self.available_tools is a list of functions, we need to get their docstrings, names, and signatures and display them as `def name(args) -> return_type:\n"""docstring"""    ...\n`
        tool_descriptions = []
        for tool in self.available_tools:
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

    @step
    async def prepare_chat(self, ev: StartEvent, ctx: Context) -> InputEvent:
        """Prepare chat history from user input."""
        logger.info("💬 Preparing chat history...")
        # Get or create memory
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
    async def handle_llm_input(self, ev: InputEvent, ctx: Context) -> Union[ModelOutputEvent, FinalizeEvent]:
        """Handle LLM input."""
        # Get chat history from event
        chat_history = ev.input
        assert len(chat_history) > 0, "Chat history cannot be empty."

        self.steps_counter += 1
        logger.info(f"🧠 Step {self.steps_counter}/{self.max_steps}: Calling LLM...")
        if self.steps_counter > self.max_steps:
            logger.warning(f"🚫 Max steps ({self.max_steps}) reached. Stopping execution.")
            return FinalizeEvent(result={'finished':True, 'message':"Max steps reached. Stopping execution.", 'steps': self.steps_counter, 'code_executions': self.code_exec_counter}) # Return final message and steps
        # Get LLM response
        response = await self._get_llm_response(chat_history)
        # Add response to memory
        await self.memory.aput(response.message)
        logger.info("🤖 LLM response received.")
        code, thoughts = self._extract_code_and_thought(response.message.content)
        logger.info(f"  - Thoughts: {'Yes' if thoughts else 'No'}, Code: {'Yes' if code else 'No'}")
        return ModelOutputEvent(thoughts=thoughts, code=code)

    @step
    async def handle_llm_output(self, ev: ModelOutputEvent, ctx: Context) -> Union[ExecutionEvent, FinalizeEvent]:
        """Handle LLM output."""
        logger.info("⚙️ Handling LLM output...")
        # Get code and thoughts from event
        code = ev.code
        thoughts = ev.thoughts

        # Warning if no thoughts are provided
        if not thoughts:
            logger.warning("🤔 LLM provided code without thoughts. Adding reminder prompt.")
            await self.memory.aput(self.no_thoughts_prompt)
        else:
            # print thought but start with emoji at the start of the log
            logger.info(f"🤔 Thoughts: {thoughts}...")

        # If code is present, execute it
        if code:
            return ExecutionEvent(code=code)
        else:
            final_message = thoughts or "No code provided and no final message."
            logger.info(f"✅ No code to execute. Stopping workflow. Final Message: {final_message}...")
            return FinalizeEvent(result={'finished': True, 'message': final_message, 'steps': self.steps_counter, 'code_executions': self.code_exec_counter}) # Return final message and steps

    @step
    async def execute_code(self, ev: ExecutionEvent, ctx: Context) -> ExecutionResultEvent:
        """Execute the code and return the result."""
        code = ev.code
        assert code, "Code cannot be empty."
        logger.info(f"⚡ Executing code:\n```python\n{code}\n```")
        # Execute the code using the provided function
        try:
            self.code_exec_counter += 1
            result = await self.code_execute_fn(code)
            logger.info(f"💡 Code execution successful. Result: {result}")
            return ExecutionResultEvent(output=str(result)) # Ensure output is string
        except Exception as e:
            logger.error(f"💥 Code execution failed: {e}", exc_info=True)
            error_message = f"Error during execution: {e}"
            return ExecutionResultEvent(output=error_message) # Return error message as output

    @step
    async def handle_execution_result(self, ev: ExecutionResultEvent, ctx: Context) -> InputEvent:
        """Handle the execution result. Currently it just returns InputEvent."""
        logger.info("📊 Handling execution result...")
        # Get the output from the event
        output = ev.output
        if output is None:
            output = "Code executed, but produced no output."
            logger.warning("  - Execution produced no output.")
        else:
             logger.info(f"  - Execution output: {output[:100]}...") # Log first 100 chars
        # Add the output to memory as an user message (observation)
        observation_message = ChatMessage(role="user", content=f"Execution Result:\n```\n{output}\n```")
        await self.memory.aput(observation_message)
        logger.info("  - Added execution result to memory.")
        return InputEvent(input=self.memory.get())
    

    @step
    async def finalize(self, ev: FinalizeEvent, ctx: Context) -> StopEvent:
        """Finalize the workflow."""
        logger.info("🔚 Finalizing workflow...")
        ctx.set("memory", self.memory) # Ensure memory is set in context
        return StopEvent(result=ev.result)

    async def _get_llm_response(self, chat_history: List[ChatMessage]) -> ChatResponse:
        """Get streaming response from LLM."""
        logger.debug(f"  - Sending {len(chat_history)} messages to LLM.")
        # Combine system prompt with chat history
        if self.always_screenshot:
            chat_history = await add_screenshot_image_block(self.tools, chat_history)
        elif self.tools.last_screenshot:
            chat_history = await add_screenshot(chat_history, self.tools.last_screenshot)
            self.tools.last_screenshot = None # Reset last screenshot after sending it
        if self.always_ui:
            chat_history = await add_ui_text_block(self.tools, chat_history)
        
        messages_to_send = [self.system_prompt] + chat_history 
        try:
            response = await self.llm.achat(
                messages=messages_to_send
            )
            assert hasattr(response, "message"), f"LLM response does not have a message attribute.\nResponse: {response}"
        except Exception as e:
            if self.llm.class_name() == "Gemini_LLM" and "You exceeded your current quota" in str(e):
                    s = str(e._details[2])
                    match = re.search(r'seconds:\s*(\d+)', s)
                    if match:
                        seconds = int(match.group(1)) + 1
                        logger.error(f"Rate limit error. Retrying in {seconds} seconds...")
                        time.sleep(seconds)
                    else:
                        logger.error(f"Rate limit error. Retrying in 5 seconds...")
                        time.sleep(40)
                    response = await self.llm.achat(
                        messages=messages_to_send
                    )
            else:
                logger.error(f"Error getting LLM response: {e}")
                return StopEvent(result={'finished': True, 'message': f"Error getting LLM response: {e}", 'steps': self.steps_counter, 'code_executions': self.code_exec_counter}) # Return final message and steps
        #time.sleep(3)
        logger.debug("  - Received response from LLM.")
        return response
    
async def add_ui_text_block(tools: 'Tools', chat_history: List[ChatMessage], retry = 5) -> List[ChatMessage]:
    """Add UI elements to the chat history without modifying the original."""
    ui_elements = None
    for i in range(retry):
        try:
            ui_elements = await tools.get_clickables()
            if ui_elements:
                break
        except Exception as e:
            if i < 4:
                logger.warning(f"  - Error getting UI elements: {e}. Retrying...")
            else:
                logger.error(f"  - Error getting UI elements: {e}. No UI elements will be sent.")
    if ui_elements:
        ui_block = TextBlock(text="\nCurrent Clickable UI elements from the device using the custom TopViewService:\n```json\n" + json.dumps(ui_elements) + "\n```\n")
        chat_history = chat_history.copy()
        chat_history[-1] = message_copy(chat_history[-1])
        chat_history[-1].blocks.append(ui_block)
    return chat_history

async def add_screenshot_image_block(tools: 'Tools', chat_history: List[ChatMessage], retry: int = 5) -> None:
    screenshot = await take_screenshot(tools, retry)
    if screenshot:
        image_block = ImageBlock(image=base64.b64encode(screenshot))
        chat_history = chat_history.copy()  # Create a copy of chat history to avoid modifying the original
        chat_history[-1] = message_copy(chat_history[-1])
        chat_history[-1].blocks.append(image_block)
    return chat_history


async def take_screenshot(tools: 'Tools', retry: int = 5) -> None:
    """Take a screenshot and return the image."""
    # Retry taking screenshot
    tools.last_screenshot = None
    for i in range(retry):
        try:
            await tools.take_screenshot()
            if tools.last_screenshot:
                break
        except Exception as e:
            if i < 4:
                logger.warning(f"  - Error taking screenshot: {e}. Retrying...")
            else:
                logger.error(f"  - Error taking screenshot: {e}. No screenshot will be sent.")
                return None
    screenshot = tools.last_screenshot
    tools.last_screenshot = None # Reset last screenshot after taking it
    return screenshot

async def add_screenshot(chat_history: List[ChatMessage], screenshot) -> List[ChatMessage]:
    """Add a screenshot to the chat history."""
    image_block = ImageBlock(image=base64.b64encode(screenshot))
    chat_history = chat_history.copy()  # Create a copy of chat history to avoid modifying the original
    chat_history[-1] = message_copy(chat_history[-1])
    chat_history[-1].blocks.append(image_block)
    return chat_history

import base64
import logging
import re
import inspect
from enum import Enum
from typing import Awaitable, Callable, List, Optional, Dict, Any, Tuple, TYPE_CHECKING, Union

# LlamaIndex imports for LLM interaction and types
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ImageBlock, TextBlock
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Context, step
from llama_index.core.memory import ChatMemoryBuffer
from .events import InputEvent, ModelOutputEvent, ExecutionEvent, ExecutionResultEvent
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

# --- Agent Definition ---

DEFAULT_CODE_ACT_SYSTEM_PROMPT = """You are a helpful AI assistant that can write and execute Python code to solve problems.

You will be given a task to perform. You should output:
- Python code wrapped in ``` tags that provides the solution to the task, or a step towards the solution. Any output you want to extract from the code should be printed to the console.
- Text to be shown directly to the user, if you want to ask for more information or provide the final answer.
- If the previous code execution can be used to respond to the user, then respond directly (typically you want to avoid mentioning anything related to the code execution in your response).
-

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

In addition to the Python Standard Library and any functions you have already written, you can use the following functions:
{tool_descriptions}

All code executions from previous steps are not available in the current step. You can only use the code you write in this step.

## Final Answer Guidelines:
- When providing a final answer, focus on directly answering the user's question
- Avoid referencing the code you generated unless specifically asked
- Present the results clearly and concisely as if you computed them directly
- If relevant, you can briefly mention general methods used, but don't include code snippets in the final answer
- Structure your response like you're directly answering the user's query, not explaining how you solved it
- Write simple code to complete the task step by step, dont try to solve the entire task in one go.

Reminder: Always place your Python code between ```...``` tags when you want to run code. 

You MUST ALWAYS to include your reasoning and thought process outside of the code block.
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
        self.memory = ChatMemoryBuffer.from_defaults(llm=self.llm) # Initialize memory buffer
        self.steps_counter = 0 # Initialize step counter
        self.code_exec_counter = 0 # Initialize execution counter
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
        Extracts code from Markdown blocks (```python ... ```) and the surrounding text (thought).

        Returns:
            Tuple[Optional[code_string], thought_string]
        """
        logger.debug("✂️ Extracting code and thought from response...")
        code_pattern = r"```python\s*\n(.*?)\n```"
        code_matches = list(re.finditer(code_pattern, response_text, re.DOTALL))

        if not code_matches:
            # No code found, the entire response is thought
            logger.debug("  - No code block found. Entire response is thought.")
            return None, response_text.strip()

        # Combine all extracted code blocks
        extracted_code = "\n\n".join([match.group(1).strip() for match in code_matches])
        logger.debug(f"  - Extracted code:\n```python\n{extracted_code}\n```")

        # Extract thought text (text before the first code block and after the last)
        thought_parts = []
        last_end = 0
        for match in code_matches:
            start, end = match.span()
            thought_parts.append(response_text[last_end:start])
            last_end = end
        thought_parts.append(response_text[last_end:]) # Text after the last block

        thought_text = "".join(thought_parts).strip()
        logger.debug(f"  - Extracted thought: {thought_text}...")

        return extracted_code, thought_text

    @step
    async def prepare_chat(self, ev: StartEvent, ctx: Context) -> InputEvent:
        """Prepare chat history from user input."""
        logger.info("💬 Preparing chat history...")
        # Get or create memory
        memory: ChatMemoryBuffer = await ctx.get(
            "memory", default=ChatMemoryBuffer.from_defaults(llm=self.llm)
        )
        user_input = ev.get("input", default=None)
        assert len(memory.get_all()) > 0 or user_input or self.user_prompt, "Memory input, user prompt or user input cannot be empty."
        # Add user input to memory
        logger.info("  - Adding goal to memory.")
        if user_input:
            await memory.aput(ChatMessage(role="user", content=user_input))
        elif self.user_prompt:
            # Add user prompt to memory
            await memory.aput(self.user_prompt)
        # Update context
        await ctx.set("memory", memory)
        input_messages = memory.get()
        return InputEvent(input=input_messages)
    @step
    async def handle_llm_input(self, ev: InputEvent, ctx: Context) -> Union[ModelOutputEvent, StopEvent]:
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
        return ModelOutputEvent(thoughts=thoughts, code=code)

    @step
    async def handle_llm_output(self, ev: ModelOutputEvent, ctx: Context) -> Union[ExecutionEvent, StopEvent]:
        """Handle LLM output."""
        logger.info("⚙️ Handling LLM output...")
        # Get code and thoughts from event
        code = ev.code
        thoughts = ev.thoughts
        # Warning if no thoughts are provided
        if not thoughts:
            logger.warning("🤔 LLM provided code without thoughts. Adding reminder prompt.")
            await self.memory.aput(self.no_thoughts_prompt)
        # If code is present, execute it
        if code:
            return ExecutionEvent(code=code)
        else:
            final_message = thoughts or "No code provided and no final message."
            logger.info(f"✅ No code to execute. Stopping workflow. Final Message: {final_message}...")
            return StopEvent(result={'finished': True, 'message': final_message, 'steps': self.steps_counter, 'code_executions': self.code_exec_counter}) # Return final message and steps

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


    async def _get_llm_response(self, chat_history: List[ChatMessage]) -> ChatResponse:
        """Get streaming response from LLM."""
        logger.debug(f"  - Sending {len(chat_history)} messages to LLM.")
        # Combine system prompt with chat history
        if self.tools.last_screenshot:
            image_block = ImageBlock(image=base64.b64encode(self.tools.last_screenshot))
            self.tools.last_screenshot = None
            chat_history = chat_history.copy() # Create a copy of chat history to avoid modifying the original
            chat_history[-1].blocks.append(image_block)
        messages_to_send = [self.system_prompt] + chat_history
        response = await self.llm.achat(
            messages=messages_to_send
        )
        logger.debug("  - Received response from LLM.")
        return response
import logging
from rich.layout import Layout
from rich.panel import Panel
from rich.spinner import Spinner
from rich.console import Console
from rich.live import Live
from typing import List

from droidrun.agent.common.events import ScreenshotEvent
from droidrun.agent.planner.events import (
    PlanInputEvent,
    PlanThinkingEvent,
    PlanCreatedEvent,
)
from droidrun.agent.codeact.events import (
    TaskInputEvent,
    TaskThinkingEvent,
    TaskExecutionEvent,
    TaskExecutionResultEvent,
    TaskEndEvent,
)
from droidrun.agent.droid.events import (
    CodeActExecuteEvent,
    CodeActResultEvent,
    ReasoningLogicEvent,
    TaskRunnerEvent,
    FinalizeEvent,
)


class LogHandler(logging.Handler):
    def __init__(self, goal: str, current_step: str = "Initializing..."):
        super().__init__()

        self.goal = goal
        self.current_step = current_step
        self.is_completed = False
        self.is_success = False
        self.spinner = Spinner("dots")
        self.console = Console()
        self.layout = self._create_layout()
        self.logs: List[str] = []

    def emit(self, record):
        # Format the record as text
        msg = self.format(record)

        # Print the log line(s) directly.  When a `Live` display is active on
        # the same console, Rich will automatically scroll the printed text
        # above the live region – giving us an effectively unlimited stream
        # of historical logs that can be viewed via the terminal scroll-back
        # buffer.
        self.console.print(msg, highlight=False, overflow="fold")

        # Keep an internal list so that we can still show a count of entries
        # (and potentially other future metadata).  We no longer truncate it.
        self.logs.extend(msg.splitlines())

        # Refresh the goal / status panels.
        self.rerender()

    def render(self):
        return Live(self.layout, refresh_per_second=4, console=self.console)

    def rerender(self):
        self._update_layout(
            self.layout,
            self.logs,
            self.current_step,
            self.goal,
            self.is_completed,
            self.is_success,
        )

    def update_step(self, step: str):
        self.current_step = step
        self.rerender()

    def _create_layout(self):
        """Create a layout with logs at top and status at bottom"""
        layout = Layout()
        layout.split(
            # The logs are now printed directly to the console, so this section
            # is just a zero-height placeholder to keep the layout structure
            # intact.
            #Layout(name="logs", size=0),
            Layout(name="goal", size=3),
            Layout(name="status", size=3),
        )
        return layout

    def _update_layout(
        self,
        layout: Layout,
        log_list: List[str],
        step_message: str,
        goal: str = None,
        completed: bool = False,
        success: bool = False,
    ):
        """Update the layout with current logs and step information"""
        from rich.text import Text
        import shutil

        # Cache terminal size to avoid frequent recalculation
        try:
            terminal_height = shutil.get_terminal_size().lines
        except:
            terminal_height = 24  # fallback

        # We no longer render the full activity log inside the live layout – it
        # scrolls naturally above.  Leave the "logs" placeholder empty.
        #layout["logs"].update("")

        if goal:
            goal_text = Text(goal, style="bold")
            layout["goal"].update(
                Panel(
                    goal_text,
                    title="Goal",
                    border_style="magenta",
                    title_align="left",
                    padding=(0, 1),
                    height=3,
                )
            )

        step_display = Text()

        if completed:
            if success:
                step_display.append("✓ ", style="bold green")
                panel_title = "Completed"
                panel_style = "green"
            else:
                step_display.append("✗ ", style="bold red")
                panel_title = "Failed"
                panel_style = "red"
        else:
            step_display.append("⚡ ", style="bold yellow")
            panel_title = "Status"
            panel_style = "yellow"

        step_display.append(step_message)

        layout["status"].update(
            Panel(
                step_display,
                title=panel_title,
                border_style=panel_style,
                title_align="left",
                padding=(0, 1),
                height=3,
            )
        )

    def handle_event(self, event):
        """Handle streaming events from the agent workflow."""
        logger = logging.getLogger("droidrun")

        # Log different event types with proper names
        if isinstance(event, ScreenshotEvent):
            logger.debug("📸 Taking screenshot...")

        # Planner events
        elif isinstance(event, PlanInputEvent):
            self.current_step = "Planning..."
            logger.info("💭 Planner receiving input...")

        elif isinstance(event, PlanThinkingEvent):
            if event.thoughts:
                thoughts_preview = (
                    event.thoughts[:150] + "..."
                    if len(event.thoughts) > 150
                    else event.thoughts
                )
                logger.info(f"🧠 Planning: {thoughts_preview}")
            if event.code:
                logger.info(f"📝 Generated plan code")

        elif isinstance(event, PlanCreatedEvent):
            if event.tasks:
                task_count = len(event.tasks) if event.tasks else 0
                self.current_step = f"Plan ready ({task_count} tasks)"
                logger.info(f"📋 Plan created with {task_count} tasks")
                for task in event.tasks:
                    desc = task.description
                    logger.info(f"- {desc}")

        # CodeAct events
        elif isinstance(event, TaskInputEvent):
            self.current_step = "Processing task input..."
            logger.info("💬 Task input received...")

        elif isinstance(event, TaskThinkingEvent):
            if hasattr(event, "thoughts") and event.thoughts:
                thoughts_preview = (
                    event.thoughts[:150] + "..."
                    if len(event.thoughts) > 150
                    else event.thoughts
                )
                logger.info(f"🧠 Thinking: {thoughts_preview}")
            if hasattr(event, "code") and event.code:
                logger.info(f"💻 Executing action code")
                logger.debug(f"{event.code}")

        elif isinstance(event, TaskExecutionEvent):
            self.current_step = "Executing action..."
            logger.info(f"⚡ Executing action...")

        elif isinstance(event, TaskExecutionResultEvent):
            if hasattr(event, "output") and event.output:
                output = str(event.output)
                if "Error" in output or "Exception" in output:
                    output_preview = (
                        output[:100] + "..." if len(output) > 100 else output
                    )
                    logger.info(f"❌ Action error: {output_preview}")
                else:
                    output_preview = (
                        output[:100] + "..." if len(output) > 100 else output
                    )
                    logger.info(f"⚡ Action result: {output_preview}")

        elif isinstance(event, TaskEndEvent):
            if hasattr(event, "success") and hasattr(event, "reason"):
                if event.success:
                    self.current_step = event.reason
                    logger.info(f"✅ Task completed: {event.reason}")
                else:
                    self.current_step = f"Task failed"
                    logger.info(f"❌ Task failed: {event.reason}")

        # Droid coordination events
        elif isinstance(event, CodeActExecuteEvent):
            self.current_step = "Executing task..."
            logger.info(f"🔧 Starting task execution...")

        elif isinstance(event, CodeActResultEvent):
            if hasattr(event, "success") and hasattr(event, "reason"):
                if event.success:
                    self.current_step = event.reason
                    logger.info(f"✅ Task completed: {event.reason}")
                else:
                    self.current_step = f"Task failed"
                    logger.info(f"❌ Task failed: {event.reason}")

        elif isinstance(event, ReasoningLogicEvent):
            self.current_step = "Planning..."
            logger.info(f"🤔 Planning next steps...")

        elif isinstance(event, TaskRunnerEvent):
            self.current_step = "Processing tasks..."
            logger.info(f"🏃 Processing task queue...")

        elif isinstance(event, FinalizeEvent):
            if hasattr(event, "success") and hasattr(event, "reason"):
                self.is_completed = True
                self.is_success = event.success
                if event.success:
                    self.current_step = f"Success: {event.reason}"
                    logger.info(f"🎉 Goal achieved: {event.reason}")
                else:
                    self.current_step = f"Failed: {event.reason}"
                    logger.info(f"❌ Goal failed: {event.reason}")

        else:
            logger.debug(f"🔄 {event.__class__.__name__}")

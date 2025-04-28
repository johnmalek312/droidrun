"""
DroidRun - A framework for controlling Android devices through LLM agents.
"""

__version__ = "0.1.0"

# Import main classes for easier access
from .agent.codeact.codeact_agent import CodeActAgent as Agent

# Make main components available at package level
__all__ = [
    "Agent"
]
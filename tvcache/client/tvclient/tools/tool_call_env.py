"""ToolCallEnv class for TVCache."""

from abc import ABC, abstractmethod
from typing import Optional

class ToolCall(ABC):
    """Represents a command to be executed in the tool call environment."""

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the tool command.

        Args:
            keystrokes: The keystrokes to execute.
            is_blocking: Whether the command is blocking.
            timeout_sec: The timeout for the command.
        """
    
    @abstractmethod
    def to_dict(self) -> str:
        """Convert the tool call to a serializable dictionary representation.

        Returns:
            A dictionary containing the tool call's data.
        """

    @staticmethod
    @abstractmethod
    def from_dict() -> 'ToolCall':
        """Create a ToolCall instance from a dictionary representation.

        Args:
            data: Dictionary containing the tool call data.

        Returns:
            A ToolCall instance constructed from the dictionary.
        """
    
    @abstractmethod
    def will_mutate_state(self) -> bool:
        """Tells whether this tool call will update the environment

        Returns:
            boolean. True if this tool call will modify the enviornment, False otherwise.
        """

class ToolCallEnv(ABC):
    """Represents the environment for tool calls."""

    @abstractmethod
    def __init__(self, env_id: Optional[str] = None, task_name: str = "default_task"):
        """Initialize the tool call environment.

        Args:
            env_id: Optional environment ID. If None, creates a new environment.
                    If provided, restores an existing environment with that ID.
        """
        pass

    @abstractmethod
    def stop(self, **kwargs) -> None:
        """Stop the tool call environment.

        Stops and cleans up the environment.
        """
        pass

    @abstractmethod
    def execute(self, tool_call, **kwargs):
        """Execute a tool call in the environment.

        Args:
            tool_call: The tool call to execute.

        Returns:
            The result of executing the tool call.
        """
        pass

    @abstractmethod
    def fork(self, **kwargs) -> 'ToolCallEnv':
        """Fork the current environment.

        Creates a copy of the current environment state.

        Returns:
            A new ToolCallEnv instance at the same state as the forked environment.
        """
        pass

    @abstractmethod
    def get_state(self, **kwargs):
        """Get the current state of the environment.

        Returns:
            The current environment state.
        """
        pass

    @abstractmethod
    def get_id(self, **kwargs) -> str:
        """Get a unique ID for this environment
        
        Returns:
            string representing the unique ID of this environment
        """
        pass

    @abstractmethod
    def test(self) -> str:
        """
        Execute tests to evaluate the task completion

        Returns:
            string representing the test result
        """
        pass
    @abstractmethod
    def hash(self) -> str:
        """Get the state hash of the environment.

        Returns:
            A dictionary representing the state hash.
        """
        pass
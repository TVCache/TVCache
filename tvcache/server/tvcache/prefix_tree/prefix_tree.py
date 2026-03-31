"""Abstract base class for prefix tree cache implementations."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional


class PrefixTree(ABC):
    """Abstract interface for prefix tree cache implementations used by tvcache_server."""

    @abstractmethod
    def get(self, task_name: str, tool_calls: List[str]) -> Tuple[bool, Optional[str], Any, Optional[float]]:
        """Get cached environment by task name and tool calls.

        Args:
            task_name: The name of the task
            tool_calls: List of tool calls

        Returns:
            Tuple of (found, env_id, value, tool_exec_time)
        """
        pass

    @abstractmethod
    def prefix_match(self, task_name: str, tool_calls: List[str]) -> Tuple[bool, Optional[str], Any]:
        """Find cached environments with matching tool call prefix.

        Args:
            task_name: The name of the task
            tool_calls: List of tool calls to match against

        Returns:
            Tuple of (found, env_id, history)
        """
        pass
    
    @abstractmethod
    def intel_prefix_match(self, task_name: str, tool_calls: List[str]) -> Tuple[bool, Optional[str], Any]:
        """Find cached environments with matching tool call prefix.

        Args:
            task_name: The name of the task
            tool_calls: List of tool calls to match against

        Returns:
            Tuple of (found, env_id, history)
        """
        pass
    @abstractmethod
    def put(
        self,
        task_name: str,
        history: List[str],
        env_id: str,
        values: Optional[List[Any]] = None,
        tool_exec_times: Optional[List[float]] = None,
        start_idx: int = 0,
        ttl: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """Store tool call history.

        Args:
            task_name: The name of the task
            history: The list of tool calls
            env_id: The environment ID
            values: List of values for each tool call (optional)
            tool_exec_times: List of execution times for each tool call (optional)
            start_idx: Starting index for values/tool_exec_times arrays (default 0)

        Returns:
            Tuple of (success, removed_env_ids)
        """
        pass

    @abstractmethod
    def delete_path(self, task_name: str, history: List[str]) -> bool:
        """Delete a specific tool call history path.

        Args:
            task_name: The name of the task
            history: The list of tool calls

        Returns:
            Boolean indicating success
        """
        pass

    @abstractmethod
    def can_extend(
        self,
        task_name: str,
        history: List[str],
        suffix: List[str]
    ) -> Tuple[bool, List[str]]:
        """Check if a node can be extended with a suffix and mark it as consumed.

        Args:
            task_name: The name of the task
            history: The list of tool calls leading to the node
            suffix: The suffix to potentially extend with

        Returns:
            Tuple of (can_extend, returned_suffix)
        """
        pass

    @abstractmethod
    def should_fork(self, task_name: str, history: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if a node should be forked based on its consumed status.

        Args:
            task_name: The name of the task
            history: The list of tool calls leading to the node

        Returns:
            Tuple of (should_fork, env_id)
        """
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Get the entire prefix tree structure for visualization.

        Returns:
            Dictionary representing the complete prefix tree structure
        """
        pass

    @abstractmethod
    def get_hot_nodes(self, k: int = 1) -> Dict[str, List[List[str]]]:
        """Get hot nodes (nodes with more than k children) from the cache.

        Args:
            k: The threshold for number of children (default: 1)

        Returns:
            Dictionary mapping task names to lists of paths
        """
        pass

    @abstractmethod
    def check_env_marked(self, env_id: str) -> bool:
        """Check if an environment ID is marked as present in the cache.

        Args:
            env_id: The environment ID to check

        Returns:
            Boolean indicating if the env_id is marked as present
        """
        pass

    @abstractmethod
    def store_test_result(self, task_name: str, history: List[str], test_result: str) -> bool:
        """Store a test result for a specific node in the cache.

        Args:
            task_name: The name of the task
            history: The list of tool calls leading to the node
            test_result: The test result string to store

        Returns:
            Boolean indicating success
        """
        pass

    @abstractmethod
    def get_test_result(self, task_name: str, history: List[str]) -> Tuple[bool, Optional[str]]:
        """Get the test result for a specific node in the cache.

        Args:
            task_name: The name of the task
            history: The list of tool calls leading to the node

        Returns:
            Tuple of (found, test_result)
        """
        pass

    @abstractmethod
    def unref(self, env_id: str, task_name: Optional[str] = None) -> None:
        """Unreference an environment ID from the cache.

        Args:
            env_id: The environment ID to unreference
            task_name: The name of the task (optional)
        """
        pass

"""AsyncTVCacheClient for interacting with the TVCache HTTP server."""

import httpx
from typing import Optional, Tuple, Dict, Any, List


class AsyncTVCacheClient:
    """Async client for interacting with the TVCache HTTP server."""

    def __init__(self, base_url: str = "http://localhost:8001"):
        """Initialize the async TVCache client.

        Args:
            base_url: The base URL of the TVCache server.
        """
        self.base_url = base_url.rstrip('/')
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the httpx async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient()
        return self._client

    async def close(self):
        """Close the httpx client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def exact_match(self, task_name: str, tool_calls: List[str]) -> bool:
        """Check if there's an exact match for the given task and tool calls.

        Args:
            task_name: The name of the task.
            tool_calls: The list of tool calls to match against.

        Returns:
            True if an exact match exists, False otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/get",
                params={"task_name": task_name, "tool_calls": tool_calls}
            )
            return response.status_code == 200 and response.json().get("found", False)
        except httpx.HTTPError:
            return False

    async def get(self, task_name: str, tool_calls: List[str]) -> Tuple[Optional[str], Optional[Any], Optional[float]]:
        """Get the environment ID, value, and execution time for an exact match.

        Args:
            task_name: The name of the task.
            tool_calls: The list of tool calls to match against.

        Returns:
            A tuple of (env_id, value, tool_exec_time). All are None if no exact match found.

        Raises:
            ValueError: If no exact match is found.
        """
        client = await self._get_client()
        response = await client.get(
            f"{self.base_url}/get",
            params={"task_name": task_name, "tool_calls": tool_calls}
        )
        response.raise_for_status()
        data = response.json()

        # if not data.get("found"):
        #     raise ValueError(f"No exact match found for task '{task_name}'")

        return data["env_id"], data["value"], data.get("tool_exec_time", 0)

    async def intel_prefix_match(self, task_name: str, tool_calls: List[Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Check for a prefix match and return environment ID and history if found.

        Args:
            task_name: The name of the task.
            tool_calls: The list of tool calls to match against.

        Returns:
            A tuple of (env_id, serialized_history, suffix). All are None if no prefix match found.
        """
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/intel_prefix_match",
                json={"task_name": task_name, "tool_calls": tool_calls}
            )
            response.raise_for_status()
            data = response.json()

            if data.get("found"):
                return data.get("env_id"), data.get("history"), data.get("suffix")
            else:
                return None, [], []
        except httpx.HTTPError:
            return None, [], []

    async def prefix_match(self, task_name: str, tool_calls: List[Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Check for a prefix match and return environment ID and history if found.

        Args:
            task_name: The name of the task.
            tool_calls: The list of tool calls to match against.

        Returns:
            A tuple of (env_id, serialized_history). Both are None if no prefix match found.
        """
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/prefix_match",
                json={"task_name": task_name, "tool_calls": tool_calls}
            )
            response.raise_for_status()
            data = response.json()

            if data.get("found"):
                return data.get("env_id"), data.get("history")
            else:
                return None, []
        except httpx.HTTPError:
            return None, []

    async def mark_stateless(self, task_name: str, history: List[str], env_id: str) -> bool:
        """Update the state of a cached environment.

        Args:
            task_name: The name of the task.
            history: The serialized history list.
            env_id: The environment ID.

        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/mark_stateless",
                json={"task_name": task_name, "history": history, "env_id": env_id}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("success", False)
        except httpx.HTTPError:
            print(f"State update request failed for {task_name} with env_id {env_id} and history {history}")
            return False

    async def put(self, task_name: str, history: List[str], env_id: str, values: List[str] = None, tool_exec_times: List[float] = None, start_idx: int = 0) -> List[str]:
        """Store the tool call history for the given task.

        Args:
            task_name: The name of the task.
            history: The list of tool calls (history).
            env_id: The environment ID.
            values: List of values for each tool call (optional).
            tool_exec_times: List of execution times for each tool call (optional).
            start_idx: Starting index for values/tool_exec_times arrays (optional, default 0).

        Returns:
            List of environment IDs that were removed due to cache budget constraints.
        """
        client = await self._get_client()
        response = await client.put(
            f"{self.base_url}/put",
            json={
                "task_name": task_name,
                "history": history,
                "env_id": env_id,
                "values": values,
                "tool_exec_times": tool_exec_times,
                "start_idx": start_idx
            }
        )
        response.raise_for_status()
        data = response.json()
        return data.get("removed_env_ids", [])

    async def remove(self, task_name: str, history: List[str]) -> bool:
        """Remove a specific tool call history from the cache.

        Args:
            task_name: The name of the task.
            history: The list of tool calls (history) to remove.

        Returns:
            True if the removal was successful, False otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.delete(
                f"{self.base_url}/remove",
                json={"task_name": task_name, "history": history}
            )
            response.raise_for_status()
            return response.json().get("success", False)
        except httpx.HTTPError:
            return False

    async def can_extend(self, task_name: str, history: List[str], suffix: List[str]) -> Tuple[bool, Optional[List[str]]]:
        """Check if a node can be extended with a suffix and mark it as consumed.

        This method checks if a node at the given history path can be extended
        with a suffix. If the node hasn't been consumed, it marks it as consumed
        and returns the provided suffix. If already consumed, it returns the
        existing suffix.

        Args:
            task_name: The name of the task.
            history: The list of tool calls leading to the node.
            suffix: The suffix to potentially extend with.

        Returns:
            A tuple of (can_extend, suffix) where:
            - can_extend: True if the node can be extended (wasn't consumed), False otherwise.
            - suffix: The suffix (either the new one if can_extend=True, or existing one if can_extend=False).
        """
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/can_extend",
                json={"task_name": task_name, "history": history, "suffix": suffix}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("can_extend", False), data.get("suffix")
        except httpx.HTTPError:
            return False, None

    async def should_fork(self, task_name: str, history: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if a node should be forked based on its consumed status.

        This method checks if a node at the given history path should be forked
        (i.e., if it has been consumed and should_fork flag is set).

        Args:
            task_name: The name of the task.
            history: The list of tool calls leading to the node.

        Returns:
            A tuple of (should_fork, env_id) where:
            - should_fork: True if the node should be forked, False otherwise.
            - env_id: The environment ID if should_fork is True, None otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/should_fork",
                json={"task_name": task_name, "history": history}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("should_fork", False), data.get("env_id")
        except httpx.HTTPError:
            return False, None

    async def get_hot_nodes(self, k: int = 1) -> Dict[str, List[List[str]]]:
        """Get hot nodes (nodes with more than k children) from the cache.

        Args:
            k: The threshold for number of children (default: 1).

        Returns:
            Dictionary mapping task names to lists of paths (where each path is a list of tool calls).
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/get_hot_nodes",
                params={"k": k}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("hot_nodes", {})
        except httpx.HTTPError:
            return {}

    async def check_env_marked(self, env_id: str) -> bool:
        """Check if an environment ID is marked as present in the cache.

        Args:
            env_id: The environment ID to check.

        Returns:
            True if the environment ID is marked as present, False otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/check_env_marked",
                params={"env_id": env_id}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("marked", False)
        except httpx.HTTPError:
            return False

    async def store_test_result(self, task_name: str, history: List[str], test_result: str) -> bool:
        """Store a test result for a specific node in the cache.

        Args:
            task_name: The name of the task.
            history: The list of tool calls leading to the node.
            test_result: The test result string to store.

        Returns:
            True if the storage was successful, False otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/store_test_result",
                json={"task_name": task_name, "history": history, "test_result": test_result}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("success", False)
        except httpx.HTTPError:
            return False

    async def get_test_result(self, task_name: str, tool_calls: List[str]) -> Tuple[bool, Optional[str]]:
        """Get the test result for a specific node in the cache.

        Args:
            task_name: The name of the task.
            tool_calls: The list of tool calls leading to the node.

        Returns:
            A tuple of (found, test_result) where:
            - found: True if the node was found, False otherwise.
            - test_result: The test result string if found, None otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/get_test_result",
                params={"task_name": task_name, "tool_calls": tool_calls}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("found", False), data.get("test_result")
        except httpx.HTTPError:
            return False, None

    async def unref(self, env_id: str, task_name: Optional[str] = None) -> bool:
        """Unreference an environment ID from the cache.

        Args:
            env_id: The environment ID to unreference.
            task_name: The name of the task (optional).

        Returns:
            True if the operation was successful, False otherwise.
        """
        try:
            payload = {"env_id": env_id}
            if task_name is not None:
                payload["task_name"] = task_name

            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/unref",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data.get("success", False)
        except httpx.HTTPError:
            return False

    async def get_all_envs(self, task_name: str) -> List[str]:
        """Get all environment IDs in the prefix tree for a given task.

        Args:
            task_name: The name of the task.

        Returns:
            List of all environment IDs present in the task's prefix tree.
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/get_all_envs",
                params={"task_name": task_name}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("env_ids", [])
        except httpx.HTTPError:
            return []

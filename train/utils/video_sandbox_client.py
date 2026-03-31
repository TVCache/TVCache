import httpx
import json
from typing import Dict, Any


class SandboxClient:
    """Simple client for interacting with the sandbox server."""

    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize the sandbox client.

        Args:
            base_url: Base URL of the sandbox server (default: http://localhost:5000)
        """
        self.base_url = base_url.rstrip('/')
        self.sandbox_id = None

    async def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request to the server."""
        url = f"{self.base_url}/{endpoint}"
        # Set timeout to 5 minutes for long-running operations like preprocess
        timeout = httpx.Timeout(300.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=data)
            if response.status_code != 200:
                print(f'Response status is {response.status_code} for output = {response.json()}')
            # response.raise_for_status()
            return response.json()

    async def start_sandbox(self, sandbox_id: str) -> Dict[str, Any]:
        """Start a new sandbox."""
        print(f'[SANDBOX]: Starting: {sandbox_id}')
        result = await self._post('start', {'sandbox_id': sandbox_id})
        self.sandbox_id = sandbox_id
        return result

    async def stop_sandbox(self, sandbox_id: str) -> Dict[str, Any]:
        """Stop and remove a sandbox."""

        print(f'[SANDBOX]: Stopping: {sandbox_id}')
        return await self._post('stop', {'sandbox_id': sandbox_id})

    async def execute(self, function_name: str, argument: str = '') -> Dict[str, Any]:
        """
        Execute a command in the sandbox.

        Args:
            function_name: Name of the function to execute (e.g., 'load_video', 'preprocess',
                          'object_memory_querying', 'segment_localization', 'caption_retrieval',
                          'visual_question_answering')
            argument: Argument for the function as a string
                     - For load_video_into_sandbox: video filename (e.g., 'tea.mp4')
                     - For object_memory_querying: question string
                     - For segment_localization: description string
                     - For caption_retrieval: tuple string (e.g., '(0, 5)')
                     - For visual_question_answering: tuple string (e.g., '("What is happening?", 3)')
                     - For preprocess: empty string or omit

        Returns:
            Response from the server
        """
        if not self.sandbox_id:
            raise ValueError("No sandbox started. Call start_sandbox() first.")

        return await self._post('execute', {
            'sandbox_id': self.sandbox_id,
            'command': function_name,
            'argument': argument
        })
    

    async def fork(self) -> Dict[str, str]:
        print(f'[SANDBOX]: Forking: {self.sandbox_id}')
        if not self.sandbox_id:
            raise ValueError("No sandbox started. Call start_sandbox() first.")

        return await self._post('fork', {
            'sandbox_id': self.sandbox_id
        })
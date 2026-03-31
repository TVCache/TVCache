from typing import Dict, List, Optional
from tinker import SamplingClient
from tool_schema import Response
from utils.video_sandbox_client import SandboxClient
from tinker_cookbook.renderers import Renderer
from threading import Lock
import json
import time


class SharedToolCache:
    """Thread-safe shared cache for tool execution results across agent loops."""

    def __init__(self):
        self._cache: Dict[str, str] = {}
        self._lock = Lock()

    def _make_key(self, function_name: str, argument: str) -> str:
        return json.dumps({"function_name": function_name, "argument": argument})

    def get(self, function_name: str, argument: str) -> Optional[str]:
        key = self._make_key(function_name, argument)
        with self._lock:
            return self._cache.get(key, None)

    def put(self, function_name: str, argument: str, result: str):
        key = self._make_key(function_name, argument)
        with self._lock:
            self._cache[key] = result

    def has(self, function_name: str, argument: str) -> bool:
        key = self._make_key(function_name, argument)
        with self._lock:
            return key in self._cache


class CachedVideoAgentLoop:

    sampler_lock: Lock = Lock()

    def __init__(self,
                 training_data_point: Dict[str, str|int],
                 sampling_client: SamplingClient,
                 num_turns: int,
                 renderer: Renderer,
                 shared_cache: SharedToolCache,
                 sandbox_base_url: str = "http://localhost:5000"):

        self.q = training_data_point['question']
        self.answer = training_data_point['answer']

        self.messages = [{"role": "user", "content": self.q}]
        self.sampling_client = sampling_client
        self.num_turns = num_turns
        self.renderer = renderer
        self.final_answer = None
        self.invalid_parse = False
        self.shared_cache = shared_cache

        # Create sandbox client
        self.sandbox_client = SandboxClient(base_url=sandbox_base_url)
        self.sandbox_id = None
        self.log_file_path = None

    async def start_sandbox(self, sandbox_id: str):
        self.sandbox_id = sandbox_id
        self.log_file_path = f'./rollouts/{sandbox_id}.log'
        return await self.sandbox_client.start_sandbox(sandbox_id)

    async def stop_sandbox(self):
        if self.sandbox_id:
            return await self.sandbox_client.stop_sandbox(self.sandbox_id)

    def log(self, log_line: str):
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_line)
            log_file.write('\n')

    async def run(self, sampling_params) -> tuple[List[int], List[float]]:
        """
        Run the agent loop for multiple turns.

        Args:
            sampling_params: Parameters for sampling from the model

        Returns:
            all_tokens: List of all tokens (prompt + completions) across all turns
            all_logprobs: List of logprobs with masking (0.0 for prompt/tool messages, actual logprobs for assistant messages)
        """
        all_tokens = []
        all_logprobs = []
        advantage_mask = []

        for turn in range(self.num_turns):
            prev_len = len(all_tokens)

            model_input = self.renderer.build_generation_prompt(self.messages)
            prompt_tokens = model_input.to_ints()

            self.log(f'=============Starting turn {turn}=============')
            for message in self.messages:
                self.log(f'Prompt messages: {message["content"]}')

            # Sample from the model (async)
            st = time.perf_counter()
            self.log(f'Waiting for sampler to return data in rollout {self.sandbox_id}')

            sample_result = await self.sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params
            )

            et = time.perf_counter()
            self.log(f'[GEN-TIME]: Time taken to generate: {et - st} seconds')

            sampled_tokens = sample_result.sequences[0].tokens
            sampled_logprobs = sample_result.sequences[0].logprobs
            assert sampled_logprobs is not None, "Logprobs must be enabled in sampling"

            # Calculate new tokens added in this turn (tool responses from previous turn)
            new_prompt_tokens = prompt_tokens[prev_len - 1:]
            all_tokens = prompt_tokens + sampled_tokens

            # Update logprobs: mask prompt/tool tokens (0.0), keep assistant token logprobs
            if turn == 0:
                all_logprobs = [0.0] * (len(prompt_tokens) - 1) + list(sampled_logprobs)
                advantage_mask = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(sampled_logprobs)
            else:
                all_logprobs += [0.0] * (len(new_prompt_tokens) - 1) + list(sampled_logprobs)
                advantage_mask += [0.0] * (len(new_prompt_tokens) - 1) + [1.0] * len(sampled_logprobs)

            parsed_message, _ = self.renderer.parse_response(sampled_tokens)

            self.log(f'Parsed message {parsed_message["content"]}')
            self.log(f"Log probs must always be 1 less than tokens but seeing token len = {len(all_tokens)} logprobs len = {len(all_logprobs)}")

            assert len(all_tokens) == len(all_logprobs) + 1, f"Log probs must always be 1 less than tokens but seeing token len = {len(all_tokens)} logprobs len = {len(all_logprobs)}"

            try:
                response = Response.model_validate_json(parsed_message["content"])
            except Exception as e:
                self.log(f'[PARSE-ERROR]: Failed to parse response as Response schema: {e}')
                self.log(f'[PARSE-ERROR]: Raw content: {parsed_message["content"]}')
                self.invalid_parse = True
                break

            self.messages.append(parsed_message)

            if response.final_answer is not None:
                self.final_answer = response.final_answer
                break

            for action in response.actions:
                try:

                    function_name = action.tool
                    argument = action.inputs

                    self.log(f'Calling tool {function_name} with arguments {argument}')

                    # Check cache first
                    cached_result = self.shared_cache.get(function_name, argument)

                    if cached_result is not None:
                        self.log(f'[CACHE-HIT]: {function_name} with {argument}')
                        tool_result = f"Result of calling {function_name} with {argument} as argument is {cached_result}"
                    else:
                        self.log(f'[CACHE-MISS]: {function_name} with {argument}')

                        st = time.perf_counter()
                        result = await self.sandbox_client.execute(function_name, argument)
                        et = time.perf_counter()

                        self.log(f'[TOOL-TIME]: Time taken to call {function_name} with {argument}: {et - st} seconds')

                        if 'result' in result:
                            self.shared_cache.put(function_name, argument, result['result'])
                            tool_result = f"Result of calling {function_name} with {argument} as argument is {result['result']}"
                        else:
                            tool_result = f"Failed to execute {function_name} with {argument}. There was an error calling the {function_name}"

                except Exception as e:
                    import traceback
                    tool_result = f"Tool: {action.tool} failed to execute, there was an error."
                    self.log(traceback.format_exc())


                self.messages.append({"role": "tool", "content": tool_result})

        return all_tokens, all_logprobs, advantage_mask

    def get_reward(self) -> float:
        if self.invalid_parse:
            return -2

        if self.final_answer == self.answer:
            return 1

        return 0

from tvclient.tools import ToolCallEnv, ToolCall
from tvclient.fork.dict_bank import SimpleDictBank
from tvclient.tools.async_semantic_stateful_executor import AsyncSemanticStatefulExecutor
import uuid
from utils.video_sandbox_client import SandboxClient
from typing import Dict, List, Optional
from tinker import SamplingClient, types, SampleResponse
from tool_schema import Response
from utils.video_sandbox_client import SandboxClient
from tinker_cookbook.renderers import Renderer
from tinker.types.sampling_params import SamplingParams
from threading import Lock
import time
import asyncio

class VideoToolCall(ToolCall):

    def __init__(self, function_name: str = None, argument: str = ''):
        self.function_name = function_name
        self.argument = argument

    def to_dict(self) -> dict:
        return {
            "function_name": self.function_name,
            "argument": self.argument
        }

    @staticmethod
    def from_dict(data: dict) -> 'VideoToolCall':
        return VideoToolCall(
            function_name=data.get("function_name"),
            argument=data.get("argument", '')
        )
    
    def will_mutate_state(self) -> bool:
        return self.function_name == 'preprocess' or self.function_name == 'load_video_into_sandbox'


class VideoSandboxEnv(ToolCallEnv):

    def __init__(self, env_id: Optional[str] = None, task_name: str = "default_task"):

        self.task_name = task_name
        self.sandbox_client = SandboxClient(base_url="http://localhost:5000")
        self.state = {}
        self.started = False

        if env_id == None:
            self.env_id =  f"{self.task_name}_{uuid.uuid4().hex}"
        else:
            self.env_id = env_id
            self.sandbox_client.sandbox_id = env_id
            self.started = True


    async def stop(self, **kwargs) -> None:

        if not self.env_id:
            raise ValueError("No env_id to stop")

        print(f"Stopping sandbox environment with id {self.env_id}")
        try:
            result = await self.sandbox_client.stop_sandbox(self.env_id)
            print(f"Stopped sandbox: {result}")
        except Exception as e:
            print(f'Failed to stop sandbox: {self.env_id}')
        return result

    async def execute(self, tool_call: VideoToolCall, **kwargs):
        if not self.started:
            await self.sandbox_client.start_sandbox(self.env_id)

        result = await self.sandbox_client.execute(tool_call.function_name, tool_call.argument)
        try:
            return result['result']
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f'Failed to execute {tool_call.function_name} with {tool_call.argument}. There was an error calling the {tool_call.function_name}'

    async def fork(self, **kwargs) -> 'VideoSandboxEnv':
        if not self.started:
            await self.sandbox_client.start_sandbox(self.env_id)
        
        forked_response = await self.sandbox_client.fork()
        assert "sandbox_id" in forked_response
        forked_env = VideoSandboxEnv(env_id=forked_response["sandbox_id"], task_name=self.task_name)

        return forked_env

    async def get_state(self, **kwargs):
        return self.state

    def get_id(self, **kwargs) -> str:
        return self.env_id

    async def test(self) -> str:
        raise NotImplementedError("Not implemented for this")

    async def hash(self) -> str:
        raise NotImplementedError("Hash not implemented")



class VideoAgentLoop:

    sampler_lock: Lock = Lock()

    def __init__(self, 
                 training_data_point: Dict[str, str|int], 
                 sampling_client: SamplingClient, 
                 num_turns: int, 
                 renderer: Renderer, 
                 sandbox_base_url: str = "http://localhost:5000",
                 task_id: str = ""):
        
        self.q = training_data_point['question']
        self.answer = training_data_point['answer']

        self.messages = [{"role": "user", "content": self.q}]
        self.sampling_client = sampling_client
        self.num_turns = num_turns
        self.renderer = renderer
        self.final_answer = None
        self.invalid_parse = False

        # Create sandbox client
        self.log_file_path = None
        self.executor = AsyncSemanticStatefulExecutor(tool_call_class=VideoToolCall, tool_call_env_class=VideoSandboxEnv, task_id=task_id)
        self.executed_tool_calls: List[VideoToolCall] = []
        self.sandbox_id = task_id

        # set fork generator for the executor
        self.fork_generator = SimpleDictBank(env_class=VideoSandboxEnv)
        self.executor.set_fork_bank(self.fork_generator)
        self.warmup_task_ids = training_data_point.get("next_batch", None)
        self.rollout_count = training_data_point.get("rollout_count", 0)

    async def start_sandbox(self, sandbox_id: str):
        self.log_file_path = f'./rollouts/{sandbox_id}.log'
        self.executor.set_rollout_id(sandbox_id)

    async def stop_sandbox(self):
        await self.fork_generator.withdraw(task_name=self.sandbox_id)
    
    def log(self, log_line: str):
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_line)
            log_file.write('\n')
    
    async def sample_response(self, max_retries, model_input, sampling_params: SamplingParams) -> SampleResponse:
        backoff_seconds = 1

        for attempt in range(max_retries):
            try:
                sample_result = await self.sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params
                )
                return sample_result
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {backoff_seconds} second(s)...")
                    await asyncio.sleep(backoff_seconds)
                else:
                    print(f"All {max_retries} attempts failed.")
                    raise

    async def run(self, sampling_params: SamplingParams) -> tuple[List[int], List[float]]:
        """
        Run the agent loop for multiple turns.

        Args:
            renderer: The chat renderer for building prompts and parsing responses
            sampling_params: Parameters for sampling from the model

        Returns:
            all_tokens: List of all tokens (prompt + completions) across all turns
            all_logprobs: List of logprobs with masking (0.0 for prompt/tool messages, actual logprobs for assistant messages)
        """
        all_tokens = []
        all_logprobs = []
        advantage_mask = []
        loop = asyncio.get_event_loop()

        started_tasks = []

        if self.warmup_task_ids != None:
            for next_task_name in self.warmup_task_ids:
                bt = asyncio.create_task(
                    self.fork_generator.deposit(task_name=next_task_name, rollout_count=self.rollout_count)
                )
                started_tasks.append(bt)

        for turn in range(self.num_turns):
            prev_len = len(all_tokens)

            model_input = self.renderer.build_generation_prompt(self.messages)
            prompt_tokens = model_input.to_ints()

            self.log(f'=============Starting turn {turn}=============')
            for message in self.messages:
                self.log(f'Prompt messages: {message["content"]}')

            # Sample from the model (async)
            if len(prompt_tokens) + sampling_params.max_tokens >= 32768:
                self.log(f'Breaking because of prompt length: {len(prompt_tokens) + sampling_params.max_tokens}')
                break

            st = time.perf_counter()
            self.log(f'Waiting for sampler to return data in rollour {self.sandbox_id}')


            sample_result = await self.sample_response(5, model_input, sampling_params)

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

                    # Execute the tool
                    self.log(f'Calling tool {function_name} with arguments {argument}')

                    st = time.perf_counter()
                    
                    
                    tool_call = VideoToolCall(function_name=function_name, argument=argument)
                    self.executed_tool_calls.append(tool_call)
                    
                    # result = await loop.run_in_executor(None, self.executor.execute, tool_calls)
                    result = await self.executor.execute(self.executed_tool_calls)
                    et = time.perf_counter()

                    self.log(f'[TOOL-TIME]: Time taken to call {function_name} with {argument}: {et - st} seconds')

                    tool_result = f"Result of calling {function_name} with {argument} as argument is: {result}"

                except Exception as e:
                    import traceback
                    tool_result = f"Tool: {action.tool} failed to execute with arguemt {argument}, there was an error."
                    self.log(traceback.format_exc())


                self.messages.append({"role": "tool", "content": tool_result})

        await self.executor.close()

        st = time.perf_counter()
        await asyncio.gather(*started_tasks)
        et = time.perf_counter()
        
        self.log(f'Time taken to gather started {len(started_tasks)} tasks = {et - st}')

        return all_tokens, all_logprobs, advantage_mask

    def get_reward(self) -> float:
        if self.invalid_parse:
            return -2
        
        if self.final_answer == self.answer:
            return 1
        
        return 0
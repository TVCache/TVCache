from typing import Type
from tvclient.utils.async_tvcache_client import AsyncTVCacheClient
from tvclient.tools.tool_call_env import ToolCallEnv, ToolCall
from tvclient.fork.abstract_bank import AbstractForkGenerator
from typing import List, Tuple
import time
import logging
import json
from threading import Thread
import asyncio


FORK_THRESHOLD = 0 # seconds

class TestToolCall(ToolCall):
    def __init__(self, command: str):
        self.command = command

    def to_dict(self) -> dict:
        return {"TV_CACHE_TOOL_TYPE": "TESTING TOOL"}

    @staticmethod
    def from_dict(data: dict) -> 'TestToolCall':
        return TestToolCall(command=data["command"])

class AsyncSemanticStatefulExecutor:
    """Executes tool calls."""

    def __init__(self, tool_call_env_class: Type[ToolCallEnv], tool_call_class: Type[ToolCall], task_id):
        self.client = AsyncTVCacheClient()
        self.tool_call_env_class = tool_call_env_class
        self.tool_call_class = tool_call_class
        self.tool_call_env_obj = None
        self.task_name = task_id
        self.executed_commands = 0
        self.rollout_id = str(time.time())
        self.rollout_environment: ToolCallEnv = None
        self.total_calls = 0
        self.total_executions = 0
        self.logger = logging.getLogger(__name__)
        self.fork_generator = None


    async def close(self):
        await self.client.close()
        if self.rollout_environment:
            await self.rollout_environment.stop()

    def set_rollout_id(self, rollout_id):
        self.rollout_id = rollout_id

        self.logger = logging.getLogger(self.rollout_id)
        self.logger.setLevel(logging.DEBUG)
        
        file_handler = logging.FileHandler(f'./rollouts/{rollout_id}.log')
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
    

    def set_fork_bank(self, fork_bank: AbstractForkGenerator):
        self.fork_generator = fork_bank


    def _serialize_tool_calls(self, tool_calls: List[ToolCall]) -> List[str]:
        return [json.dumps(c.to_dict()) for c in tool_calls]
    

    def _handle_removed_envs(self, removed_envs: List[str]):
        try:
            for env_id in removed_envs:
                self.logger.debug(f'Deleting The environment {env_id} of task {self.task_name}')
                env_obj = self.tool_call_env_class(env_id=env_id, task_name=self.task_name)
                asyncio.run(env_obj.stop())
        except Exception as e:
            self.logger.debug(f'Failed to Remove environment {env_id} of task {self.task_name} due to {e}')


    def _get_serialized_stateful_chain(self, tool_commands: List[ToolCall]) -> List[str]:
        stateful_chain = []
        for tool_cmd in tool_commands[: -1]:
            if tool_cmd.will_mutate_state():
                stateful_chain.append(tool_cmd)
        
        stateful_chain.append(tool_commands[-1])

        serialized_stateful_chain = self._serialize_tool_calls(stateful_chain)
        return serialized_stateful_chain

    async def _maybe_put_to_cache(self, history: List[str], values: List[str], exec_times: List[float], env: ToolCallEnv):
        env_id, value, _ = await self.client.get(self.task_name, history)
        self.logger.debug(f'Got output form check in maybe_put_to_cache: {env_id}, {value}')
        
        if env != None:
            if env_id == None:
                fst = time.perf_counter()
                to_store = await env.fork()
                fet = time.perf_counter()

                removed_envs = await self.client.put(self.task_name, history, to_store.get_id(), values, exec_times, len(history) - len(values))

                self.logger.debug(f'Stored {history} in cache after forking which took {fst - fet} seconds. Parent={env.get_id()}, child = {to_store.get_id()}')

                # assert len(removed_envs) == 0
                if len(removed_envs) > 0:
                    remover_thread = Thread(target=self._handle_removed_envs, kwargs={'removed_envs': removed_envs})
                    remover_thread.start()


            else:
                self.logger.debug(f'Skipping storing env in cache because there is already environment')
        
        else:
            if value == None:
                removed_envs = await self.client.put(self.task_name, history, None, values, exec_times, len(history) - len(values))
                
                # assert len(removed_envs) == 0
                if len(removed_envs) > 0:
                    remover_thread = Thread(target=self._handle_removed_envs, kwargs={'removed_envs': removed_envs})
                    remover_thread.start()
            else:
                self.logger.debug(f'SKipping storing key and value in the cache')


    async def _execute_commands(self, tool_calls: List[ToolCall], start_idx: int, env: ToolCallEnv) -> Tuple[List[str], List[float], str]:
        values = []
        execution_times = []
        test_result = None

        for idx in range(start_idx, len(tool_calls)):
            
            tool_call = tool_calls[idx]
            self.total_executions += 1

            if idx < len(tool_calls) - 1:
                if not tool_call.will_mutate_state():
                    self.logger.debug(f'[SKIPPING]: {tool_call.to_dict()}')
                    values.append('SKIP')
                    execution_times.append(-1)
                    self.executed_commands += 1
                    continue

            if isinstance(tool_call, TestToolCall):
                assert tool_call == tool_calls[-1]
                test_result = env.test()
                self.logger.debug(f'[TEST EXEC]: Executed test tool call {self._serialize_tool_calls(tool_calls)} for rollout {self.rollout_id} with result {test_result}')

            else:
                self.executed_commands += 1
                st = time.perf_counter()
                last_state = await env.execute(tool_call)
                et = time.perf_counter()
                values.append(last_state)
                execution_times.append((et - st))
                
                self.logger.debug(f'[TOOL EXEC]: Executed tool call {self._serialize_tool_calls([tool_call])} in {et - st} seconds for rollout {self.rollout_id}')

        # logger.debug(f'[TOTAL EXEC_TIME]: {sum(execution_times)} for {self.rollout_id}')
        return values, execution_times, test_result

    async def _execute_and_put(self, commands: List[ToolCall], start_idx: int, env: ToolCallEnv) -> str:
        """Executes the commands in the tool calling environment and updates the prefix tree"""
        values, execution_times, test_result = await self._execute_commands(commands, start_idx, env)

        if test_result != None:
            history = self._serialize_tool_calls(commands)
            self.logger.debug(f'Storing test result {test_result} for history: {history}')
            await self.client.store_test_result(self.task_name, history[: len(history) - 1], test_result)
            self.logger.debug(f'[ENV]: Deleting in last step for test tool call environment {env.get_id()} of task {self.task_name} for rollout {self.rollout_id}')
            
            stop_thread = Thread(target=env.stop)
            stop_thread.start()
            
            if env is self.rollout_environment:
                self.rollout_environment = None
        
        else:
            st = time.perf_counter()

            assert len(execution_times) == len(commands) - start_idx, f"{execution_times} and start index {start_idx}" # this is to make sure that execute is called for each command

            tool_call_time = execution_times[-1]

            if tool_call_time > (1.0) * FORK_THRESHOLD and commands[-1].will_mutate_state():
                # We need to store this environment in the prefix tree

                stateful_chain = self._get_serialized_stateful_chain(commands)
                stateful_values = [v for v in values if v != 'SKIP']
                stateful_exec_times = [t for t in execution_times if t != -1]

                if not stateful_values[-1].startswith('Failed to execute'):
                    self.logger.debug(f'[ENV]: Maybe Storing environment after long tool call of {tool_call_time} seconds for commands: {commands}')
                    await self._maybe_put_to_cache(stateful_chain, stateful_values, stateful_exec_times, env)

            else:
                stateful_chain = self._get_serialized_stateful_chain(commands)
                stateful_values = [v for v in values if v != 'SKIP']
                stateful_exec_times = [t for t in execution_times if t != -1]
                # we do not need to store this environment in the prefix tree

                if not stateful_values[-1].startswith('Failed to execute'):
                    await self._maybe_put_to_cache(stateful_chain, stateful_values, stateful_exec_times, None)

            et = time.perf_counter()
            
            self.logger.debug(f'Time taken to store value in cache: {et - st} seconds')
            # No need to wait because the env pruning happens in the background

        if test_result != None:
            return test_result
        return values[-1]
        

    async def execute(self, tool_commands: List[ToolCall]):
        """Executes the last tool call if the cache doesn't have a value associated with `current_tool_calls` prefix. The function also populates the cache if it executes the tool call."""
        current_tool_calls = self._serialize_tool_calls(tool_commands)
        self.total_calls += 1
 
        stateful_serialized_chain = self._get_serialized_stateful_chain(tool_commands)

        if await self.client.exact_match(self.task_name, stateful_serialized_chain):
            cst = time.perf_counter()
            env_id, value, tool_exec_time = await self.client.get(self.task_name, stateful_serialized_chain)
            est = time.perf_counter()
            self.logger.debug(f"CACHE HIT, type 1 for task id {self.task_name} with tool calls : {current_tool_calls} in {(est - cst)} seconds and saved tool call time {tool_exec_time} and depth {len(current_tool_calls)} for rollout {self.rollout_id}")
            return value

        else:

            env_id, prefix_tool_calls = await self.client.prefix_match(self.task_name, stateful_serialized_chain)
            assert prefix_tool_calls != None

            if len(prefix_tool_calls) == len(stateful_serialized_chain):
                self.logger.debug(f'CACHE HIT, type 2 after miss for: {self.rollout_id}')
                env_id, value, _ = await self.client.get(self.task_name, stateful_serialized_chain)
                return value
            
            self.logger.debug('CACHE MISS, for task id {} need to execute tool calls: {} for rollout {}'.format(self.task_name, current_tool_calls, self.rollout_id))

            if env_id == None:
                # Rollout always had cache hits so far, no environment stored
                if self.rollout_environment == None:

                    cst = time.perf_counter()
                    # TODO: Use fork bank here
                    bank_env_id = self.fork_generator.get_forked_env(task_name=self.task_name, parent_env_id="root")
                    if bank_env_id != None:
                        self.logger.debug(f'Found Bank root environment for task {self.task_name} and rollout {self.rollout_id}')
                        env_obj = self.tool_call_env_class(task_name=bank_env_id)
                    else:
                        env_obj = self.tool_call_env_class(task_name=self.task_name)
                    self.rollout_environment = env_obj
                    
                    est = time.perf_counter()
                    
                    return await self._execute_and_put(tool_commands, 0, env_obj)

                # Execute in the current rollout environment
                else:
                    
                    # logger.debug(f'[ENV]: Continuing with current rollout environment: {self.rollout_environment.get_id()} for rollout {self.rollout_id} and executing commands: {self._serialize_tool_calls(tool_commands[self.executed_commands:])} so negCACHEHIT of {len(tool_commands) - self.executed_commands - 1}')

                    return await self._execute_and_put(tool_commands, self.executed_commands, self.rollout_environment)
                
            else:
                parent_env_id = env_id
                self.logger.debug(f'[ENV]: Found cached env: {env_id} for rollout {self.rollout_id} and commands: {prefix_tool_calls}')

                if len(prefix_tool_calls) > self.executed_commands:
                    self.executed_commands = len(prefix_tool_calls)

                    # TODO: check fork bank
                    start_time = time.perf_counter()
                    bank_env_id = self.fork_generator.get_forked_env(task_name=self.task_name, parent_env_id=parent_env_id)
                    if bank_env_id != None:
                        self.logger.debug(f'Found fork bank entry for {self.task_name} and {parent_env_id}')
                        forked_env = self.tool_call_env_class(env_id=bank_env_id, task_name=self.task_name)
                    else:
                        parent_env = self.tool_call_env_class(env_id=parent_env_id, task_name=self.task_name)
                        self.logger.debug(f'Could not Find fork bank entry for {self.task_name} and {parent_env_id}, forking from scratch')
                        forked_env = await parent_env.fork()

                    end_time = time.perf_counter()

                    duration = end_time - start_time
                    self.logger.debug(f'[ENV]: Extended environment: {forked_env.get_id()} in {duration:.2f} seconds from parent: {parent_env_id} for rollout {self.rollout_id} so negCACHEHIT of {len(tool_commands) - len(prefix_tool_calls) - 1} with prefix {prefix_tool_calls}')

                    await self.client.unref(parent_env_id, self.task_name)
                
                    result = await self._execute_and_put(tool_commands, len(prefix_tool_calls), forked_env)
                    
                    if self.rollout_environment != None:
                        removed_envs = [self.rollout_environment.get_id()]
                        remover_thread = Thread(target=self._handle_removed_envs, kwargs={'removed_envs': removed_envs})
                        remover_thread.start()
                    
                    self.rollout_environment = forked_env
                else:
                    result = await self._execute_and_put(tool_commands, self.executed_commands, self.rollout_environment)
    
                return result


    async def test(self, tool_call_history: List[ToolCall]) -> str:
        found, value = await self.client.get_test_result(self.task_name, self._serialize_tool_calls(tool_call_history))

        if found:
            self.total_calls += 1
            self.logger.debug(f'Found test result in Cache, CACHE HIT for task id {self.task_name}')
            return value
        self.logger.debug(f'No test result in Cache, CACHE MISS for task id {self.task_name}')
        test_tool_call = TestToolCall("")
        tool_call_history.append(test_tool_call)
        return await self.execute(tool_call_history)
        
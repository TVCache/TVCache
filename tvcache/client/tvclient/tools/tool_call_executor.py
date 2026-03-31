"""ToolCallExecutor class for TVCache."""

from typing import Type
from tvclient.utils.tvcache_client import TVCacheClient
from tvclient.tools.tool_call_env import ToolCallEnv, ToolCall
from typing import List, Tuple
import time
import logging
import json
from threading import Thread

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TestToolCall(ToolCall):
    def __init__(self, command: str):
        self.command = command

    def to_dict(self) -> dict:
        return {"TV_CACHE_TOOL_TYPE": "TESTING TOOL"}

    @staticmethod
    def from_dict(data: dict) -> 'TestToolCall':
        return TestToolCall(command=data["command"])

class ToolCallExecutor:
    """Executes tool calls."""

    def __init__(self, tool_call_env_class: Type[ToolCallEnv], tool_call_class: Type[ToolCall], task_id):
        self.client = TVCacheClient()
        self.tool_call_env_class = tool_call_env_class
        self.tool_call_class = tool_call_class
        self.tool_call_env_obj = None
        self.task_name = task_id
        self.executed_commands = []

    def _serialize_tool_calls(self, tool_calls: List[ToolCall]) -> List[str]:
        return [json.dumps(c.to_dict()) for c in tool_calls]
    
    def _handle_removed_envs(self, removed_envs: List[str]):
        try:
            for env_id in removed_envs:
                logger.debug(f'Deleting The environment {env_id} of task {self.task_name}')
                env_obj = self.tool_call_env_class(env_id, self.task_name)
                env_obj.stop()
        except Exception as e:
            logger.debug(f'Failed to Remove environment {env_id} of task {self.task_name} due to {e}')

    def _execute_commands(self, tool_calls: List[ToolCall], start_idx: int, env: ToolCallEnv) -> Tuple[List[str], List[float], str]:
        values = []
        execution_times = []
        test_result = None

        for idx in range(start_idx, len(tool_calls)):
            
            tool_call = tool_calls[idx]

            if isinstance(tool_call, TestToolCall):
                assert tool_call == tool_calls[-1]
                test_result = env.test()

            else:
                st = time.perf_counter()
                last_state = env.execute(tool_call)
                et = time.perf_counter()
                values.append(last_state)
                execution_times.append((et - st))
        
        return values, execution_times, test_result

    def _execute_and_put(self, commands: List[ToolCall], start_idx: int, env: ToolCallEnv) -> str:
        """Executes the commands in the tool calling environment and updates the prefix tree"""
        values, execution_times, test_result = self._execute_commands(commands, start_idx, env)

        if test_result != None:
            history = self._serialize_tool_calls(commands)
            logger.debug(f'Storing test result {test_result} for history: {history}')
            self.client.store_test_result(self.task_name, history[: len(history) - 1], test_result)
        
        else:
            removed_envs = self.client.put(self.task_name, self._serialize_tool_calls(commands), env.get_id(), values, execution_times, start_idx)
            if len(removed_envs) > 0:
                remover_thread = Thread(target=self._handle_removed_envs, kwargs={'removed_envs': removed_envs})
                remover_thread.start()
            # No need to wait because the env pruning happens in the background

        if test_result != None:
            return test_result
        return values[-1]
        

    def execute(self, tool_commands: List[ToolCall]):
        """Executes the last tool call if the cache doesn't have a value associated with `current_tool_calls` prefix. The function also populates the cache if it executes the tool call."""
        current_tool_calls = self._serialize_tool_calls(tool_commands)
        tc_task_name = self.task_name

        if self.client.exact_match(tc_task_name, current_tool_calls):
            logger.debug('CACHE HIT, type 1')
            env_id, value, _ = self.client.get(tc_task_name, current_tool_calls)
            return value

        else:
            env_id, prefix_tool_calls = self.client.prefix_match(tc_task_name, current_tool_calls)
            assert prefix_tool_calls != None

            if len(prefix_tool_calls) == len(current_tool_calls):
                env_id, value, _ = self.client.get(tc_task_name, current_tool_calls)
                return value
            
            suffix = current_tool_calls[len(prefix_tool_calls): ]

            # Check if the current rollout can consume the env_id
            can_extend, locked_suffix = self.client.can_extend(task_name=tc_task_name, history=prefix_tool_calls, suffix=suffix)

            if can_extend:
                # execute commands on the env
                logger.debug(f'Extending execution on env_id: {env_id} for commands: {current_tool_calls}')
                env = self.tool_call_env_class(task_name=tc_task_name, env_id=env_id)
                val = self._execute_and_put(tool_commands, len(prefix_tool_calls), env)
                self.client.unref(env_id)
                return val
            else:
                # Someone else is executing the same commands, wait
                logger.debug(f'CACHE MISS, type 2, locked_suffix: {locked_suffix}, my suffix: {suffix}')
                if locked_suffix == suffix:

                    if not isinstance(tool_commands[-1], TestToolCall):
                        while not self.client.exact_match(tc_task_name, current_tool_calls):
                            logger.debug(f'Waiting for other rollout to proceed on {locked_suffix} when my suffix is {suffix}')
                            time.sleep(0.1)
                        env_id, value, _ = self.client.get(tc_task_name, current_tool_calls)
                        logger.debug(f'Done waiting on suffix: {suffix}, found cached value: {value}')
                        return value
                    
                    else:
                        tool_commands = tool_commands[: len(tool_commands) - 1]
                        found, value = self.client.get_test_result(self.task_name, self._serialize_tool_calls(tool_commands))

                        while not found:
                            logger.debug(f'Waiting on test result on history: {current_tool_calls}')
                            found, value = self.client.get_test_result(self.task_name, self._serialize_tool_calls(tool_commands))
                        
                        return value

                
                # Use your own environment and execute 
                else: 
                    should_fork, env_id = self.client.should_fork(tc_task_name, current_tool_calls)
                    if should_fork:
                        env = self.tool_call_env_class(task_name=tc_task_name, env_id=env_id).fork()
                        # Execute commands here and put to cache
                        return self._execute_and_put(tool_commands, len(prefix_tool_calls), env)
                    else:
                        # Execute commands on current env id
                        logger.debug(f'Using rollout environment to execute {current_tool_calls} and previously executed {self.executed_commands}')
                        
                        if self.tool_call_env_obj and not self.client.check_env_marked(self.tool_call_env_obj.get_id()):
                            self.tool_call_env_obj = None
                            self.executed_commands.clear()
                            logger.debug(f'ENV Invalidated, so removing it')

                        if self.tool_call_env_obj == None:
                            self.tool_call_env_obj = self.tool_call_env_class(task_name=tc_task_name)
                            logger.debug(f'Creating rollout environment: {self.tool_call_env_obj.get_id()}')

                        value = self._execute_and_put(tool_commands, len(self.executed_commands), self.tool_call_env_obj)
                        self.executed_commands = [c for c in current_tool_calls]
                        return value
            
    def __del__(self):
        if self.tool_call_env_obj != None:
            if not self.client.check_env_marked(self.tool_call_env_obj.get_id()):
                logger.debug(f'Deleting The created environment {self.tool_call_env_obj.get_id()} was not used by the prefix tree, so deleting it')
                self.tool_call_env_obj.stop()
    
    def erase_history(self, current_tool_calls: List[ToolCall]):
        """Invalidates all the values in the `current_tool_calls` path in the cache if no other sequence in the cache shares the prefix"""
        self.client.remove(self.task_name, self._serialize_tool_calls(current_tool_calls))
    
    def test(self, tool_call_history: List[ToolCall]) -> str:
        found, value = self.client.get_test_result(self.task_name, self._serialize_tool_calls(tool_call_history))

        if found:
            logger.debug(f'Found test result in Cache, CACHE HIT')
            return value
        
        test_tool_call = TestToolCall("")
        tool_call_history.append(test_tool_call)
        return self.execute(tool_call_history)
        
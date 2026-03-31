from typing import Type
from tvclient.utils.tvcache_client import TVCacheClient
from tvclient.tools.tool_call_env import ToolCallEnv, ToolCall
from typing import List, Tuple
import time
import logging
import json
from threading import Thread
from tvclient.fork.bank import ForkGenerator

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

class ImmutableToolCallExecutor:
    """Executes tool calls."""

    def __init__(self, tool_call_env_class: Type[ToolCallEnv], tool_call_class: Type[ToolCall], task_id):
        self.client = TVCacheClient()
        self.tool_call_env_class = tool_call_env_class
        self.tool_call_class = tool_call_class
        self.tool_call_env_obj = None
        self.task_name = task_id
        self.executed_commands = []
        self.rollout_id = None
        self.fork_generator = ForkGenerator(env_class=self.tool_call_env_class)
    
    def set_rollout_id(self, rollout_id):
        self.rollout_id = rollout_id


    def _serialize_tool_calls(self, tool_calls: List[ToolCall]) -> List[str]:
        return [json.dumps(c.to_dict()) for c in tool_calls]
    

    def _handle_removed_envs(self, removed_envs: List[str]):
        try:
            for env_id in removed_envs:
                logger.debug(f'Deleting The environment {env_id} of task {self.task_name}')
                env_obj = self.tool_call_env_class(env_id=env_id, task_name=self.task_name)
                env_obj.stop()
        except Exception as e:
            logger.debug(f'Failed to Remove environment {env_id} of task {self.task_name} due to {e}')


    def check_hash(self, env_id: str, parent_env_id: str) -> bool:
        "calls the client to check if the state hash of the new environment matches that of the parent environment"
        logger.debug(f'[ENV]: Checking hash for env: {env_id} with parent: {parent_env_id}')
        forked_env = self.tool_call_env_class(env_id=env_id, task_name=self.task_name)
        parent_env = self.tool_call_env_class(env_id=parent_env_id, task_name=self.task_name)
        forked_hash = forked_env.hash()
        # parent_hash = parent_env.hash()
        # match = True if forked_hash == parent_hash or forked_hash == "0000" else False
        if forked_hash == "0000":
            match = True
        else:
            match = False
        # logger.debug(f'[ENV]: Hash match result: {match} because forked_hash: {forked_hash}, parent_hash: {parent_hash}')
        logger.debug(f'[ENV]: Hash match result: {match} because forked_hash: {forked_hash}')
        return match


    def async_state_update(self, task_name: str, history: List[str], env_id: str, parent_env_id: str):
        logger.debug(f'[ENV]: Starting async state update for env: {env_id} with parent: {parent_env_id}' + f' for task: {task_name}' + f' with history: {history}')
        def state_updater(task_name: str, history: List[str], env_id: str, parent_env_id: str):
            try:
                is_same = self.check_hash(env_id, parent_env_id)
                if is_same:
                    logger.debug(f'[ENV]: State is SAME as parent, marking stateless')
                    self.client.mark_stateless(env_id=env_id, task_name=task_name, history=history)
                else:
                    logger.debug(f'[ENV]: State is DIFFERENT from parent, marking stateful')
            except Exception as e:
                logger.debug(f'Failed to update state info for environment {env_id} of task {task_name} due to {e}')

        updater_thread = Thread(target=state_updater, args=(task_name, history, env_id, parent_env_id))
        updater_thread.start()
    

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
        
        logger.debug(f'[EXEC_TIME]: {sum(execution_times)} for {self.rollout_id}')
        return values, execution_times, test_result

    def _execute_and_put(self, commands: List[ToolCall], start_idx: int, env: ToolCallEnv) -> str:
        """Executes the commands in the tool calling environment and updates the prefix tree"""
        values, execution_times, test_result = self._execute_commands(commands, start_idx, env)

        if test_result != None:
            history = self._serialize_tool_calls(commands)
            logger.debug(f'Storing test result {test_result} for history: {history}')
            self.client.store_test_result(self.task_name, history[: len(history) - 1], test_result)
            logger.debug(f'[ENV]: Deleting in last step')
            env.stop()
        
        else:
            st = time.perf_counter()
            removed_envs = self.client.put(self.task_name, self._serialize_tool_calls(commands), env.get_id(), values, execution_times, start_idx)
            et = time.perf_counter()
            logger.debug(f'Time taken to store value in cache: {et - st} seconds')
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

        if self.client.exact_match(self.task_name, current_tool_calls):
            cst = time.perf_counter()
            env_id, value, _ = self.client.get(self.task_name, current_tool_calls)
            est = time.perf_counter()
            logger.debug('CACHE HIT, type 1 for task id {} with tool calls : {} in {} seconds and depth of {} for rollout {}'.format(self.task_name, current_tool_calls, (est - cst), len(current_tool_calls), self.rollout_id))
            return value

        else:
            logger.debug('CACHE MISS, for task id {} need to execute tool calls: {} for rollout {}'.format(self.task_name, current_tool_calls, self.rollout_id))
            env_id, prefix_tool_calls = self.client.prefix_match(self.task_name, current_tool_calls)
            assert prefix_tool_calls != None

            suffix = current_tool_calls[len(prefix_tool_calls): ]
            
            if len(prefix_tool_calls) == len(current_tool_calls):
                logger.debug(f'Hit after miss for: {self.rollout_id}')
                env_id, value, _ = self.client.get(self.task_name, current_tool_calls)
                return value
            
            if env_id == None:
                cst = time.perf_counter()
                env_obj = self.tool_call_env_class(task_name=self.task_name)
                est = time.perf_counter()
                logger.debug(f'[ENV]: Starting fresh: so creating an evironment from scratch: {env_obj.get_id()}, took: {est - cst} for rollout {self.rollout_id}')
                return self._execute_and_put(tool_commands, 0, env_obj)

            else:
                can_fork, parent_env_id = self.client.should_fork(self.task_name, prefix_tool_calls)

                assert can_fork
                parent_env = self.tool_call_env_class(env_id=parent_env_id, task_name=self.task_name)
                
                start_time = time.perf_counter()
                quick_fork_id = self.fork_generator.get_forked_env(self.task_name, parent_env_id)
                if quick_fork_id != None:
                    logger.debug(f'[ENV]: Fork cache hit for {parent_env_id}')
                    forked_env = self.tool_call_env_class(env_id=quick_fork_id, task_name=self.task_name)
                else:
                    logger.debug(f'[ENV]: Fork cache miss')
                    forked_env = parent_env.fork()
                end_time = time.perf_counter()

                duration = end_time - start_time
                logger.debug(f'[ENV]: Forked environment: {forked_env.get_id()} in {duration:.2f} seconds from parent: {parent_env_id} for rollout {self.rollout_id}')

                self.client.unref(parent_env_id, self.task_name)
                result = self._execute_and_put(tool_commands, len(prefix_tool_calls), forked_env)
                self.async_state_update(self.task_name, self._serialize_tool_calls(tool_commands), forked_env.get_id(), parent_env_id)
                logger.debug(f'[ENV]: State update thread started for env: {forked_env.get_id()} with parent: {parent_env_id} for rollout {self.rollout_id}')
                return result


    def test(self, tool_call_history: List[ToolCall]) -> str:
        found, value = self.client.get_test_result(self.task_name, self._serialize_tool_calls(tool_call_history))

        if found:
            logger.debug(f'Found test result in Cache, CACHE HIT for task id {self.task_name}')
            return value
        logger.debug(f'No test result in Cache, CACHE MISS for task id {self.task_name}')
        test_tool_call = TestToolCall("")
        tool_call_history.append(test_tool_call)
        return self.execute(tool_call_history)
        
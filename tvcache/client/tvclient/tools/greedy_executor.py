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
FORK_THRESHOLD = 1 # seconds

class TestToolCall(ToolCall):
    def __init__(self, command: str):
        self.command = command

    def to_dict(self) -> dict:
        return {"TV_CACHE_TOOL_TYPE": "TESTING TOOL"}

    @staticmethod
    def from_dict(data: dict) -> 'TestToolCall':
        return TestToolCall(command=data["command"])

class GreedyToolCallExecutor:
    """Executes tool calls."""

    def __init__(self, tool_call_env_class: Type[ToolCallEnv], tool_call_class: Type[ToolCall], task_id):
        self.client = TVCacheClient()
        self.tool_call_env_class = tool_call_env_class
        self.tool_call_class = tool_call_class
        self.tool_call_env_obj = None
        self.task_name = task_id
        self.executed_commands = 0
        self.rollout_id = str(time.time())
        self.fork_generator = ForkGenerator(env_class=self.tool_call_env_class)
        self.rollout_environment: ToolCallEnv = None
        self.total_calls = 0
        self.total_executions = 0
        self.is_stateless = False

    def __del__(self):
        logger.debug(f'Executor for task {self.task_name} with rollout id {self.rollout_id} had total calls: {self.total_calls} and total executions: {self.total_executions}')
        if self.rollout_environment is not None:
            stop_thread = Thread(target=self.rollout_environment.stop)
            stop_thread.start()

    def set_rollout_id(self, rollout_id):
        self.rollout_id = rollout_id
    
    def set_stateless(self):
        self.is_stateless = True


    def _serialize_tool_calls(self, tool_calls: List[ToolCall]) -> List[str]:
        try:
            return [json.dumps(c.to_dict()) for c in tool_calls]
        except Exception as e:
            logger.debug(f'Failed to serialize tool calls for task {self.task_name} due to {e}')
            return []

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
            self.total_executions += 1

            if isinstance(tool_call, TestToolCall):
                assert tool_call == tool_calls[-1]
                test_result = env.test()
                logger.debug(f'[TEST EXEC]: Executed test tool call {self._serialize_tool_calls(tool_calls)} for rollout {self.rollout_id} with result {test_result}')

            else:
                self.executed_commands += 1
                st = time.perf_counter()
                last_state = env.execute(tool_call)
                et = time.perf_counter()
                values.append(last_state)
                execution_times.append((et - st))
                
                # logger.debug(f'[TOOL EXEC]: Executed tool call {self._serialize_tool_calls([tool_call])} in {et - st} seconds for rollout {self.rollout_id}')

        logger.debug(f'[TOTAL EXEC_TIME]: {sum(execution_times)} for {self.rollout_id}')
        return values, execution_times, test_result

    def _execute_and_put(self, commands: List[ToolCall], start_idx: int, env: ToolCallEnv) -> str:
        """Executes the commands in the tool calling environment and updates the prefix tree"""
        values, execution_times, test_result = self._execute_commands(commands, start_idx, env)

        if test_result != None:
            history = self._serialize_tool_calls(commands)
            logger.debug(f'Storing test result {test_result} for history: {history}')
            self.client.store_test_result(self.task_name, history[: len(history) - 1], test_result)
            logger.debug(f'[ENV]: Deleting in last step for test tool call environment {env.get_id()} of task {self.task_name} for rollout {self.rollout_id}')
            
            stop_thread = Thread(target=env.stop)
            stop_thread.start()
            
            if env is self.rollout_environment:
                self.rollout_environment = None
        
        else:
            st = time.perf_counter()

            assert len(execution_times) == len(commands) - start_idx, f"{execution_times} and start index {start_idx}" # this is to make sure that execute is called for each command

            tool_call_time = execution_times[-1]
            removed_envs = []

            if tool_call_time > (1.0) * FORK_THRESHOLD:
                # We need to store this environment in the prefix tree
                st = time.perf_counter()
                status_checker = []
                to_store = env.fork(use_async=True, result=status_checker)
                et = time.perf_counter()

                status_checker_fn = status_checker[0]
                # removed_envs = self.client.put(self.task_name, self._serialize_tool_calls(commands), to_store.get_id(), values, execution_times, start_idx)

                def wait_and_put():
                    logger.debug(f'[ENV] [FORK CHECK]: Now checking if the container has started before adding to store in rollout {self.rollout_id} for terminal {to_store.get_id()}')
                    skip_logs = 0
                    while not status_checker_fn():
                        skip_logs += 1
                        if skip_logs % 10 == 0:
                            logger.debug(f'[ENV] [FORK CHECK] WAITING STILL: {self.rollout_id} {to_store.get_id()} for task {self.task_name}')
                        time.sleep(0.5)
                    
                    logger.debug(f'[ENV] [FORK CHECK]: successfully started terminal {to_store.get_id()} in rollout {self.rollout_id} for task {self.task_name}')
                    removed_envs = self.client.put(
                        self.task_name, 
                        self._serialize_tool_calls(thread_commands), 
                        to_store.get_id(), 
                        values, 
                        execution_times, 
                        start_idx
                    )

                    if len(removed_envs) > 0:
                        remover_thread = Thread(target=self._handle_removed_envs, kwargs={'removed_envs': removed_envs})
                        remover_thread.start()

                logger.debug(f'[ENV] [FORK TIME]: Storing environment after long tool call of {tool_call_time} seconds and forking took: {et - st} seconds for rollout {self.rollout_id} and task {self.task_name} and command {self._serialize_tool_calls([commands[-1]])}')

                thread_commands = [c for c in commands]
                
                thread = Thread(target=wait_and_put, daemon=True)
                thread.start()
            else: 
                # we do not need to store this environment in the prefix tree
                def put_in_background():
                    removed_envs = self.client.put(self.task_name, self._serialize_tool_calls(commands), None, values, execution_times, start_idx)
                    if len(removed_envs) > 0:
                        remover_thread = Thread(target=self._handle_removed_envs, kwargs={'removed_envs': removed_envs})
                        remover_thread.start()
                
                thread = Thread(target=put_in_background, daemon=True)
                thread.start()
            et = time.perf_counter()
            

        if test_result != None:
            return test_result
        return values[-1]
        

    def execute(self, tool_commands: List[ToolCall]):
        """Executes the last tool call if the cache doesn't have a value associated with `current_tool_calls` prefix. The function also populates the cache if it executes the tool call."""
        current_tool_calls = self._serialize_tool_calls(tool_commands)
        if self.is_stateless:
            assert len(current_tool_calls) == 1, "Stateless mode can only execute single tool calls"
        self.total_calls += 1
        cst = time.perf_counter()
        found, env_id, value, tool_exec_time = self.client.get(self.task_name, current_tool_calls)
        est = time.perf_counter()
        if found:
            logger.debug(
                f"CACHE HIT, type 1 for task id {self.task_name} with tool calls : {current_tool_calls}  in {(est - cst)} seconds "
                f"and saved tool call time {tool_exec_time} and depth {len(current_tool_calls)} "
                f"for rollout {self.rollout_id}"
            )
            return value

        else:
            if self.is_stateless:
                logger.debug(f'CACHE MISS, in stateless mode, for task id {self.task_name} need to execute tool calls: {current_tool_calls} for rollout {self.rollout_id}')
                env_obj = self.tool_call_env_class(task_name=self.task_name)
                return self._execute_and_put(tool_commands, 0, env_obj)
            
            env_id, prefix_tool_calls = self.client.prefix_match(self.task_name, current_tool_calls)
            assert prefix_tool_calls != None
            
            if len(prefix_tool_calls) == len(current_tool_calls):
                logger.debug(f'CACHE HIT, type 2 after miss for: {self.rollout_id}')
                found, env_id, value, tool_exec_time = self.client.get(self.task_name, current_tool_calls)
                if not found:
                    raise Exception(f'Inconsistent cache state for task {self.task_name} and rollout {self.rollout_id} for tool calls {current_tool_calls}')
                return value
            logger.debug('CACHE MISS, for task id {} need to execute tool calls: {} for rollout {}'.format(self.task_name, current_tool_calls, self.rollout_id))
            if env_id == None:
                # Rollout always had cache hits so far, no environment stored
                if self.rollout_environment == None:
                    cst = time.perf_counter()
                    
                    try:
                        quick_fork_id = self.fork_generator.get_forked_env(self.task_name, "root")
                    except Exception as e:
                        logger.error(f'[ENV]: Failed to get quick forked env for root for rollout {self.rollout_id} and task {self.task_name} due to {e}')
                        quick_fork_id = None
                    if quick_fork_id != None:
                        logger.debug(f'[ENV]: Fork cache hit for root for rollout {self.rollout_id} and task {self.task_name}')
                        env_obj = self.tool_call_env_class(env_id=quick_fork_id, task_name=self.task_name)
                    else:
                        logger.debug(f'[ENV]: Fork cache miss for root for rollout {self.rollout_id} and task {self.task_name}')
                        env_obj = self.tool_call_env_class(task_name=self.task_name)
                    # env_obj = self.tool_call_env_class(task_name=self.task_name)
                    self.rollout_environment = env_obj
                    est = time.perf_counter()
                    if quick_fork_id != None:
                        logger.debug(f'[ENV]: Started from forked root environment: {env_obj.get_id()}, took: {est - cst} for rollout {self.rollout_id} and executing commands: {self._serialize_tool_calls(tool_commands)} so negCACHEHIT of {len(tool_commands) - 1}')
                    else:
                        logger.debug(f'[ENV] [FORK-TIME]: Starting fresh: so creating an evironment from scratch: {env_obj.get_id()}, took: {est - cst} for rollout {self.rollout_id} and executing commands: {self._serialize_tool_calls(tool_commands)} so negCACHEHIT of {len(tool_commands) - 1}')
                    return self._execute_and_put(tool_commands, 0, env_obj)

                # Execute in the current rollout environment
                else:
                    logger.debug(f'[ENV]: Continuing with current rollout environment: {self.rollout_environment.get_id()} for rollout {self.rollout_id} and executing commands: {self._serialize_tool_calls(tool_commands[self.executed_commands:])} so negCACHEHIT of {len(tool_commands) - self.executed_commands - 1}')
                    return self._execute_and_put(tool_commands, self.executed_commands, self.rollout_environment)
                
            else:
                parent_env_id = env_id
                logger.debug(f'[ENV]: Found cached env: {env_id} for rollout {self.rollout_id} and commands: {prefix_tool_calls}')
                if len(prefix_tool_calls) > self.executed_commands:
                    self.executed_commands = len(prefix_tool_calls)
                    logger.debug(f'[ENV]: Env has more executed commands than current count, updating executed commands to {self.executed_commands} for rollout {self.rollout_id}')
                    start_time = time.perf_counter()
                    
                    try:
                        quick_fork_id = self.fork_generator.get_forked_env(self.task_name, parent_env_id)
                    except Exception as e:
                        logger.error(f'[ENV]: Failed to get quick forked env for {parent_env_id} for rollout {self.rollout_id} for task {self.task_name} due to {e}')
                        quick_fork_id = None
                    if quick_fork_id != None:
                        logger.debug(f'[ENV]: Prefix Fork cache hit for {parent_env_id} for rollout {self.rollout_id} for task {self.task_name}')
                        forked_env = self.tool_call_env_class(env_id=quick_fork_id, task_name=self.task_name)
                    else:
                        logger.debug(f'[ENV]: Prefix Fork cache miss for rollout {self.rollout_id} for task {self.task_name}')
                        parent_env = self.tool_call_env_class(env_id=parent_env_id, task_name=self.task_name)
                        forked_env = parent_env.fork()
                        # TODO: Add priority forking here
                    end_time = time.perf_counter()

                    duration = end_time - start_time
                    logger.debug(f'[ENV]: Extended environment: {forked_env.get_id()} in {duration:.2f} seconds from parent: {parent_env_id} for rollout {self.rollout_id} so negCACHEHIT of {len(tool_commands) - len(prefix_tool_calls) - 1}')
                    self.client.unref(parent_env_id, self.task_name)
                
                    result = self._execute_and_put(tool_commands, len(prefix_tool_calls), forked_env)
                    self.rollout_environment = forked_env
                else:
                    logger.debug(f'[ENV]: Current env is ahead of cached env, for rollout {self.rollout_id} so negCACHEHIT of {len(tool_commands) - self.executed_commands - 1}')
                    result = self._execute_and_put(tool_commands, self.executed_commands, self.rollout_environment)
    
    
                # self.async_state_update(self.task_name, self._serialize_tool_calls(tool_commands), forked_env.get_id(), parent_env_id)
                # logger.debug(f'[ENV]: State update thread started for env: {self.rollout_environment.get_id()} with parent: {parent_env_id} for rollout {self.rollout_id}')
                return result


    def test(self, tool_call_history: List[ToolCall]) -> str:
        found, value = self.client.get_test_result(self.task_name, self._serialize_tool_calls(tool_call_history))

        if found:
            self.total_calls += 1
            logger.debug(f'Found test result in Cache, CACHE HIT for task id {self.task_name}')
            return value
        logger.debug(f'No test result in Cache, CACHE MISS for task id {self.task_name}')
        test_tool_call = TestToolCall("")
        tool_call_history.append(test_tool_call)
        return self.execute(tool_call_history)
        
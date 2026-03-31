# Integrating TVCache with a Training Agent Loop

This guide walks through integrating the [`tvclient`](../tvcache/client) library into an RL training agent loop. It uses this repo's video QA agent as a running example, showing the diff from the uncached baseline (`agent_loop.py`) to the TVCache version (`tvc_agent_loop.py`).

See the [tvclient README](../tvcache/client/README.md) for architecture details on the TCG and execution model.

## Overview

Integration requires three things:

1. **A `ToolCall` subclass** -- serializable representation of each tool invocation
2. **A `ToolCallEnv` subclass** -- stateful environment that executes tool calls and supports forking
3. **Wiring the executor** into the agent loop and training script

## Step 1: Define your ToolCall

Subclass `tvclient.tools.ToolCall` to represent a single tool invocation. You must implement serialization (`to_dict`/`from_dict`) and declare whether the call mutates environment state.

```python
from tvclient.tools import ToolCall

class VideoToolCall(ToolCall):
    def __init__(self, function_name: str = None, argument: str = ''):
        self.function_name = function_name
        self.argument = argument

    def to_dict(self) -> dict:
        return {"function_name": self.function_name, "argument": self.argument}

    @staticmethod
    def from_dict(data: dict) -> 'VideoToolCall':
        return VideoToolCall(
            function_name=data.get("function_name"),
            argument=data.get("argument", '')
        )

    def will_mutate_state(self) -> bool:
        # load_video_into_sandbox and preprocess change the sandbox state;
        # query tools are read-only
        return self.function_name in ('preprocess', 'load_video_into_sandbox')
```

`will_mutate_state()` is critical for Tvcache's performance.

## Step 2: Define your ToolCallEnv

Subclass `tvclient.tools.ToolCallEnv`. This wraps your actual execution backend (here, the video sandbox HTTP server) and must support `fork()` for the TCG's fork-on-divergence strategy.

```python
from tvclient.tools import ToolCallEnv
import uuid

class VideoSandboxEnv(ToolCallEnv):
    def __init__(self, env_id=None, task_name="default_task"):
        self.task_name = task_name
        self.sandbox_client = SandboxClient(base_url="http://localhost:5000")
        self.started = False

        if env_id is None:
            self.env_id = f"{self.task_name}_{uuid.uuid4().hex}"
        else:
            self.env_id = env_id
            self.sandbox_client.sandbox_id = env_id
            self.started = True

    async def execute(self, tool_call: VideoToolCall, **kwargs):
        if not self.started:
            await self.sandbox_client.start_sandbox(self.env_id)
        result = await self.sandbox_client.execute(
            tool_call.function_name, tool_call.argument
        )
        return result['result']

    async def fork(self, **kwargs) -> 'VideoSandboxEnv':
        if not self.started:
            await self.sandbox_client.start_sandbox(self.env_id)
        response = await self.sandbox_client.fork()
        return VideoSandboxEnv(env_id=response["sandbox_id"], task_name=self.task_name)

    async def stop(self, **kwargs):
        await self.sandbox_client.stop_sandbox(self.env_id)

    def get_id(self, **kwargs) -> str:
        return self.env_id

    async def get_state(self, **kwargs):
        return {}
```

Key points:
- The constructor must accept `env_id` and `task_name`. When `env_id` is provided, it restores an existing environment (used after forking or cache hits).
- `fork()` creates an independent copy of the current environment state. The sandbox server's `/fork` endpoint handles this.
- `execute()` returns the tool result as a string.

## Step 3: Wire into the agent loop

Replace direct sandbox calls with `AsyncSemanticStatefulExecutor`. The executor manages cache lookups, environment forking, and cache population automatically.

```python
from tvclient.tools.async_semantic_stateful_executor import AsyncSemanticStatefulExecutor
from tvclient.fork.dict_bank import SimpleDictBank

class VideoAgentLoop:
    def __init__(self, ..., task_id: str):
        # Create the executor instead of a raw SandboxClient
        self.executor = AsyncSemanticStatefulExecutor(
            tool_call_class=VideoToolCall,
            tool_call_env_class=VideoSandboxEnv,
            task_id=task_id,
        )

        # Track the full tool call history for this rollout
        self.executed_tool_calls: list[VideoToolCall] = []
```

Then in the tool execution loop, accumulate tool calls and pass the full history each time:

```python
# Before (uncached):
result = await self.sandbox_client.execute(function_name, argument)

# After (TVCache):
tool_call = VideoToolCall(function_name=function_name, argument=argument)
self.executed_tool_calls.append(tool_call)
result = await self.executor.execute(self.executed_tool_calls)
```

The executor compares the full tool call sequence against the TCG. On a cache hit, it returns the stored result immediately. On a partial match, it forks from the longest matching prefix and only executes the remaining calls.

Close the executor at the end of each rollout:

```python
await self.executor.close()
```

Steps 1-3 are all that's needed for TVCache integration. The executor handles cache lookups and environment forking on demand.

## Optional: Pre-warm environments with a fork bank

For additional speedup, you can pre-fork cached environments for upcoming batches so that rollouts skip the fork-on-demand latency. This is useful when forking is expensive (e.g. copying sandbox state over the network).

### Set up the fork bank

In the agent loop constructor, create a `SimpleDictBank` and attach it to the executor:

```python
from tvclient.fork.dict_bank import SimpleDictBank

# In __init__:
self.fork_generator = SimpleDictBank(env_class=VideoSandboxEnv)
self.executor.set_fork_bank(self.fork_generator)
```

### Deposit forks for the next batch

In the training script, pass next-batch task IDs to the first rollout. In the agent loop's `run()`, kick off warmup before the main loop:

```python
# In the training script, pass next batch info to the first rollout:
if d_idx == 0 and i == 0:
    data["next_batch"] = next_batch_task_ids
    data["rollout_count"] = config.group_size

# In the agent loop's run(), kick off warmup tasks:
if self.warmup_task_ids is not None:
    for task_name in self.warmup_task_ids:
        asyncio.create_task(
            self.fork_generator.deposit(
                task_name=task_name, rollout_count=self.rollout_count
            )
        )
```

`deposit()` pre-forks all cached environments for the given task, so rollouts in the next batch retrieve ready-made forks via `get_forked_env()` instead of forking on demand.

### Clean up after each batch

```python
await self.fork_generator.withdraw(task_name=self.sandbox_id)
```

## Summary of changes from uncached to TVCache

| Component | Uncached (`agent_loop.py`) | TVCache (`tvc_agent_loop.py`) |
|---|---|---|
| Tool execution | `SandboxClient.execute()` directly | `AsyncSemanticStatefulExecutor.execute()` with full history |
| Environment lifecycle | Manual `start_sandbox`/`stop_sandbox` | Executor manages environments; fork bank pre-warms |
| Cache | None | TCG via TVCache server |
| Training script | `train_without_cache.py` | `train_with_tvcache.py` (passes `task_id`, next-batch warmup) |

## Reference

- `tvclient.tools.ToolCall` -- abstract tool call interface ([source](../tvcache/client/tvclient/tools/tool_call_env.py))
- `tvclient.tools.ToolCallEnv` -- abstract environment interface ([source](../tvcache/client/tvclient/tools/tool_call_env.py))
- `AsyncSemanticStatefulExecutor` -- main executor ([source](../tvcache/client/tvclient/tools/async_semantic_stateful_executor.py))
- `SimpleDictBank` -- fork bank implementation ([source](../tvcache/client/tvclient/fork/dict_bank.py))
- `AbstractForkGenerator` -- fork bank interface ([source](../tvcache/client/tvclient/fork/abstract_bank.py))

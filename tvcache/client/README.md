# TVCache: Tool-Value Caching for Agent Training

TVCache is a caching system that accelerates RLHF-based agent training by storing and reusing intermediate tool execution states. It is designed for integration with training frameworks such as VERL, Agent-Lightning, and TRL.

## Installation

### Using pip

```bash
pip install -e .
```

### Using uv

```bash
uv pip install -e .
```

## Architecture

TVCache consists of two components:

- **Server**: Maintains a prefix tree of tool call sequences for each training example
- **Client**: Integrates with the agent training loop

## Prefix Tree Structure

For each data point in the training dataset, the server maintains a prefix tree where:

- Each node stores a tool call identifier (name and arguments), its response, and an associated environment ID
- Environment IDs are invalidated when subsequent commands modify the environment
- Tool call sequences from different rollouts form paths through the tree

## Execution Model

Tool call reuse follows these rules:

1. **Cache hit**: If a rollout's tool call sequence matches an existing path, cached values are returned directly
2. **Partial match**: If no complete match exists, the system identifies the environment with the longest common prefix and executes only the remaining commands
3. **Prefix locking**: When multiple rollouts share a prefix, one rollout executes while others wait for the result
4. **Environment forking**: Rollouts with divergent suffixes fork the environment and execute independently

## Integration

Training frameworks integrate TVCache by implementing a `ToolCallEnv` class for the dataset and using it with `ToolCallExecutor`.

### Example: Calculator Environment

```python
from typing import Optional
import uuid
from tvclient.tools import ToolCallEnv, ToolCallExecutor

class CalculatorEnv(ToolCallEnv):
    """Environment that maintains calculator state."""

    def __init__(self, env_id: Optional[str] = None):
        self.env_id = env_id if env_id else str(uuid.uuid4())
        self.value = 0

    def execute(self, tool_call):
        """Execute arithmetic operations."""
        operation, operand = tool_call.split()
        operand = float(operand)

        if operation == "add":
            self.value += operand
        elif operation == "multiply":
            self.value *= operand

        return self.value

    def fork(self) -> 'CalculatorEnv':
        """Create a copy of the current state."""
        forked = CalculatorEnv(env_id=str(uuid.uuid4()))
        forked.value = self.value
        return forked

    def get_state(self):
        return self.value

    def get_id(self):
        return self.env_id

    def stop(self):
        pass

# Usage in training loop
executor = ToolCallExecutor(CalculatorEnv, task_id="data_point_123")

# Execute sequence of tool calls
commands = ["add 5", "multiply 2", "add 3"]
for i, cmd in enumerate(commands):
    result = executor.execute(commands[:i+1])
    print(f"After '{cmd}': {result}")  # Cached if previously executed
```

For more complex examples, such as terminal environments with concurrent rollouts, see the `examples/` directory.

## Prefix Tree Evolution

The following visualizations demonstrate how the prefix tree evolves across training epochs. Each batch of rollouts executes in a different epoch.

### Epoch 1

**Rollouts:**
```
ROLLOUT-0: ['ls -la', 'cat test.txt', './thread_0']
ROLLOUT-1: ['ls -la', 'cat test.txt', './thread_1']
ROLLOUT-2: ['ls -la', 'cat test.txt', './thread_2']
ROLLOUT-3: ['ls -la', 'cat test.txt', './thread_3']
ROLLOUT-4: ['ls -la', 'cat test.txt', './thread_4']
```

![Epoch 1 Prefix Tree](examples/demo/5.png)

All rollouts share the common prefix `['ls -la', 'cat test.txt']`, which is cached. The tree then branches into five different thread executions.

### Epoch 2

**Rollouts:**
```
ROLLOUT-0: ['ls -la', 'cat test.txt', 'echo code > input.txt', './thread_0']
ROLLOUT-1: ['ls -la', 'cat test.txt', 'echo code > input.txt', './thread_1']
ROLLOUT-2: ['ls -la', 'cat test.txt', 'echo code > input.txt', './thread_2']
ROLLOUT-3: ['ls -la', 'cat test.txt', 'echo code > input.txt', './thread_3']
ROLLOUT-4: ['ls -la', 'cat test.txt', 'echo code > input.txt', './thread_4']
```

![Epoch 2 Prefix Tree](examples/demo/10.png)

A new node `'echo code > input.txt'` is added to the common prefix. The existing thread nodes from Epoch 1 are preserved, and new forked environments execute the updated sequences.

### Epoch 3

**Rollouts:**
```
ROLLOUT-0: ['ls -la', 'cat test.txt', 'echo code > input.txt', './thread_0', 'rm input.txt']
ROLLOUT-1: ['ls -la', 'cat test.txt', 'echo code > input.txt', './thread_1', 'rm input.txt']
ROLLOUT-2: ['ls -la', 'cat test.txt', 'echo code > input.txt', './thread_2', 'rm input.txt']
ROLLOUT-3: ['ls -la', 'cat test.txt', 'echo code > input.txt', './thread_3', 'rm input.txt']
ROLLOUT-4: ['ls -la', 'cat test.txt', 'echo code > input.txt', './thread_4', 'rm input.txt']
```

![Epoch 3 Prefix Tree](examples/demo/15.png)

The tree grows deeper as `'rm input.txt'` commands are added. Each thread execution branches further, demonstrating how TVCache efficiently manages increasingly complex execution paths while maximizing prefix reuse.
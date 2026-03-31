# TVCache Server

A high-performance HTTP server implementation for the TVCache tool-value caching system. This server provides efficient prefix-based caching and retrieval of tool call execution environments for intelligent agents.

## Overview

The TVCache server implements a prefix tree (trie) data structure to enable fast lookups of cached tool call sequences. When an agent executes a sequence of tool calls, the server can quickly determine if a matching prefix exists in the cache, allowing the agent to resume from a previously cached state rather than re-executing identical tool call sequences.

## Architecture

The server exposes the following HTTP endpoints:

- **PUT /put**: Store a new tool call history with associated environment state
- **GET /get**: Retrieve an exact match for a complete tool call sequence
- **POST /prefix_match**: Find the longest prefix match for a tool call sequence
- **POST /lock**: Acquire an exclusive execution lock on a task environment
- **DELETE /unlock**: Release an execution lock on a task environment
- **GET /visualize**: Get the entire prefix tree structure for visualization
- **GET /**: Serve the interactive tree visualizer web interface

The server maintains an in-memory prefix tree structure where each node represents a tool call in a sequence. Each node can store an environment ID and associated value, enabling efficient retrieval of cached execution states.

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

Install dependencies with uv:

```bash
uv sync
```

This will automatically install all required dependencies (Flask, requests) specified in `pyproject.toml`.

## Usage

### Starting the Server

Run the server with uv:

```bash
uv run tvcache_server.py
```

By default, the server runs on `http://localhost:8000`. To customize the host and port:

```python
from tvcache_server import run_server

run_server(host='0.0.0.0', port=9000, debug=False)
```

### API Endpoints

#### Store Tool Call History

```
PUT /put
Content-Type: application/json

{
  "task_name": "string",
  "history": ["tool1", "tool2", "tool3"],
  "env_id": "string",
  "value": "any"
}
```

**Response:**
```json
{
  "success": true
}
```

#### Exact Match Retrieval

```
GET /get?task_name=<task>&tool_calls=<tool1>&tool_calls=<tool2>
```

**Response:**
```json
{
  "found": true,
  "env_id": "string",
  "value": "any"
}
```

#### Prefix Match Retrieval

```
POST /prefix_match
Content-Type: application/json

{
  "task_name": "string",
  "tool_calls": ["tool1", "tool2", "tool3", "tool4"]
}
```

**Response:**
```json
{
  "found": true,
  "env_id": "string",
  "history": ["tool1", "tool2", "tool3"]
}
```

The server returns the longest matching prefix from the cache. In this example, if only the first three tools were cached, the response includes those three tools in the history field.

#### Acquire Execution Lock

```
POST /lock
Content-Type: application/json

{
  "task_name": "string",
  "env_id": "string",
  "history": ["tool1", "tool2", "tool3"]
}
```

**Response:**
```json
{
  "acquired": true,
  "tool_calls": ["tool1", "tool2", "tool3"]
}
```

Acquires an exclusive lock to prevent concurrent execution of the same cached environment.

#### Release Execution Lock

```
DELETE /unlock
Content-Type: application/json

{
  "task_name": "string",
  "env_id": "string"
}
```

**Response:**
```json
{
  "released": true
}
```

#### Visualize Prefix Tree

```
GET /visualize
```

**Response:**
```json
{
  "task_name": {
    "env_id": "string",
    "value": "any",
    "children": { ... }
  }
}
```

Returns the complete prefix tree structure for all tasks. Visit `http://localhost:8000/` in a browser to see an interactive visualization.

## Testing

### Running Tests

The server includes a comprehensive test suite. First, start the server in one terminal:

```bash
uv run tvcache_server.py
```

Then run the tests in another terminal:

```bash
uv run tests/simple_test_server.py
```

## Data Structure

The server uses a prefix tree (trie) data structure for efficient storage and retrieval:

```
task_name -> Node
              |
              +-- tool1 -> Node (env_id, value)
                           |
                           +-- tool2 -> Node (env_id, value)
                                        |
                                        +-- tool3 -> Node (env_id, value)
```

Each node in the tree represents a tool call, and nodes can store environment IDs and values at any depth. This structure enables O(k) lookups where k is the length of the tool call sequence.

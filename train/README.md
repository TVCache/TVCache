# Video QA Agent Training with Tinker API

RL training pipeline for a video question-answering agent on the [EgoSchema](https://egoschema.github.io/) benchmark. The agent uses tool-calling to analyze long-form egocentric videos and answer multiple-choice questions. Training uses [Tinker](https://tinker-docs.thinkingmachines.ai/training-sampling) for model serving, sampling, and gradient updates.

## Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- A Tinker API key
- A running video sandbox server (default: `http://localhost:5000`) -- see [Setting Up the Sandbox Server](#setting-up-the-sandbox-server) below

## Setup

Install dependencies:

```bash
uv sync
```

This installs the `tinker` SDK and all transitive dependencies (torch, pydantic, httpx, etc.) as specified in `pyproject.toml` and locked in `uv.lock`.

## Setting Up the Sandbox Server

The training scripts call video analysis tools via an HTTP sandbox server from [`../video-agent-tools/VideoAgent`](../video-agent-tools/VideoAgent). See its [README](../video-agent-tools/VideoAgent/README.md) for full setup instructions. In short:

1. Create the VideoAgent conda environment and the Video-LLaVA venv
2. Download model weights (`cache_dir.zip` and `tool_models.zip`) from [Zenodo](https://zenodo.org/records/11031717)
3. Start the Video-LLaVA server in one terminal, then the sandbox server in another:
   ```bash
   # Terminal 1: Video-LLaVA
   cd ../video-agent-tools/Video-LLaVA && source .venv/bin/activate
   cd ../VideoAgent && python video-llava.py

   # Terminal 2: Sandbox server
   cd ../video-agent-tools/VideoAgent && conda activate ./cenv
   python sandbox_server.py  # http://0.0.0.0:5000
   ```

The training scripts connect to this server via `sandbox_base_url` (default `http://localhost:5000`).

---

## Step 1: Download EgoSchema Videos

The `EgoSchema/` directory contains the dataset metadata. Use `download.py` to fetch video files from Google Drive.

```bash
cd EgoSchema
uv run download.py
```

This will:
1. Load `questions.json` and filter to the 500 questions with public answers in `subset_answers.json`
2. Randomly sample 250 videos from that subset (seeded for reproducibility)
3. Download each video as an `.mp4` into `EgoSchema/videos/`
4. Validate downloaded files

Downloads are skipped for videos that already exist in `videos/`, so the script is safe to re-run if interrupted.

## Step 2: Process Videos

After downloading, run `process_videos.py` to build the structured dataset file:

```bash
cd EgoSchema
uv run process_videos.py
```

This reads `questions.json`, `subset_answers.json`, and the contents of `videos/`, then writes `processed_videos.json`. Each entry contains:

```json
{
  "video_id": "abcdef123",
  "video_file": "abcdef123.mp4",
  "question": "What did the person do after ...",
  "options": {"0": "...", "1": "...", "2": "...", "3": "...", "4": "..."},
  "correct_answer_index": 2,
  "correct_answer_text": "...",
  "google_drive_id": "..."
}
```

The training scripts load this file from `./EgoSchema/processed_videos.json`.

## Step 3: Install TVCache Client

[TVCache](../tvcache) accelerates training by caching tool execution results across rollouts using a Tool Call Graph. This avoids redundant sandbox calls when multiple rollouts share the same tool call prefix.

Install the client library from the sibling directory:

```bash
uv pip install -e ../tvcache/client
```

This makes the `tvclient` package available, which provides `AsyncSemanticStatefulExecutor`, `ToolCallEnv`, and the fork bank utilities used by `tvc_agent_loop.py`.

The TVCache server must also be running for the cached training variant to work. See `../tvcache/server/` for server setup.

## Step 4: Run Training

Edit `run.sh` and replace `YOUR_KEY` with your Tinker API key:

```bash
#!/bin/bash
export TINKER_API_KEY=YOUR_KEY
uv run $1
```

Then run one of the three training scripts:

### Without caching (baseline)

```bash
./run.sh train_without_cache.py
```

### With stateless in-memory caching

```bash
./run.sh train_with_stateless_cache.py
```

Caches tool results in a thread-safe dictionary keyed by `(function_name, argument)`. Cache is shared across rollouts within a batch but does not persist across batches.

### With TVCache (TCG caching)

```bash
./run.sh train_with_tvcache.py
```

Uses `AsyncSemanticStatefulExecutor` to maintain a Tool Call Graph of tool call sequences per data point. Supports environment forking, prefix reuse across epochs, and warmup of the next batch's environments.

## Training Configuration

All three scripts use `chz` for configuration. Defaults:

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Base model for LoRA fine-tuning |
| `batch_size` | `4` | Data points per training step |
| `group_size` | `8` | Rollouts per data point |
| `learning_rate` | `4e-5` | Adam learning rate |
| `lora_rank` | `32` | LoRA adapter rank |
| `max_tokens` | `1024` | Max tokens per model generation |
| `num_turns` | `5` | Max agent turns per rollout |
| `max_length` | `32768` | Max sequence length |
| `epochs` | `10` | Number of passes over the dataset |
| `save_every` | `5` | Checkpoint frequency (batches) |
| `sandbox_base_url` | `http://localhost:5000` | Video sandbox server URL |

Override any parameter via CLI:

```bash
./run.sh train_with_tvcache.py --config.learning_rate=1e-5 --config.batch_size=2
```

## How Training Works

1. The agent receives a video QA prompt with 6 available tools: `load_video_into_sandbox`, `preprocess`, `object_memory_querying`, `segment_localization`, `caption_retrieval`, and `visual_question_answering`.
2. For each data point, `group_size` rollouts are generated. Each rollout runs a multi-turn agent loop where the model generates JSON tool calls, executes them against the video sandbox, and iterates until it produces a `final_answer`.
3. Rewards: **+1** for correct answer, **-2** for invalid JSON parse, **0** otherwise.
4. Advantages are computed per group (`reward - mean_reward`). Groups where all advantages are zero are skipped.
5. A Tinker `forward_backward_async` call with `importance_sampling` loss updates the LoRA weights.

## Project Structure

```
├── run.sh                         # Entry point (sets API key, runs via uv)
├── pyproject.toml                 # Project dependencies
├── prompt.txt                     # System prompt template for the agent
├── tool_schema.py                 # Pydantic schema for agent responses
├── agent_loop.py                  # Agent loop (no caching)
├── cached_agent_loop.py           # Agent loop with in-memory cache
├── tvc_agent_loop.py              # Agent loop with TVCache
├── train_without_cache.py         # Training script (baseline)
├── train_with_stateless_cache.py  # Training script (dict cache)
├── train_with_tvcache.py          # Training script (TVCache)
├── plot.py                        # Plot training metrics across runs
├── EgoSchema/
│   ├── download.py                # Download videos from Google Drive
│   ├── process_videos.py          # Build processed_videos.json
│   ├── questions.json             # EgoSchema questions
│   ├── subset_answers.json        # Public answers (500 subset)
│   └── videos/                    # Downloaded video files
└── utils/
    └── video_sandbox_client.py    # HTTP client for the video sandbox
```

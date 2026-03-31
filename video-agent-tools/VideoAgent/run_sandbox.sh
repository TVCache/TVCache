#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY=your_key

python3 sandbox_server.py
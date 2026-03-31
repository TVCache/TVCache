# VideoAgent

Sandbox adapted from VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding (ECCV 2024).

## Prerequisites

- Ubuntu 20.04+
- 2 NVIDIA GPUs
- [Conda](https://docs.conda.io/en/latest/miniconda.html) installed
- FFmpeg (`sudo apt install ffmpeg` or provided by conda)
- An OpenAI API key (GPT-4 for reasoning, GPT-4.1-mini for frame captioning, `text-embedding-3-large` for text embeddings)

## Installation

### 1. Create the VideoAgent conda environment

```bash
conda env create -f environment.yaml -p ./cenv   # creates env in ./cenv
conda activate ./cenv
```

### 2. Create the Video-LLaVA environment

Video-LLaVA runs as a separate process for visual question answering. See the [Video-LLaVA README](../Video-LLaVA/README.md) for full setup instructions. The short version:

```bash
cd ../Video-LLaVA
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e .
uv pip install flash-attn --no-build-isolation
uv pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
deactivate
```

### 3. Download model weights

Download `cache_dir.zip` and `tool_models.zip` from [Zenodo](https://zenodo.org/records/11031717) and unzip them into this directory:

```bash
# From the VideoAgent/ directory
wget https://zenodo.org/records/11031717/files/cache_dir.zip
wget https://zenodo.org/records/11031717/files/tool_models.zip
unzip cache_dir.zip
unzip tool_models.zip
```

This creates two directories:

| Directory | Contents |
|---|---|
| `cache_dir/` | Video-LLaVA-7B model weights (used by `video-llava.py`) |
| `tool_models/` | All other model weights (see below) |

**Model weights in `tool_models/`:**

| Model | Path | Purpose |
|---|---|---|
| CLIP ViT-B-32 | `tool_models/CLIP/ViT-B-32.pt` | Object embeddings (tracking, ReID, retrieval) |
| RTDETR-L | `tool_models/tracking/rtdetr-l.pt` | Object detection & tracking |
| viCLIP | `tool_models/viCLIP/ViClip-InternVid-10M-FLT.pth` | Video segment visual features |
| DINOv2 | `tool_models/facebookresearch_dinov2_main/` | Object re-identification features |
| LaViLa | `tool_models/LaViLa/vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth` | Video captioning |

## Usage

### Running the sandbox server

The sandbox server exposes VideoAgent's preprocessing and inference capabilities over HTTP, enabling parallel processing of multiple videos via isolated sandboxes.

**Terminal 1** -- Start the Video-LLaVA server (same as above):

```bash
cd ../Video-LLaVA
source .venv/bin/activate
cd ../VideoAgent
python video-llava.py
```

**Terminal 2** -- Start the sandbox server after setting the env variables in `run_sandbox.sh`:

```bash
conda activate ./cenv
./run_sandbox.sh
# Starts Flask server on http://0.0.0.0:5000
```

#### Sandbox server API

All endpoints accept JSON via POST. The `sandbox_id` field is required for every request.

| Endpoint | Description |
|---|---|
| `/start` | Creates a new sandbox directory for processing a video. |
| `/stop` | Stops and removes a sandbox, cleaning up its files. |
| `/fork` | Creates an independent copy of an existing sandbox. Returns the new `sandbox_id`. |
| `/execute` | Runs a command inside a sandbox. Takes `command` and optional `argument` fields. |

**Commands available via `/execute`:**

| Command | Argument | Description |
|---|---|---|
| `load_video_into_sandbox` | Video name | Loads a video file into the sandbox. |
| `preprocess` | -- | Runs the full memory construction pipeline (captioning, tracking, ReID, segment features). |
| `object_memory_querying` | Question string | Queries the object-segment database using natural language. |
| `segment_localization` | Description string | Finds the top-5 video segments matching a text description. |
| `caption_retrieval` | `"(start, end)"` | Retrieves or generates captions for a range of segments. |
| `visual_question_answering` | `"(start, end, question)"` | Answers a question about a specific segment range using Video-LLaVA or GPT-4V. |

## Citation

```bibtex
@inproceedings{fan2025videoagent,
  title={Videoagent: A memory-augmented multimodal agent for video understanding},
  author={Fan, Yue and Ma, Xiaojian and Wu, Rujie and Du, Yuntao and Li, Jiaqi and Gao, Zhi and Li, Qing},
  booktitle={European Conference on Computer Vision},
  pages={75--92},
  year={2025},
  organization={Springer}
}
```

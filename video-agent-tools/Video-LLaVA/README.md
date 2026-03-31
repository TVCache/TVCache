# Video-LLaVA

Learning United Visual Representation by Alignment Before Projection (EMNLP 2024). A large vision-language model that unifies image and video understanding through aligned visual representations.

## Prerequisites

- Python >= 3.10
- NVIDIA GPU with CUDA >= 11.7
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed

## Installation

```bash
cd Video-LLaVA

# Create and activate venv
uv venv --python 3.10 .venv
source .venv/bin/activate

# Install the package
uv pip install -e .

# Install additional required packages
uv pip install flash-attn --no-build-isolation
uv pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
```

To install training dependencies as well:

```bash
uv pip install -e ".[train]"
```

## Model weights

The primary model is `LanguageBind/Video-LLaVA-7B`, hosted on HuggingFace. It is downloaded automatically on first use and cached in `./cache_dir` (or wherever `--cache-dir` points).

Dependent models (also auto-downloaded from HuggingFace):
- `LanguageBind/LanguageBind_Image` -- image vision encoder
- `LanguageBind/LanguageBind_Video_merge` -- video vision encoder
- `lmsys/vicuna-7b-v1.5` -- base LLM

**For use with VideoAgent:** Download `cache_dir.zip` from [Zenodo](https://zenodo.org/records/11031717) and unzip it into the `VideoAgent/` directory. This provides the pre-downloaded Video-LLaVA-7B weights so the VideoAgent VQA server (`video-llava.py`) can load them offline.

## Usage

### CLI inference

```bash
# Video
CUDA_VISIBLE_DEVICES=0 python -m videollava.serve.cli \
  --model-path LanguageBind/Video-LLaVA-7B \
  --file path/to/video.mp4 \
  --load-4bit

# Image
CUDA_VISIBLE_DEVICES=0 python -m videollava.serve.cli \
  --model-path LanguageBind/Video-LLaVA-7B \
  --file path/to/image.jpg \
  --load-4bit
```

### Gradio web UI

```bash
python -m videollava.serve.gradio_web_server
# Open http://localhost:7860
```

### Python API (using transformers)

```python
import av, numpy as np
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

container = av.open("video.mp4")
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)

frames = []
container.seek(0)
for i, frame in enumerate(container.decode(video=0)):
    if i > indices[-1]:
        break
    if i in indices:
        frames.append(frame)
clip = np.stack([x.to_ndarray(format="rgb24") for x in frames])

inputs = processor(text="USER: <video>What is happening in this video? ASSISTANT:", videos=clip, return_tensors="pt")
output = model.generate(**inputs, max_length=80)
print(processor.batch_decode(output, skip_special_tokens=True)[0])
```

### As a VQA server for VideoAgent

Video-LLaVA serves as the visual question answering backend for [VideoAgent](../VideoAgent/README.md). To run it:

```bash
cd Video-LLaVA
source .venv/bin/activate
cd ../VideoAgent
python video-llava.py
# Wait for "ready for connection!"
```

This starts a Unix socket server at `tmp/vqa.sock` that VideoAgent's sandbox manager connects to for VQA requests. The model loads with 4-bit quantization from `cache_dir/`.

## Training

### Stage 1: Pretraining (MM adapter only)

```bash
bash scripts/v1_5/pretrain.sh
```

### Stage 2: Fine-tuning (full model)

```bash
bash scripts/v1_5/finetune.sh
```

### LoRA fine-tuning

```bash
bash scripts/v1_5/finetune_lora.sh
```

See [TRAIN_AND_VALIDATE.md](TRAIN_AND_VALIDATE.md) for dataset preparation and detailed training instructions.

## Citation

```bibtex
@inproceedings{lin2024video,
  title={Video-LLaVA: Learning United Visual Representation by Alignment Before Projection},
  author={Lin, Bin and Zhu, Bin and Ye, Yang and Ning, Munan and Jin, Peng and Yuan, Li},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  year={2024}
}
```

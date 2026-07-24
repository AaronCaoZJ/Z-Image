# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Z-Image (造相) is a 6B-parameter text-to-image foundation model from Tongyi-MAI (Alibaba). This repo ships two independent ways to run inference on the **Z-Image-Turbo** checkpoint:

1. **PyTorch-native** — a from-scratch reimplementation under `src/` (package name `zimage-native`), driven by `inference.py` / `batch_inference.py`.
2. **Diffusers** — `diffusers-inference.py`, which uses the upstream `ZImagePipeline` from 🤗 diffusers (requires diffusers installed from source).

The two paths share weights but **share no code** — `diffusers-inference.py` does not touch `src/` at all.

## Commands

```bash
# Install the native package (editable). The .venv here was created with uv; prefer `uv run` if present.
pip install -e .              # or: uv pip install -e .
pip install -e ".[dev]"       # adds black, isort, ruff

# Run native inference (single image -> outputs/8.png)
python inference.py

# Batch inference over a prompt file (one prompt per line)
PROMPTS_FILE=prompts/prompt1.txt python batch_inference.py

# Diffusers path (needs `pip install git+https://github.com/huggingface/diffusers`)
python diffusers-inference.py

# Regenerate the weight-integrity manifest after updating a checkpoint
python -m tools.generate_manifest ckpts/Z-Image-Turbo            # with MD5
python -m tools.generate_manifest ckpts/Z-Image-Turbo --no-checksums
```

There is **no test suite** and no lint config is committed (`.isort.cfg` / `.pre-commit-config.yaml` are gitignored). Verify changes by running an actual generation and inspecting the output image.

### Selecting the attention backend / device
- Backend is chosen via the `ZIMAGE_ATTENTION` env var (default `_native_flash`). Valid values: `flash`, `flash_varlen`, `_flash_3`, `_flash_varlen_3`, `native`, `_native_flash`, `_native_math`. The Flash variants require `flash-attn` (2 or 3) to be installed.
- For best speed on Hopper GPUs (H100/H200/H800), use `_flash_3` + `compile=True` for sub-second latency after warm-up.
- Device is auto-detected: `cuda` → TPU (`torch_xla`) → `mps` → `cpu`.

## Weights

Checkpoints live in `ckpts/Z-Image-Turbo/` (gitignored). `ensure_model_weights()` in `src/utils/helpers.py` checks the dir against a manifest in `src/config/manifests/z-image-turbo.txt` and, if files are missing, auto-downloads from the HF repo `Tongyi-MAI/Z-Image-Turbo` via `snapshot_download`. The native path expects a diffusers-style layout: `transformer/`, `vae/`, `text_encoder/`, `tokenizer/`, `scheduler/` subdirs.

## Architecture (native path)

`src/` is laid out as **top-level packages** (`zimage`, `utils`, `config`, `tools`) because setuptools uses `where = ["src"]`. After `pip install -e .` these import as `from zimage import ...`, `from utils import ...`, `from config import ...` — note the absence of any `src.` prefix. This is why `inference.py` imports `from zimage import generate` directly.

Flow of a generation (`zimage/pipeline.py::generate` is the single entry point):

1. **Text encoding** — The text encoder is a Qwen-style causal LLM loaded via `AutoModel(trust_remote_code=True)`. Prompts are wrapped with the tokenizer's **chat template** (`enable_thinking=True`), and embeddings are taken from the **penultimate** hidden layer (`hidden_states[-2]`), not the last. Each prompt's embedding is kept as a variable-length, mask-trimmed tensor — embeddings are passed around as a **Python list**, never a padded batch tensor.
2. **Latent init + scheduler** — `FlowMatchEulerDiscreteScheduler` (flow matching, reimplemented in `zimage/scheduler.py`). `calculate_shift()` computes a resolution-dependent `mu` shift from the image sequence length.
3. **Denoising loop** — `ZImageTransformer2DModel` (`zimage/transformer.py`) is a **Scalable Single-Stream DiT (S3-DiT)**. Critically, its `forward` takes a **list of per-sample latents** (the pipeline does `latents.unbind(0)`) and a **list of caption embeddings**, not batched tensors. Internally it: patchifies image tokens, pads each sequence to a multiple of `SEQ_MULTI_OF` (32), runs `noise_refiner` (2 modulated blocks) on image tokens and `context_refiner` (2 non-modulated blocks) on caption tokens, then **concatenates image + caption tokens into one unified stream** through 30 main DiT blocks. Uses 3-axis RoPE, RMSNorm QK-norm, adaLN modulation, and SwiGLU FFN.
4. **VAE decode** — `zimage/autoencoder.py` (`AutoencoderKL`), always run in **fp32** for precision while the transformer/text-encoder run in **bf16**.

### Inference-time invariants (easy to get wrong)
- **Turbo uses no CFG**: `guidance_scale=0.0`. CFG branches only activate when `guidance_scale > 1.0`. `cfg_truncation` disables guidance for early/high-noise timesteps.
- `num_inference_steps=9` yields **8 actual DiT forwards** — the final `t==0` step is skipped.
- `height` and `width` must be divisible by **16** (`vae_scale_factor * 2`).
- The model predicts flow; the pipeline negates the model output (`noise_pred = -noise_pred`) before the scheduler step, and latents stay fp32 throughout the loop.

### Config constants
All tunables live in `src/config/` (`model.py` = architecture dims/RoPE/VAE; `inference.py` = default H/W/steps/guidance). The transformer's real config is read from `ckpts/.../transformer/config.json` at load time; the `DEFAULT_TRANSFORMER_*` constants are only fallbacks.

### Attention dispatch
`utils/attention.py` is a self-contained backend registry (adapted from diffusers). `ZImageAttention` reads the active backend from the class attribute `ZImageAttention._attention_backend`, set globally by `set_attention_backend()`. Dispatch is written as an explicit `if/elif` chain (not a dict lookup) specifically to avoid `torch.compile`/Dynamo graph breaks — keep it that way when editing.

## Gotchas

- `batch_inference.py` currently does `from inference import ensure_weights`, but `inference.py` only exposes `ensure_model_weights` (imported from `utils`). This import will fail as-is — fix the name before relying on batch mode.
- `diffusers-inference.py` and `inference.py` show local working changes (e.g. a hardcoded absolute checkpoint path, `512x512` output) — treat them as scratch/experiment scripts, not canonical config.
- `utils/loader.py` does a `sys.path.insert` assuming a sibling `Z-Image/src` layout; if you relocate the repo, loading may need adjustment.

# Muse Upscale Advanced

One node: **Upscale Model Advanced (Muse)** — load an upscale model, run it (tiled and
OOM-safe), then resize the result to an exact target resolution (a multiplier of the
original input, or a fixed width/height).

No third-party dependency — only ComfyUI's own core internals.

## Features

- Any model from `models/upscale_models`
- Exact output size: either a multiplier of the original input, or a fixed width/height
- Tiled, memory-safe processing — blended tile edges, and automatically shrinks tile
  size further on a CUDA out-of-memory error instead of failing outright
- Precision control (auto / fp32 / fp16 / bf16) and batch-size control for VRAM
- Model offloading after use, and an optional result cache for repeated calls with
  identical inputs

## Inputs

| Widget | What it does |
|---|---|
| `upscale_model_name` | Model file from `models/upscale_models` |
| `use_fixed_resolution` | On: use `fixed_width`/`fixed_height`. Off: use `output_multiplier`. |
| `output_multiplier` | Final size relative to the *original* input (1.0 = same size, 2.0 = double) |
| `fixed_width` / `fixed_height` | Exact target size when `use_fixed_resolution` is on |
| `tile_count` | Starting tile grid per side (1 = single pass) |
| `precision` | Processing precision for the model pass |
| `batch_size` | Images processed together per pass |
| `offload_model` | Offload the model to CPU after use |
| `disable_cache` | Skip the result cache and always reprocess |

Outputs: `IMAGE`, `width`, `height`.

## Install

Drop this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.

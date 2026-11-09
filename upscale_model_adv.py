"""
Muse Upscale Model Advanced.

Load an upscale model, run it (tiled and OOM-safe), then resize the result to an
exact target resolution — either a multiplier of the original input, or a fixed
width/height. No dependency beyond ComfyUI's own core (UpscaleModelLoader /
ImageUpscaleWithModel / comfy.utils) — nothing third-party required.

Two things worth knowing about how this is built, both verified directly against
ComfyUI's own core source (comfy_extras/nodes_upscale_model.py):

1. Model scale factor is read straight from the loaded model itself
   (`upscale_model.scale` — the same attribute ComfyUI's own stock upscale node
   uses), rather than guessed from the filename. A filename-substring guess
   ("2x"/"4x"/"8x") is kept only as a last-resort fallback if that attribute is
   ever missing.

2. Tiling is delegated to ComfyUI's own `comfy.utils.tiled_scale` — the exact
   routine stock's own ImageUpscaleWithModel uses — which blends overlapping tile
   edges (no hard seams at tile boundaries) and automatically halves tile size and
   retries on a CUDA out-of-memory error rather than failing outright. `tile_count`
   sets the starting tile grid; it can only get more robust from there under
   memory pressure, never less.
"""

import hashlib
import json
import logging

import torch

import comfy.model_management as model_management
import comfy.utils
import folder_paths
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader

log = logging.getLogger("MuseUpscaleAdvanced")


class MuseUpscaleModelAdv:
    """Load an upscale model, run it (tiled + OOM-safe), then resize the result to an
    exact target resolution — either a multiplier of the original input size, or a
    fixed width/height. See the module docstring for what's deliberately improved
    over a plain load-model-then-upscale approach."""

    _cache = {}
    _max_cache_size = 5

    precision_options = ["auto", "fp32", "fp16", "bf16"]
    tile_count_options = ["1", "4", "8", "16"]

    def __init__(self):
        self.upscale_loader = UpscaleModelLoader()
        self.image_upscaler = ImageUpscaleWithModel()

    @classmethod
    def INPUT_TYPES(cls):
        upscale_models = folder_paths.get_filename_list("upscale_models")
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model_name": (
                    upscale_models,
                    {"tooltip": "Select upscale model from models/upscale_models folder"},
                ),
                "use_fixed_resolution": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable to use fixed width/height instead of a multiplier"},
                ),
                "output_multiplier": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 8.0,
                        "step": 0.25,
                        "tooltip": "Final output size multiplier relative to the ORIGINAL input "
                                   "(1.0=same as input, 2.0=double input size) — independent of "
                                   "the upscale model's own native scale factor.",
                    },
                ),
                "fixed_width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 8192, "step": 8,
                     "tooltip": "Target width for the upscaled image (used when use_fixed_resolution is on)"},
                ),
                "fixed_height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 8192, "step": 8,
                     "tooltip": "Target height for the upscaled image (used when use_fixed_resolution is on)"},
                ),
                "tile_count": (
                    cls.tile_count_options,
                    {"default": "1", "tooltip": "Starting tile grid per side (1=single pass). Tiles are blended "
                                                 "at their edges and automatically shrunk further on a CUDA "
                                                 "out-of-memory error, rather than failing outright."},
                ),
                "precision": (
                    cls.precision_options,
                    {"default": "auto", "tooltip": "Processing precision (auto=fp16 on CUDA, bf16 elsewhere)"},
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 16, "step": 1,
                     "tooltip": "Number of images processed together per pass (lower = less peak VRAM)"},
                ),
                "offload_model": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Offload model to CPU after processing to save VRAM"},
                ),
                "disable_cache": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Disable caching to always reprocess images"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height")
    FUNCTION = "upscale_advanced"
    CATEGORY = "Muse/Image"
    DESCRIPTION = "Upscale with a model, then resize to an exact target resolution (multiplier or fixed)."

    @staticmethod
    def _create_cache_key(image, upscale_model_name, use_fixed_resolution, output_multiplier,
                           fixed_width, fixed_height, tile_count, precision, batch_size):
        img_signature = {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "sample_hash": hashlib.md5(image[0, :8, :8, :].cpu().numpy().tobytes()).hexdigest(),
        }
        cache_dict = {
            "image_signature": img_signature, "upscale_model_name": upscale_model_name,
            "use_fixed_resolution": use_fixed_resolution, "output_multiplier": output_multiplier,
            "fixed_width": fixed_width, "fixed_height": fixed_height,
            "tile_count": tile_count, "precision": precision, "batch_size": batch_size,
        }
        return hashlib.md5(json.dumps(cache_dict, sort_keys=True).encode()).hexdigest()

    @classmethod
    def _manage_cache_size(cls):
        if len(cls._cache) > cls._max_cache_size:
            for key in list(cls._cache.keys())[: -cls._max_cache_size]:
                del cls._cache[key]

    def _determine_dtype(self, precision, device):
        if precision == "auto":
            if device.type == "cuda":
                return torch.float16
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16 if device.type == "cuda" else torch.bfloat16
        if precision == "bf16":
            return torch.bfloat16
        return torch.float32

    def _get_model_scale_factor(self, upscale_model, model_name):
        # Prefer the model's own real scale — confirmed accurate directly from
        # ComfyUI's own core (nodes_upscale_model.py uses this exact attribute).
        try:
            scale = float(getattr(upscale_model, "scale"))
            if scale > 0:
                return scale
        except (TypeError, ValueError, AttributeError):
            pass
        # Last-resort fallback only — reached if the model object is missing its
        # own scale attribute, which shouldn't normally happen.
        log.warning("[MuseUpscaleAdvanced] Model '%s' has no usable .scale attribute — "
                    "guessing from filename instead.", model_name)
        name = model_name.lower()
        if "8x" in name or "x8" in name:
            return 8.0
        if "2x" in name or "x2" in name:
            return 2.0
        return 4.0

    def _run_upscale(self, upscale_model, image, dtype, device, tile_size_px):
        """Runs the model via ComfyUI's own tiled_scale — blended tile edges, automatic
        OOM-safe tile shrinking. image is BHWC in [0,1]; returns BHWC in [0,1]."""
        in_img = image.movedim(-1, -3).to(device)
        tile = max(128, int(tile_size_px))
        overlap = min(32, max(0, tile // 4))
        output_device = model_management.intermediate_device()

        def run_tile(a):
            a = a.to(dtype)
            if dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type=device.type, dtype=dtype):
                    return upscale_model(a)
            return upscale_model(a)

        oom = True
        result = None
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                result = comfy.utils.tiled_scale(
                    in_img, run_tile, tile_x=tile, tile_y=tile, overlap=overlap,
                    upscale_amount=upscale_model.scale, pbar=pbar, output_device=output_device,
                )
                oom = False
            except Exception as e:
                model_management.raise_non_oom(e)
                tile //= 2
                if tile < 128:
                    raise e

        return torch.clamp(result.movedim(-3, -1), 0.0, 1.0)

    def upscale_advanced(self, image, upscale_model_name, use_fixed_resolution, output_multiplier,
                          fixed_width, fixed_height, tile_count, precision, batch_size,
                          offload_model, disable_cache):
        cache_key = self._create_cache_key(
            image, upscale_model_name, use_fixed_resolution, output_multiplier,
            fixed_width, fixed_height, tile_count, precision, batch_size,
        )
        if not disable_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            log.info("[MuseUpscaleAdvanced] Cache hit — reusing %dx%d result (model=%s).",
                     cached.shape[2], cached.shape[1], upscale_model_name)
            return (cached, cached.shape[2], cached.shape[1])

        device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()

        upscale_model = self.upscale_loader.load_model(upscale_model_name)[0]
        model_scale = self._get_model_scale_factor(upscale_model, upscale_model_name)

        input_h, input_w = image.shape[1], image.shape[2]
        if use_fixed_resolution:
            target_h, target_w = fixed_height, fixed_width
            log.info("[MuseUpscaleAdvanced] Input %dx%d -> target %dx%d (fixed resolution), model=%s (%.0fx native).",
                     input_w, input_h, target_w, target_h, upscale_model_name, model_scale)
        else:
            target_h = int(input_h * output_multiplier)
            target_w = int(input_w * output_multiplier)
            log.info("[MuseUpscaleAdvanced] Input %dx%d -> target %dx%d (%.2fx multiplier), model=%s (%.0fx native).",
                     input_w, input_h, target_w, target_h, output_multiplier, upscale_model_name, model_scale)

        tile_count_int = int(tile_count)
        tile_size_px = max(input_h, input_w) // tile_count_int if tile_count_int > 1 else max(input_h, input_w)
        if tile_count_int > 1:
            log.info("[MuseUpscaleAdvanced] Tiling enabled: %sx%s grid, ~%dpx starting tile size "
                     "(blended edges, auto-shrinks further on OOM).", tile_count, tile_count, tile_size_px)

        dtype = self._determine_dtype(precision, device)

        # Loading via ComfyUI's own memory manager (rather than a manual .to(device))
        # so this coordinates properly with whatever else is currently resident on
        # the GPU, instead of fighting it for VRAM.
        memory_required = (512 * 512 * 3) * image.element_size() * max(model_scale, 1.0) * 384.0
        memory_required += image.nelement() * image.element_size()
        model_management.load_models_gpu([upscale_model.patcher], memory_required=memory_required, force_full_load=True)

        try:
            image_batches = torch.split(image, batch_size)
            processed_batches = []
            for batch in image_batches:
                upscaled = self._run_upscale(upscale_model, batch, dtype, device, tile_size_px)
                if upscaled.shape[1] != target_h or upscaled.shape[2] != target_w:
                    upscaled = comfy.utils.lanczos(
                        upscaled.permute(0, 3, 1, 2), target_w, target_h
                    ).permute(0, 2, 3, 1)
                processed_batches.append(upscaled)
            full_upscaled = torch.cat(processed_batches, dim=0)
        finally:
            if offload_model:
                upscale_model.patcher.model.to(offload_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_result = torch.clamp(full_upscaled, 0, 1).to(image.dtype).to(image.device)

        if not disable_cache:
            self._cache[cache_key] = final_result.clone()
            self._manage_cache_size()

        if offload_model:
            log.info("[MuseUpscaleAdvanced] Model offloaded to save VRAM.")
        log.info("[MuseUpscaleAdvanced] Done — output %dx%d.", final_result.shape[2], final_result.shape[1])

        return (final_result, final_result.shape[2], final_result.shape[1])

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


NODE_CLASS_MAPPINGS = {"MuseUpscaleModelAdv": MuseUpscaleModelAdv}
NODE_DISPLAY_NAME_MAPPINGS = {"MuseUpscaleModelAdv": "Upscale Model Advanced (Muse)"}

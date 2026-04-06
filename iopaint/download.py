import glob
import json
import os
from functools import lru_cache
from typing import List, Optional

from iopaint.schema import ModelType, ModelInfo
from loguru import logger
from pathlib import Path

from iopaint.const import (
    DEFAULT_MODEL_DIR,
    DIFFUSERS_SD_CLASS_NAME,
    DIFFUSERS_SD_INPAINT_CLASS_NAME,
    DIFFUSERS_SDXL_CLASS_NAME,
    DIFFUSERS_SDXL_INPAINT_CLASS_NAME,
    ANYTEXT_NAME,
)
from iopaint.model.original_sd_configs import get_config_files

# Cache versioning for invalidation on format changes
CACHE_VERSION = 1

# Import custom exceptions for unified error handling
# Note: These are currently defined but not actively used to avoid breaking changes
# They can be used in future refactoring for more structured error handling

# OpenAI-compatible API constants
def cli_download_model(model: str):
    from iopaint.model import models
    from iopaint.model.utils import handle_from_pretrained_exceptions

    if model in models and models[model].is_erase_model:
        logger.info(f"Downloading {model}...")
        models[model].download()
        logger.info("Done.")
    elif model == ANYTEXT_NAME:
        logger.info(f"Downloading {model}...")
        models[model].download()
        logger.info("Done.")
    else:
        logger.info(f"Downloading model from Huggingface: {model}")
        from diffusers import DiffusionPipeline

        try:
            downloaded_path = handle_from_pretrained_exceptions(
                DiffusionPipeline.download, pretrained_model_name=model, variant="fp16"
            )
        except Exception:
            # Some newer model families (e.g. FLUX/Klein) don't ship fp16 variant tags.
            downloaded_path = handle_from_pretrained_exceptions(
                DiffusionPipeline.download, pretrained_model_name=model
            )
        logger.info(f"Done. Downloaded to {downloaded_path}")


def folder_name_to_show_name(name: str) -> str:
    return name.replace("models--", "").replace("--", "/")


@lru_cache(maxsize=512)
def get_sd_model_type(model_abs_path: str) -> Optional[ModelType]:
    filename_lower = Path(model_abs_path).name.lower()

    if "inpaint" in filename_lower:
        return ModelType.DIFFUSERS_SD_INPAINT

    if "sd" in filename_lower and "sdxl" not in filename_lower:
        if "v1" in filename_lower or "v1.5" in filename_lower or "base" in filename_lower:
            return ModelType.DIFFUSERS_SD

    from diffusers import StableDiffusionInpaintPipeline

    try:
        StableDiffusionInpaintPipeline.from_single_file(
            model_abs_path,
            load_safety_checker=False,
            num_in_channels=9,
            original_config_file=get_config_files()["v1"],
        )
        return ModelType.DIFFUSERS_SD_INPAINT
    except ValueError as e:
        if "[320, 4, 3, 3]" in str(e):
            return ModelType.DIFFUSERS_SD
        else:
            logger.debug(f"Unsupported SD model format for {model_abs_path}: {e}")
            return None
    except (OSError, PermissionError) as e:
        logger.warning(f"Cannot access model file {model_abs_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading SD model {model_abs_path}: {e}")
        return None


@lru_cache()
def get_sdxl_model_type(model_abs_path: str) -> Optional[ModelType]:
    filename_lower = model_abs_path.lower()

    if "inpaint" in filename_lower:
        return ModelType.DIFFUSERS_SDXL_INPAINT

    if "sdxl" in filename_lower or "xl" in filename_lower:
        return ModelType.DIFFUSERS_SDXL

    from diffusers import StableDiffusionXLInpaintPipeline

    try:
        model = StableDiffusionXLInpaintPipeline.from_single_file(
            model_abs_path,
            load_safety_checker=False,
            num_in_channels=9,
            original_config_file=get_config_files()["xl"],
        )
        if model.unet.config.in_channels == 9:
            return ModelType.DIFFUSERS_SDXL_INPAINT
        else:
            return ModelType.DIFFUSERS_SDXL
    except ValueError as e:
        if "[320, 4, 3, 3]" in str(e):
            return ModelType.DIFFUSERS_SDXL
        else:
            logger.debug(f"Unsupported SDXL model format for {model_abs_path}: {e}")
            return None
    except (OSError, PermissionError) as e:
        logger.warning(f"Cannot access SDXL model file {model_abs_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading SDXL model {model_abs_path}: {e}")
        return None


def scan_single_file_diffusion_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    stable_diffusion_dir = cache_dir / "stable_diffusion"
    cache_file = stable_diffusion_dir / "iopaint_cache.json"
    model_type_cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert isinstance(data, dict)
                # Check cache version for invalidation
                if data.get("_version") == CACHE_VERSION:
                    model_type_cache = data.get("_models", {})
                else:
                    logger.info(f"Cache version mismatch (expected {CACHE_VERSION}, got {data.get('_version')}), clearing cache")
                    model_type_cache = {}
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            model_type_cache = {}

    res = []
    for it in stable_diffusion_dir.glob("*.*"):
        if it.suffix not in [".safetensors", ".ckpt"]:
            continue
        model_abs_path = str(it.absolute())
        model_type = model_type_cache.get(it.name)
        if model_type is None:
            model_type = get_sd_model_type(model_abs_path)
        if model_type is None:
            continue

        model_type_cache[it.name] = model_type
        res.append(
            ModelInfo(
                name=it.name,
                path=model_abs_path,
                model_type=model_type,
                is_single_file_diffusers=True,
            )
        )
    if stable_diffusion_dir.exists():
        with open(cache_file, "w", encoding="utf-8") as fw:
            cache_data = {
                "_version": CACHE_VERSION,
                "_models": model_type_cache
            }
            json.dump(cache_data, fw, indent=2, ensure_ascii=False)

    stable_diffusion_xl_dir = cache_dir / "stable_diffusion_xl"
    sdxl_cache_file = stable_diffusion_xl_dir / "iopaint_cache.json"
    sdxl_model_type_cache = {}
    if sdxl_cache_file.exists():
        try:
            with open(sdxl_cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert isinstance(data, dict)
                # Check cache version for invalidation
                if data.get("_version") == CACHE_VERSION:
                    sdxl_model_type_cache = data.get("_models", {})
                else:
                    logger.info(f"SDXL cache version mismatch (expected {CACHE_VERSION}, got {data.get('_version')}), clearing cache")
                    sdxl_model_type_cache = {}
        except Exception as e:
            logger.warning(f"SDXL cache read error: {e}")
            sdxl_model_type_cache = {}

    for it in stable_diffusion_xl_dir.glob("*.*"):
        if it.suffix not in [".safetensors", ".ckpt"]:
            continue
        model_abs_path = str(it.absolute())
        model_type = sdxl_model_type_cache.get(it.name)
        if model_type is None:
            model_type = get_sdxl_model_type(model_abs_path)
        if model_type is None:
            continue

        sdxl_model_type_cache[it.name] = model_type
        if stable_diffusion_xl_dir.exists():
            with open(sdxl_cache_file, "w", encoding="utf-8") as fw:
                cache_data = {
                    "_version": CACHE_VERSION,
                    "_models": sdxl_model_type_cache
                }
                json.dump(cache_data, fw, indent=2, ensure_ascii=False)

        res.append(
            ModelInfo(
                name=it.name,
                path=model_abs_path,
                model_type=model_type,
                is_single_file_diffusers=True,
            )
        )
    return res


def scan_inpaint_models(model_dir: Path) -> List[ModelInfo]:
    res = []
    from iopaint.model import models

    # logger.info(f"Scanning inpaint models in {model_dir}")

    for name, m in models.items():
        if m.is_erase_model and m.is_downloaded():
            res.append(
                ModelInfo(
                    name=name,
                    path=name,
                    model_type=ModelType.INPAINT,
                )
            )
        elif not m.is_erase_model and m.is_downloaded():
            model_type = ModelType.UNKNOWN
            res.append(
                ModelInfo(
                    name=name,
                    path=name,
                    model_type=model_type,
                )
            )
    return res


def scan_diffusers_models() -> List[ModelInfo]:
    from huggingface_hub.constants import HF_HUB_CACHE

    available_models = []
    cache_dir = Path(HF_HUB_CACHE)
    # logger.info(f"Scanning diffusers models in {cache_dir}")
    diffusers_model_names = []
    model_index_files = glob.glob(
        os.path.join(cache_dir, "**/*", "model_index.json"), recursive=True
    )
    for it in model_index_files:
        it = Path(it)
        try:
            with open(it, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read model index {it}: {e}")
            continue

        _class_name = data["_class_name"]
        name = folder_name_to_show_name(it.parent.parent.parent.name)
        if name in diffusers_model_names:
            continue
        if "PowerPaint" in name:
            model_type = ModelType.DIFFUSERS_OTHER
        elif _class_name == DIFFUSERS_SD_CLASS_NAME:
            model_type = ModelType.DIFFUSERS_SD
        elif _class_name == DIFFUSERS_SD_INPAINT_CLASS_NAME:
            model_type = ModelType.DIFFUSERS_SD_INPAINT
        elif _class_name == DIFFUSERS_SDXL_CLASS_NAME:
            model_type = ModelType.DIFFUSERS_SDXL
        elif _class_name == DIFFUSERS_SDXL_INPAINT_CLASS_NAME:
            model_type = ModelType.DIFFUSERS_SDXL_INPAINT
        elif _class_name in [
            "StableDiffusionInstructPix2PixPipeline",
            "PaintByExamplePipeline",
            "KandinskyV22InpaintPipeline",
            "AnyText",
            "Flux2KleinPipeline",
        ]:
            model_type = ModelType.DIFFUSERS_OTHER
        else:
            continue

        diffusers_model_names.append(name)
        available_models.append(
            ModelInfo(
                name=name,
                path=name,
                model_type=model_type,
            )
        )
    return available_models


def _scan_converted_diffusers_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    available_models = []
    diffusers_model_names = []
    model_index_files = glob.glob(
        os.path.join(cache_dir, "**/*", "model_index.json"), recursive=True
    )
    for it in model_index_files:
        it = Path(it)
        with open(it, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.error(
                    f"Failed to load {it}: {e}. Please try revert from original model or fix model_index.json by hand."
                )
                continue

            _class_name = data["_class_name"]
            name = folder_name_to_show_name(it.parent.name)
            if name in diffusers_model_names:
                continue
            elif _class_name == DIFFUSERS_SD_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SD
            elif _class_name == DIFFUSERS_SD_INPAINT_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SD_INPAINT
            elif _class_name == DIFFUSERS_SDXL_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SDXL
            elif _class_name == DIFFUSERS_SDXL_INPAINT_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SDXL_INPAINT
            else:
                continue

            diffusers_model_names.append(name)
            available_models.append(
                ModelInfo(
                    name=name,
                    path=str(it.parent.absolute()),
                    model_type=model_type,
                )
            )
    return available_models


def scan_converted_diffusers_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    available_models = []
    stable_diffusion_dir = cache_dir / "stable_diffusion"
    stable_diffusion_xl_dir = cache_dir / "stable_diffusion_xl"
    available_models.extend(_scan_converted_diffusers_models(stable_diffusion_dir))
    available_models.extend(_scan_converted_diffusers_models(stable_diffusion_xl_dir))
    return available_models


def scan_plugin_models() -> List[ModelInfo]:
    """Scan models provided by plugins."""
    res = []

    # Import plugin classes directly to avoid circular imports
    try:
        from iopaint.plugins.realesrgan import RealESRGANUpscaler
        plugin_instance = RealESRGANUpscaler("realesr-general-x4v3", "cpu", no_half=True)
        for model_info in plugin_instance.available_models:
            res.append(ModelInfo(
                name=model_info["name"],
                path=model_info.get("path", model_info["name"]),
                model_type=ModelType.PLUGIN,
                plugin_name="RealESRGAN",
            ))
    except Exception as e:
        logger.debug(f"Failed to scan RealESRGAN models: {e}")

    try:
        from iopaint.plugins.gfpgan_plugin import GFPGANPlugin
        plugin_instance = GFPGANPlugin("cpu")
        for model_info in plugin_instance.available_models:
            res.append(ModelInfo(
                name=model_info["name"],
                path=model_info.get("path", model_info["name"]),
                model_type=ModelType.PLUGIN,
                plugin_name="GFPGAN",
            ))
    except Exception as e:
        logger.debug(f"Failed to scan GFPGAN models: {e}")

    try:
        from iopaint.plugins.restoreformer import RestoreFormerPlugin
        plugin_instance = RestoreFormerPlugin("cpu")
        for model_info in plugin_instance.available_models:
            res.append(ModelInfo(
                name=model_info["name"],
                path=model_info.get("path", model_info["name"]),
                model_type=ModelType.PLUGIN,
                plugin_name="RestoreFormer",
            ))
    except Exception as e:
        logger.debug(f"Failed to scan RestoreFormer models: {e}")

    try:
        from iopaint.plugins.remove_bg import RemoveBG
        plugin_instance = RemoveBG("u2net", "cpu")
        for model_info in plugin_instance.available_models:
            res.append(ModelInfo(
                name=model_info["name"],
                path=model_info.get("path", model_info["name"]),
                model_type=ModelType.PLUGIN,
                plugin_name="RemoveBG",
            ))
    except Exception as e:
        logger.debug(f"Failed to scan RemoveBG models: {e}")

    try:
        from iopaint.plugins.interactive_seg import InteractiveSeg
        from iopaint.schema import InteractiveSegModel
        plugin_instance = InteractiveSeg(InteractiveSegModel.vit_b, "cpu")
        for model_info in plugin_instance.available_models:
            res.append(ModelInfo(
                name=model_info["name"],
                path=model_info.get("path", model_info["name"]),
                model_type=ModelType.PLUGIN,
                plugin_name="InteractiveSeg",
            ))
    except Exception as e:
        logger.debug(f"Failed to scan InteractiveSeg models: {e}")

    return res


def download_curated_models():
    """Auto-download curated models if they are not present."""
    import torch
    from iopaint.config import get_config
    config = get_config()
    logger.info("Checking and downloading curated models...")
    
    for model_name in config.curated_models:
        try:
            if model_name == "lama":
                from iopaint.model.lama import LaMa
                if not LaMa.is_downloaded():
                    logger.info(f"Downloading {model_name}...")
                    LaMa.download()
            elif model_name in ["u2net", "birefnet-general-lite"]:
                logger.info(f"Ensuring {model_name} is available...")
                from iopaint.plugins.remove_bg import RemoveBG
                # Initializing the plugin triggers download if missing
                RemoveBG(model_name, device=torch.device("cpu"))
            elif model_name in ["mobile_sam", "sam2_tiny"]:
                logger.info(f"Ensuring {model_name} is available...")
                from iopaint.plugins.interactive_seg import InteractiveSeg
                InteractiveSeg(model_name, device=torch.device("cpu"))
                
        except Exception as e:
            logger.warning(f"Failed to auto-download {model_name}: {e}")


def scan_models() -> List[ModelInfo]:
    from iopaint.config import get_config
    config = get_config()
    model_dir = Path(os.getenv("XDG_CACHE_HOME", DEFAULT_MODEL_DIR))
    all_models = []
    all_models.extend(scan_inpaint_models(model_dir))
    all_models.extend(scan_single_file_diffusion_models(model_dir))
    all_models.extend(scan_diffusers_models())
    all_models.extend(scan_converted_diffusers_models(model_dir))
    all_models.extend(scan_plugin_models())

    # Filter by curated list
    available_models = [m for m in all_models if m.name in config.curated_models]

    # Ensure LaMa is always available (even if not downloaded)
    if not any(m.name == "lama" for m in available_models):
        available_models.insert(0, ModelInfo(
            name="lama",
            path="lama",
            model_type=ModelType.INPAINT,
        ))

    return available_models

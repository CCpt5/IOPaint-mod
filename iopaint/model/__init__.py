from typing import TYPE_CHECKING, Dict, Type

if TYPE_CHECKING:
    from iopaint.model.base import InpaintModel

_models_cache: Dict[str, Type["InpaintModel"]] | None = None
_controlnet_cache: Type["ControlNet"] | None = None
_sd_cache: Type["SD"] | None = None
_sdxl_cache: Type["SDXL"] | None = None


def _load_models() -> Dict[str, Type["InpaintModel"]]:
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    _models_cache = {}

    from .anytext.anytext_model import AnyText
    from .fcf import FcF
    from .flux2_klein import Flux2Klein
    from .instruct_pix2pix import InstructPix2Pix
    from .kandinsky import Kandinsky22
    from .lama import LaMa, AnimeLaMa
    from .ldm import LDM
    from .manga import Manga
    from .mat import MAT
    from .mi_gan import MIGAN
    from .opencv2 import OpenCV2
    from .paint_by_example import PaintByExample
    from .power_paint.power_paint import PowerPaint
    from .sd import SD15, SD2, Anything4, RealisticVision14, SD
    from .sdxl import SDXL
    from .zits import ZITS

    _models_cache.update({
        LaMa.name: LaMa,
        AnimeLaMa.name: AnimeLaMa,
        LDM.name: LDM,
        ZITS.name: ZITS,
        MAT.name: MAT,
        FcF.name: FcF,
        Flux2Klein.name: Flux2Klein,
        OpenCV2.name: OpenCV2,
        Manga.name: Manga,
        MIGAN.name: MIGAN,
        SD15.name: SD15,
        Anything4.name: Anything4,
        RealisticVision14.name: RealisticVision14,
        SD2.name: SD2,
        PaintByExample.name: PaintByExample,
        InstructPix2Pix.name: InstructPix2Pix,
        Kandinsky22.name: Kandinsky22,
        SDXL.name: SDXL,
        PowerPaint.name: PowerPaint,
        AnyText.name: AnyText,
    })

    return _models_cache


def _get_controlnet() -> Type["ControlNet"]:
    global _controlnet_cache
    if _controlnet_cache is not None:
        return _controlnet_cache
    from .controlnet import ControlNet
    _controlnet_cache = ControlNet
    return _controlnet_cache


def _get_sd() -> Type["SD"]:
    global _sd_cache
    if _sd_cache is not None:
        return _sd_cache
    from .sd import SD
    _sd_cache = SD
    return _sd_cache


def _get_sdxl() -> Type["SDXL"]:
    global _sdxl_cache
    if _sdxl_cache is not None:
        return _sdxl_cache
    from .sdxl import SDXL
    _sdxl_cache = SDXL
    return _sdxl_cache


def __getattr__(name: str):
    if name == "models":
        return _load_models()
    if name == "ControlNet":
        return _get_controlnet()
    if name == "SD":
        return _get_sd()
    if name == "SDXL":
        return _get_sdxl()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_model(name: str):
    models = _load_models()
    if name not in models:
        raise KeyError(f"Model {name!r} not found. Available models: {list(models.keys())}")
    return models[name]


if TYPE_CHECKING:
    from .controlnet import ControlNet as ControlNet
    from .sd import SD as SD
    from .sdxl import SDXL as SDXL
    from .base import InpaintModel as InpaintModel

    models: Dict[str, Type[InpaintModel]]

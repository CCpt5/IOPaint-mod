import os
from pathlib import Path
from typing import Optional, Dict
from omegaconf import OmegaConf
from loguru import logger
from iopaint.schema import GlobalConfig, ModelManifest, ServerSettings, PluginSettings, Device
from iopaint.const import (
    DEFAULT_DIFFUSION_MODELS,
    SD_CONTROLNET_CHOICES,
    SD2_CONTROLNET_CHOICES,
    SDXL_CONTROLNET_CHOICES,
    SD_BRUSHNET_CHOICES,
    SDXL_BRUSHNET_CHOICES,
)

DEFAULT_CONFIG_PATH = Path("config.yaml")
LEGACY_CURATED_MODELS = ["lama", "u2net", "birefnet-general-lite", "mobile_sam", "sam2_tiny"]

# Built-in default models
DEFAULT_MODELS = [
    ModelManifest(
        name="lama",
        url="https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        md5="e3aa4aaa15225a33ec84f9f4bc47e500",
        version="1.0.0",
        version_url="https://api.github.com/repos/Sanster/models/releases/latest",
        is_erase_model=True,
        supported_devices=["cuda", "cpu"]
    ),
    ModelManifest(
        name="anime-lama",
        url="https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt",
        md5="29f284f36a0a510bcacf39ecf4c4d54f",
        version="1.0.0",
        version_url="https://api.github.com/repos/Sanster/models/releases/latest",
        is_erase_model=True,
        supported_devices=["cuda", "cpu"]
    ),
    ModelManifest(
        name="mat",
        url="https://github.com/Sanster/models/releases/download/add_mat/MAT_768_full.pt",
        md5="3d332616239bc7a61d15383569a93077",
        version="1.0.0",
        is_erase_model=True,
        supported_devices=["cuda", "cpu"]
    ),
    ModelManifest(
        name="fcf",
        url="https://github.com/Sanster/models/releases/download/add_fcf/fcf_512.pt",
        md5="87265a7d771d490c0a560f6448e89f64",
        version="1.0.0",
        is_erase_model=True,
        supported_devices=["cuda", "cpu"]
    ),
    ModelManifest(
        name="migan",
        url="https://github.com/Sanster/models/releases/download/migan/migan_512_face.pt",
        md5="79466f2c0022a969623e10037a1f5926",
        version="1.0.0",
        is_erase_model=True,
        supported_devices=["cuda", "cpu"]
    ),
    ModelManifest(
        name="zits",
        url="https://github.com/Sanster/models/releases/download/add_zits/zits-inpaint-0717.pt",
        md5="9978cc7157dc29699e42308d675b2154",
        version="1.0.0",
        is_erase_model=True,
        supported_devices=["cuda", "cpu"],
        extra_models={
            "edge_line": {
                "url": "https://github.com/Sanster/models/releases/download/add_zits/zits-edge-line-0717.pt",
                "md5": "55e31af21ba96bbf0c80603c76ea8c5f"
            },
            "structure_upsample": {
                "url": "https://github.com/Sanster/models/releases/download/add_zits/zits-structure-upsample-0717.pt",
                "md5": "3d88a07211bd41b2ec8cc0d999f29927"
            },
            "wireframe": {
                "url": "https://github.com/Sanster/models/releases/download/add_zits/zits-wireframe-0717.pt",
                "md5": "a9727c63a8b48b65c905d351b21ce46b"
            }
        }
    )
]

class ConfigManager:
    _instance: Optional['ConfigManager'] = None
    _config: Optional[GlobalConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def load(self, config_path: Optional[Path] = None) -> GlobalConfig:
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        if config_path.exists():
            logger.info(f"Loading config from {config_path}")
            conf = OmegaConf.load(config_path)
            # Convert OmegaConf to dict and then to Pydantic
            config_dict = OmegaConf.to_container(conf, resolve=True)
            self._config = GlobalConfig(**config_dict)
            if self._config.curated_models == LEGACY_CURATED_MODELS:
                self._config.curated_models = [*LEGACY_CURATED_MODELS, *DEFAULT_DIFFUSION_MODELS]
                logger.info(
                    "Migrated curated model list to include default diffusion models."
                )
        else:
            logger.info("Config file not found, using defaults")
            self._config = GlobalConfig()
            # Ensure default models are present if none specified
            if not self._config.models:
                self._config.models = DEFAULT_MODELS
            
            # Load diffusion and controlnet defaults from const
            self._config.diffusion_models = DEFAULT_DIFFUSION_MODELS
            self._config.controlnet_models = {
                "sd1.5": SD_CONTROLNET_CHOICES,
                "sd2": SD2_CONTROLNET_CHOICES,
                "sdxl": SDXL_CONTROLNET_CHOICES,
            }
            self._config.brushnet_models = {
                "sd1.5": SD_BRUSHNET_CHOICES,
                "sdxl": SDXL_BRUSHNET_CHOICES,
            }
            self._config.lcm_lora_models = {
                "sd1.5": "latent-consistency/lcm-lora-sdv1-5",
                "sdxl": "latent-consistency/lcm-lora-sdxl"
            }

        return self._config

    @property
    def config(self) -> GlobalConfig:
        if self._config is None:
            self.load()
        return self._config

    def save_default(self, path: Path = DEFAULT_CONFIG_PATH):
        """Save a default configuration file if it doesn't exist."""
        if not path.exists():
            config = GlobalConfig(
                models=DEFAULT_MODELS,
                diffusion_models=DEFAULT_DIFFUSION_MODELS,
                controlnet_models={
                    "sd1.5": SD_CONTROLNET_CHOICES,
                    "sd2": SD2_CONTROLNET_CHOICES,
                    "sdxl": SDXL_CONTROLNET_CHOICES,
                },
                brushnet_models={
                    "sd1.5": SD_BRUSHNET_CHOICES,
                    "sdxl": SDXL_BRUSHNET_CHOICES,
                },
                lcm_lora_models={
                    "sd1.5": "latent-consistency/lcm-lora-sdv1-5",
                    "sdxl": "latent-consistency/lcm-lora-sdxl"
                }
            )
            # Use Pydantic's dict export then OmegaConf for YAML saving
            config_dict = config.model_dump()
            conf = OmegaConf.create(config_dict)
            OmegaConf.save(config=conf, f=path)
            logger.info(f"Created default config at {path}")

def get_config() -> GlobalConfig:
    return ConfigManager().config

def load_config(path: Optional[Path] = None) -> GlobalConfig:
    return ConfigManager().load(path)

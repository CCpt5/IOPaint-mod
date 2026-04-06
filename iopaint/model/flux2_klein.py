from inspect import signature

import cv2
import numpy as np
import PIL.Image
import torch
from loguru import logger

from iopaint.const import FLUX2_KLEIN_9B_NAME
from iopaint.schema import InpaintRequest

from .base import DiffusionInpaintModel
from .utils import enable_low_mem, is_local_files_only


class Flux2Klein(DiffusionInpaintModel):
    """Mask-guided wrapper around Flux2KleinPipeline image-editing."""

    name = FLUX2_KLEIN_9B_NAME
    pad_mod = 8
    min_size = 512

    def init_model(self, device: torch.device, **kwargs):
        from diffusers import Flux2KleinPipeline

        use_gpu = device.type == "cuda"
        torch_dtype = torch.bfloat16 if use_gpu and not kwargs.get("no_half", False) else torch.float32

        self.model = Flux2KleinPipeline.from_pretrained(
            self.model_id_or_path,
            torch_dtype=torch_dtype,
            local_files_only=is_local_files_only(**kwargs),
        )

        enable_low_mem(self.model, kwargs.get("low_mem", False))

        if kwargs.get("cpu_offload", False) and use_gpu:
            logger.info("Enable sequential cpu offload")
            self.model.enable_sequential_cpu_offload(gpu_id=0)
        else:
            self.model = self.model.to(device)

        self.callback = kwargs.pop("callback", None)

    @staticmethod
    def is_downloaded() -> bool:
        return False

    def forward(self, image, mask, config: InpaintRequest):
        self.set_scheduler(config)

        img_h, img_w = image.shape[:2]
        pipe_call = {
            "prompt": config.prompt,
            "image": PIL.Image.fromarray(image),
            "num_inference_steps": config.sd_steps,
            "guidance_scale": config.sd_guidance_scale,
            "generator": torch.manual_seed(config.sd_seed),
            "output_type": "np",
            "height": img_h,
            "width": img_w,
        }

        pipe_sig = signature(self.model.__call__).parameters
        if "negative_prompt" in pipe_sig:
            pipe_call["negative_prompt"] = config.negative_prompt
        if "strength" in pipe_sig:
            pipe_call["strength"] = config.sd_strength
        if "callback_on_step_end" in pipe_sig:
            pipe_call["callback_on_step_end"] = self.callback

        output = self.model(**pipe_call).images[0]

        output = (output * 255).round().astype("uint8")

        if mask.ndim == 3:
            mask_2d = mask[:, :, -1]
        else:
            mask_2d = mask
        mask_norm = (mask_2d.astype(np.float32) / 255.0)[..., None]

        blended = (output.astype(np.float32) * mask_norm + image.astype(np.float32) * (1.0 - mask_norm)).round().astype("uint8")
        return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

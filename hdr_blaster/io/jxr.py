"""JXR loading (moved verbatim from NHC.py)."""
import logging

import imagecodecs
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..engine.metadata import HDRMetadata
from ..engine.tonemap import AdvancedToneMapper


def read_capture(path, device):
    """Read an unadjusted scRGB capture with metadata belonging to this file alone."""
    from .jxr_info import check_hdr_capture, read_jxr_info

    check_hdr_capture(read_jxr_info(path))
    loader = OptimizedJXRLoader(device)
    linear, error, _ = loader.load_jxr(str(path))
    if linear is None:
        raise ValueError(f"Could not read this capture: {error}")
    return linear, loader


class OptimizedJXRLoader:
    """
    Handles loading and processing of JXR images.
    This class includes tone mapping operations for preview and final processing.
    """

    def __init__(self, device):
        self.device = device
        self.hdr_peak_luminance = 10000.0
        self.selected_pre_gamma = 1.0
        self.selected_auto_exposure = 1.0
        self.metadata = HDRMetadata()
        self.tone_mapper = None

    def _apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Safely applies gamma correction to the input image array.
        """
        return np.copysign(np.power(np.abs(image), gamma), image)

    def load_jxr(self, file_path: str):
        """
        Loads and preprocesses a JXR image, returning a linear HDR tensor.
        Tone mapping is NOT applied here.
        """
        try:
            with open(file_path, 'rb') as f:
                jxr_data = f.read()

            # Extract metadata
            self.metadata.extract_from_jxr(jxr_data)

            # Decode JXR
            try:
                logging.debug(f"Attempting to decode JXR file: {file_path}")
                image = imagecodecs.jpegxr_decode(jxr_data)
                if not isinstance(image, np.ndarray):
                    raise ValueError(f"Unexpected decode result type: {type(image)}")
                logging.debug(f"Successfully decoded image with shape: {image.shape}, dtype: {image.dtype}")
            except Exception as decode_error:
                logging.error(f"Image decode failed: {decode_error}", exc_info=True)
                raise ValueError(f"Failed to decode JXR: {decode_error}")

            if image is None:
                raise ValueError("Failed to decode JXR image")
            image = image.astype(np.float32)

            # Update peak luminance from metadata if available
            if self.metadata.has_metadata and self.metadata.max_luminance > 0 and not np.isnan(
                    self.metadata.max_luminance):
                self.hdr_peak_luminance = self.metadata.max_luminance

            pre_gamma = self.selected_pre_gamma
            auto_exposure = self.selected_auto_exposure

            # Apply gamma correction
            if pre_gamma != 1.0:
                gamma = 1.0 / pre_gamma
                image = self._apply_gamma_correction(image, gamma)

            # Apply exposure correction
            if auto_exposure != 1.0:
                image *= auto_exposure

            # Scale image to display range
            display_peak = 1000.0
            if self.metadata.has_metadata and self.metadata.white_point > 0 and not np.isnan(self.metadata.white_point):
                display_peak = self.metadata.white_point
            image = image * (display_peak / self.hdr_peak_luminance)

            # Handle grayscale or RGBA
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            elif image.shape[2] > 4:
                image = image[:, :, :3]

            # Replace NaNs and inf
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

            # Convert to torch Tensor
            tensor = torch.from_numpy(image).float().permute(2, 0, 1).contiguous().unsqueeze(0)
            tensor = tensor.to(self.device, non_blocking=True)

            # Create tone mapper with metadata for later use
            if self.tone_mapper is None:
                self.tone_mapper = AdvancedToneMapper(self.device, self.metadata)

            return tensor, None, self.metadata.to_dict()
        except Exception as e:
            logging.error(f"Failed to load HDR image: {e}", exc_info=True)
            return None, str(e), {}

    def linear_to_srgb(self, linear_rgb):
        """Converts linear RGB to sRGB."""
        linear_rgb = torch.clamp(linear_rgb, 0.0, 1.0)
        a = 0.055
        gamma = 2.4
        srgb = torch.where(
            linear_rgb <= 0.0031308,
            12.92 * linear_rgb,
            (1 + a) * torch.pow(linear_rgb, 1.0 / gamma) - a
        )
        return torch.clamp(srgb, 0.0, 1.0)

    def process_preview(self, tensor_data, target_width: int, target_height: int):
        """
        Generates a lower-resolution preview from the loaded tensor.
        """
        try:
            tensor = tensor_data[0] if isinstance(tensor_data, tuple) else tensor_data
            if tensor is None:
                return None
            _, _, h, w = tensor.shape
            width_ratio = target_width / w
            height_ratio = target_height / h
            scale_factor = min(width_ratio, height_ratio)
            new_width = int(w * scale_factor)
            new_height = int(h * scale_factor)
            with torch.inference_mode():
                preview_tensor = F.interpolate(tensor, size=(new_height, new_width),
                                               mode='bicubic', align_corners=False, antialias=True)
            return torch.clamp(preview_tensor, 0.0, 1.0)
        except Exception as e:
            logging.error(f"Preview generation failed: {str(e)}")
            return None

    def tensor_to_pil(self, tensor: torch.Tensor):
        """
        Converts a tensor (0-1 range in float) to a PIL Image.
        """
        try:
            if tensor.is_cuda:
                tensor = tensor.cpu()
            tensor = tensor.float()  # Convert to float32 if not already
            tensor = torch.clamp(tensor, 0.0, 1.0)
            tensor = (tensor * 255).byte()
            img_array = tensor.squeeze(0).permute(1, 2, 0).numpy()
            return Image.fromarray(img_array)
        except Exception as e:
            logging.error(f"Tensor to PIL conversion failed: {e}")
            return None

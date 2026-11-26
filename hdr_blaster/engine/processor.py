"""Colour and edge enhancement, image writing (moved verbatim from NHC.py)."""
import logging

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image

from .enhance import EdgeEnhancementBlock
from .lut3d import ImageAdaptive3DLUT
from .params import MissingRenderParam
from .tonemap import AdvancedToneMapper


class HDRColorProcessor:
    """
    Processes HDR images with optional color correction and edge enhancements.
    Manages GPU/CPU usage and half-precision toggling if desired.
    """

    def __init__(self, device, jxr_loader, use_fp16=False):
        self.device = device
        self.jxr_loader = jxr_loader
        self.use_fp16 = use_fp16
        self.color_net = ImageAdaptive3DLUT().eval().to(device)

        if device.type == 'cuda' and self.use_fp16:
            self.color_net.half()
        elif device.type == 'cpu' and self.use_fp16:
            try:
                self.color_net.half()
            except Exception as e:
                logging.error(f"Failed to convert model to FP16 on CPU: {e}")
                self.use_fp16 = False

        self.edge_enhancement = EdgeEnhancementBlock().to(device)
        if device.type == 'cuda' and self.use_fp16:
            self.edge_enhancement.half()
        elif device.type == 'cpu' and self.use_fp16:
            try:
                self.edge_enhancement.half()
            except Exception as e:
                logging.error(f"Failed to convert EdgeEnhancementBlock to FP16 on CPU: {e}")
                self.use_fp16 = False

    def clear_gpu_memory(self):
        """Clears GPU memory cache if on CUDA."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def process_image(self, original_tensor, color_strength=0.0, edge_strength=0.0, use_enhancement=True,
                      params=None, keep_levels=False):
        """
        Process an HDR image tensor with color and edge enhancements, returning the processed tensor.
        """
        try:
            tensor = original_tensor.to(self.device, dtype=torch.float16 if self.use_fp16 else torch.float32)

            pad_size = 32
            padded_tensor = F.pad(tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

            with torch.inference_mode():
                if use_enhancement and color_strength > 0 and self.color_net.enabled:
                    lut_output = self.color_net(padded_tensor, params=params)
                    alpha = color_strength / 100.0
                    # Luminance-aware blend: the FiveK LUT lifts shadows and may
                    # shift highlights — both undesirable on dramatic HDR captures.
                    # We protect deep shadows (lum < ~0.05) and specular highlights
                    # (lum > ~0.98) by smoothly attenuating the LUT effect there,
                    # leaving full effect on midtones where retouching makes sense.
                    lum = (0.2126 * padded_tensor[:, 0:1]
                           + 0.7152 * padded_tensor[:, 1:2]
                           + 0.0722 * padded_tensor[:, 2:3])
                    shadow_ramp = ((lum - 0.05) / 0.25).clamp(0.0, 1.0)
                    shadow_ramp = shadow_ramp * shadow_ramp * (3.0 - 2.0 * shadow_ramp)
                    highlight_ramp = (1.0 - (lum - 0.85) / 0.13).clamp(0.0, 1.0)
                    highlight_ramp = highlight_ramp * highlight_ramp * (3.0 - 2.0 * highlight_ramp)
                    tone_mask = shadow_ramp * highlight_ramp
                    effective_alpha = alpha * tone_mask
                    enhanced = padded_tensor * (1.0 - effective_alpha) + lut_output * effective_alpha
                else:
                    enhanced = padded_tensor

                if edge_strength > 0:
                    enhanced = self.edge_enhancement(enhanced, edge_strength=edge_strength, params=params,
                                                     keep_levels=keep_levels)

                enhanced = enhanced[:, :, pad_size:-pad_size, pad_size:-pad_size]

                if self.device.type == 'cpu' or (self.device.type == 'cuda' and self.use_fp16):
                    enhanced = enhanced.float()

                self.clear_gpu_memory()
                return enhanced

        except MissingRenderParam:
            # A replay asked for a value the analysis never recorded: returning the
            # unprocessed image here would hide the mistake. Let it surface.
            raise
        except Exception as e:
            if params is not None and params.frozen:
                # Replaying recorded parameters (video): falling back to the
                # unprocessed image would silently produce a frame that does not
                # match its neighbours. A photo keeps the original fallback below.
                raise
            logging.error(f"Processing failed: {str(e)}", exc_info=True)
            return original_tensor

    def save_tensor_as_image(self, tensor_to_save, output_path, output_format, is_hdr_data=False, quality=95):
        """
        Saves a tensor as an image file (JPEG or TIFF).
        Applies adaptive ordered dithering before quantization for LDR formats
        to prevent banding.
        """
        try:
            if tensor_to_save.is_cuda:
                tensor_to_save = tensor_to_save.cpu()

            tensor_to_save = tensor_to_save.detach()

            # --- Start of dithering logic ---
            # Apply dithering only to LDR (non-HDR) images that will be quantized.
            if not is_hdr_data:
                dither_strength = 0.0
                if output_format == 'JPEG':
                    # Dithering intensity is adjusted for 8-bit target.
                    dither_strength = 1.0 / 255.0
                elif output_format == 'TIFF':
                    # Dithering intensity is adjusted for 16-bit target.
                    dither_strength = 1.0 / 65535.0

                # If an intensity has been defined, apply dithering.
                if dither_strength > 0 and self.jxr_loader.tone_mapper is not None:
                    logging.info(
                        f"Applying dithering for {output_format} format with intensity {dither_strength:.6f}")
                    tensor_to_save = self.jxr_loader.tone_mapper.apply_dithering(tensor_to_save,
                                                                                 strength=dither_strength)
            # --- End of dithering logic ---

            if output_format == 'JPEG':
                # Convert tensor (potentially dithered) to NumPy array.
                array = tensor_to_save.squeeze(0).permute(1, 2, 0).contiguous().numpy()
                array = np.clip(array, 0.0, 1.0)
                # Round (not truncate) to 8-bit; the TPDF dither above already
                # broke the banding. Save 4:4:4 (subsampling=0): JPEG's default
                # 4:2:0 chroma subsampling re-bands smooth colour gradients.
                array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
                result_image = Image.fromarray(array)
                result_image.save(output_path, 'JPEG', quality=max(quality, 97),
                                  subsampling=0, optimize=True)

            elif output_format == 'TIFF':
                # Convert tensor (potentially dithered) to NumPy array.
                array = tensor_to_save.squeeze(0).permute(1, 2, 0).contiguous().numpy()

                if is_hdr_data:
                    # Save raw HDR data as float32, without dithering or quantization.
                    array = np.ascontiguousarray(array, dtype=np.float32)
                    tifffile.imwrite(output_path, array, photometric='rgb', compression='lzw')
                else:
                    # For LDR data, 16-bit quantization (rounded).
                    array = np.clip(array, 0.0, 1.0)
                    array = np.clip(np.rint(array * 65535.0), 0, 65535).astype(np.uint16)
                    tifffile.imwrite(output_path, array, photometric='rgb', compression='lzw')

            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            logging.info(f"Image saved successfully: {output_path}")

        except Exception as e:
            logging.error(f"Failed to save image to {output_path}: {e}", exc_info=True)
            raise

    def switch_device(self, use_gpu, use_fp16=False):
        """
        Switches the processing device and updates model data types accordingly.
        """
        new_device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

        if new_device.type != 'cuda':
            use_fp16 = False

        if new_device != self.device or use_fp16 != self.use_fp16:
            self.clear_gpu_memory()
            self.device = new_device
            self.use_fp16 = use_fp16

            desired_dtype = torch.float16 if (self.device.type == 'cuda' and self.use_fp16) else torch.float32

            try:
                self.color_net = self.color_net.to(device=self.device, dtype=desired_dtype)
                self.edge_enhancement = self.edge_enhancement.to(device=self.device, dtype=desired_dtype)

                self.jxr_loader.tone_mapper = AdvancedToneMapper(self.device, self.jxr_loader.metadata)

            except Exception as e:
                logging.error(f"Failed to switch device or precision: {e}")
                return False

            logging.info(f"Switched to {self.device} with {'FP16' if self.use_fp16 else 'FP32'} precision")
            return True
        else:
            logging.info(f"Already on {self.device} with {'FP16' if self.use_fp16 else 'FP32'} precision")
            return False

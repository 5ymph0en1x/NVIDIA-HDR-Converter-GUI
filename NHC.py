import os
import sys
import gc
import traceback
import io
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import TKinterModernThemes as TKMT
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging
import matplotlib
import numpy as np
import imagecodecs


# Logging Configuration
logging.basicConfig(
    filename='hdr_converter.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Use 'Agg' backend for matplotlib to prevent GUI issues
matplotlib.use('Agg')

# Constants
DEFAULT_TONE_MAP = "adaptive"
DEFAULT_PREGAMMA = "1.0"
DEFAULT_AUTOEXPOSURE = "1.0"
SUPPORTED_TONE_MAPS = {"adaptive", "hable", "reinhard", "filmic", "aces", "uncharted2"}

PRETRAINED_MODELS = {
    'vgg': models.vgg16,
    'resnet': models.resnet34,
    'densenet': models.densenet121
}

MODES = {
    'big': {
        'window_width': 2200,
        'window_height': 850,
        'preview_width': 720,
        'preview_height': 406,
    },
    'small': {
        'window_width': 1760,
        'window_height': 840,
        'preview_width': 512,
        'preview_height': 288,
    },
}

# Set PyTorch backend configurations for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class DeviceManager:
    """Manages the computational device (GPU or CPU) for the application."""
    def __init__(self):
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        if self.use_gpu:
            torch.cuda.empty_cache()
        logging.info(f"DeviceManager initialized on {self.device}")

    def switch_device(self, use_gpu: bool):
        """Switches the device between GPU and CPU based on availability and user preference."""
        self.use_gpu = use_gpu and torch.cuda.is_available()
        new_device = torch.device("cuda" if self.use_gpu else "cpu")
        if new_device != self.device:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            self.device = new_device
            logging.info(f"DeviceManager switched to {self.device}")

    def get_device(self):
        """Returns the current computational device."""
        return self.device


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa
        return x


class EdgeEnhancementBlock(nn.Module):
    """Enhances edges in the image using Sobel filters."""
    def __init__(self):
        super(EdgeEnhancementBlock, self).__init__()
        gaussian_kernel = torch.tensor([
            [1, 4, 6, 4, 1],
            [4,16,24,16, 4],
            [6,24,36,24, 6],
            [4,16,24,16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=torch.float32)/256.0
        self.register_buffer('gaussian_kernel',
                             gaussian_kernel.view(1, 1, 5, 5).repeat(3, 1, 1, 1))

        kernel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ], dtype=torch.float32)/4.0
        kernel_y = torch.tensor([
            [-1.0,-2.0,-1.0],
            [ 0.0, 0.0, 0.0],
            [ 1.0, 2.0, 1.0]
        ], dtype=torch.float32)/4.0

        self.register_buffer('kernel_x', kernel_x.view(1,1,3,3))
        self.register_buffer('kernel_y', kernel_y.view(1,1,3,3))

    def forward(self, x, edge_strength=0.0):
        if edge_strength <= 0.0:
            return x
        orig_min = x.min()
        orig_max = x.max()
        x_norm = (x - orig_min)/(orig_max - orig_min + 1e-8)
        luminance = 0.2989*x_norm[:,0:1]+0.5870*x_norm[:,1:2]+0.1140*x_norm[:,2:3]
        edge_x = F.conv2d(luminance, self.kernel_x, padding=1)
        edge_y = F.conv2d(luminance, self.kernel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x.pow(2)+edge_y.pow(2))
        edge_magnitude = edge_magnitude/(edge_magnitude.max()+1e-8)
        edge_strength = (edge_strength/100.0)*0.2
        enhancement = edge_magnitude*edge_strength
        result = torch.zeros_like(x_norm)
        for c in range(3):
            result[:,c:c+1] = x_norm[:,c:c+1]*(1.0+enhancement)
        result = (result - result.min())/(result.max()-result.min()+1e-8)
        result = result*(orig_max - orig_min)+orig_min
        result = torch.clamp(result, min=orig_min, max=orig_max)
        return result


class ColorBalanceBlock(nn.Module):
    """Balances colors across different luminance ranges."""
    def __init__(self, channels, color_preservation=0.5):
        super(ColorBalanceBlock, self).__init__()
        self.color_preservation = color_preservation
        self.shadows_param = nn.Parameter(torch.zeros(channels))
        self.midtones_param = nn.Parameter(torch.zeros(channels))
        self.highlights_param = nn.Parameter(torch.zeros(channels))
        self.color_temp = nn.Parameter(torch.tensor([0.0]))
        self.channel_corr = nn.Linear(channels, channels, bias=False)
        self.shadow_threshold = nn.Parameter(torch.tensor([0.2]))
        self.highlight_threshold = nn.Parameter(torch.tensor([0.8]))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.context_transform = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        luminance = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        shadows_mask = (luminance < self.shadow_threshold).float()
        highlights_mask = (luminance > self.highlight_threshold).float()
        midtones_mask = 1.0 - shadows_mask - highlights_mask

        shadows_adjust = self.shadows_param.view(1, C, 1, 1)
        midtones_adjust = self.midtones_param.view(1, C, 1, 1)
        highlights_adjust = self.highlights_param.view(1, C, 1, 1)
        adjust_map = shadows_mask * shadows_adjust + midtones_mask * midtones_adjust + highlights_mask * highlights_adjust
        x_balanced = x + adjust_map

        x_reshaped = x_balanced.view(B, C, -1).transpose(1, 2)

        # Ensure x_reshaped is the same dtype as channel_corr's weights
        x_reshaped = x_reshaped.to(self.channel_corr.weight.dtype)

        x_corr = self.channel_corr(x_reshaped).transpose(1, 2).view(B, C, H, W)

        temp_val = self.color_temp
        x_corr[:, 0, :, :] += temp_val * 0.05
        x_corr[:, 2, :, :] -= temp_val * 0.05

        context = self.global_pool(x_corr)
        context = self.context_transform(context)
        x_corr = x_corr * context
        x_final = self.color_preservation * x + (1.0 - self.color_preservation) * x_corr
        return x_final


class ColorCorrectionNet(nn.Module):
    """Neural network for color correction using pretrained models."""

    def __init__(self):
        super(ColorCorrectionNet, self).__init__()
        try:
            self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        except Exception as e:
            print(f"Error loading pre-trained models: {e}")
            sys.exit(1)

        # Freeze pretrained models
        for model in [self.vgg, self.resnet, self.densenet]:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

        # Simplified adaptation layers
        self.vgg_adapt = nn.Sequential(
            nn.Conv2d(64, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.resnet_adapt = nn.Sequential(
            nn.Conv2d(64, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.densenet_adapt = nn.Sequential(
            nn.Conv2d(64, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Simplified fusion with anti-artifact measures
        self.fusion = nn.Sequential(
            nn.Conv2d(768, 384, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        # Modified color transform to prevent banding
        self.color_transform = nn.Sequential(
            nn.Conv2d(192, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 3, 3, padding=1),
            nn.Sigmoid()  # Changed to Sigmoid for smoother color transitions
        )

        # Minimal enhancement blocks
        self.edge_enhance = EdgeEnhancementBlock()
        self.cbam = CBAM(192, reduction=8)  # Reduced complexity

    def extract_features(self, x):
        # VGG features
        vgg_feat = self.vgg.features[:5](x)
        vgg_feat = self.vgg_adapt(vgg_feat)

        # ResNet features
        x_res = self.resnet.conv1(x)
        x_res = self.resnet.bn1(x_res)
        x_res = self.resnet.relu(x_res)
        resnet_feat = self.resnet_adapt(x_res)

        # DenseNet features
        densenet_feat = self.densenet.features[:4](x)
        densenet_feat = self.densenet_adapt(densenet_feat)

        return vgg_feat, resnet_feat, densenet_feat

    def forward(self, x):
        with torch.inference_mode():
            # Ensure proper dtype and range
            if x.device.type == 'cpu':
                x = x.float()

            # Preserve original range
            x_min = x.amin(dim=[2, 3], keepdim=True)
            x_max = x.amax(dim=[2, 3], keepdim=True)
            x_range = x_max - x_min
            eps = 1e-8

            # Normalize with epsilon to prevent division by zero
            x_normalized = (x - x_min) / (x_range + eps)

            # Extract features
            vgg_feat, resnet_feat, densenet_feat = self.extract_features(x_normalized)

            # Ensure consistent spatial dimensions
            target_size = vgg_feat.shape[-2:]
            resnet_feat = F.interpolate(resnet_feat, size=target_size, mode='bilinear', align_corners=False)
            densenet_feat = F.interpolate(densenet_feat, size=target_size, mode='bilinear', align_corners=False)

            # Fuse features
            fused = self.fusion(torch.cat([vgg_feat, resnet_feat, densenet_feat], dim=1))
            del vgg_feat, resnet_feat, densenet_feat

            # Apply attention and enhancement
            fused = self.cbam(fused)

            # Generate color adjustment
            delta = self.color_transform(fused)
            del fused

            # Upsample delta to match input resolution
            delta = F.interpolate(delta, size=x.shape[-2:], mode='bilinear', align_corners=False)

            # Apply subtle color enhancement
            delta = (delta - 0.5) * 0.2  # Scale adjustments to ±0.1 range
            enhanced = x_normalized * (1.0 + delta)

            # Restore original range with smooth clamping
            enhanced = enhanced * x_range + x_min
            enhanced = torch.clamp(enhanced, min=x_min, max=x_max)

            # Cleanup
            if x.device.type == 'cuda':
                torch.cuda.empty_cache()

            return enhanced


class OptimizedJXRLoader:
    """Handles loading and processing of JXR images."""
    def __init__(self, device):
        self.device = device
        self.hdr_peak_luminance = 10000.0
        self.selected_tone_map = "adaptive"
        self.selected_pre_gamma = 1.0
        self.selected_auto_exposure = 1.0
        self.ACES_INPUT_MAT = torch.tensor([
            [0.59719,0.35458,0.04823],
            [0.07600,0.90834,0.01566],
            [0.02840,0.13383,0.83777]
        ])
        self.ACES_OUTPUT_MAT = torch.tensor([
            [1.60475,-0.53108,-0.07367],
            [-0.10208,1.10813,-0.00605],
            [-0.00327,-0.07276,1.07602]
        ])

    def _apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        pos_mask = image > 0
        result = np.zeros_like(image)
        if np.any(pos_mask):
            result[pos_mask] = np.power(image[pos_mask], gamma)
        return result

    def load_jxr(self, file_path: str, is_preview: bool=False):
        """Loads and preprocesses a JXR image."""
        try:
            with open(file_path, 'rb') as f:
                jxr_data = f.read()

            image = imagecodecs.jpegxr_decode(jxr_data)
            if image is None:
                raise ValueError("Failed to decode JXR image")
            image = image.astype(np.float32)

            if is_preview:
                pre_gamma = 1.0
                auto_exposure = 1.0
                tone_map = 'uncharted2'
            else:
                pre_gamma = 1.0 + (self.selected_pre_gamma - 1.0)/3.0
                auto_exposure = 1.0 + (self.selected_auto_exposure - 1.0)/3.0
                tone_map = self.selected_tone_map

            if pre_gamma != 1.0:
                gamma = 1.0 / pre_gamma
                image = self._apply_gamma_correction(image, gamma)

            if auto_exposure != 1.0:
                mean_luminance = np.mean(image)
                exposure_factor = auto_exposure
                if mean_luminance > 0:
                    exposure_factor *= (0.18 / mean_luminance)
                image *= exposure_factor

            display_peak = 1000.0
            image = image * (display_peak / self.hdr_peak_luminance)

            if image.ndim == 2:
                image = np.stack([image]*3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:,:,:3]

            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            tensor = torch.from_numpy(image).float().permute(2,0,1).contiguous().unsqueeze(0)
            # Move tensor to device and maybe half precision for memory saving
            if self.device.type == 'cuda':
                tensor = tensor.to(self.device, non_blocking=True)
                tensor = tensor.half()

            # Apply tone map
            if tone_map == 'hable' or is_preview:
                tensor = self._tone_map_hable(tensor)
            elif tone_map == 'reinhard':
                tensor = self._tone_map_reinhard(tensor)
            elif tone_map == 'filmic':
                tensor = self._tone_map_filmic(tensor)
            elif tone_map == 'aces':
                tensor = self._tone_map_aces(tensor)
            elif tone_map == 'uncharted2':
                tensor = self._tone_map_uncharted2(tensor)
            elif tone_map == 'adaptive':
                tensor = self._tone_map_adaptive(tensor)

            tensor = self.linear_to_srgb(tensor)
            return tensor, None, {}
        except Exception as e:
            logging.error(f"Failed to load HDR image: {e}")
            return None, str(e), {}

    def _tone_map_hable(self, x):
        """Applies Hable's tone mapping."""
        A,B,C,D,E,F_=0.22,0.30,0.10,0.20,0.01,0.30
        W=11.2
        def evaluate(v):
            num=(v*(A*v+C*B)+D*E)
            den=(v*(A*v+B)+D*F_)
            return num/(den+1e-6)-E/F_
        nom = evaluate(x)
        denom = evaluate(torch.tensor(W, device=x.device, dtype=x.dtype))
        return nom/(denom+1e-6)

    def _tone_map_reinhard(self, x, L_white=4.0):
        """Applies Reinhard's tone mapping."""
        L = 0.2126*x[:,0] + 0.7152*x[:,1] + 0.0722*x[:,2]
        L_avg = torch.mean(L)
        L = L / (L_avg + 1e-6)
        L_scaled = (L * (1.0 + L / (L_white * L_white))) / (1.0 + L)
        ratio = torch.where(L > 1e-8, L_scaled / (L + 1e-6), torch.ones_like(L))
        return torch.stack([x[:,i] * ratio for i in range(3)], dim=1)

    def _tone_map_aces(self, x):
        """Applies ACES tone mapping."""
        if self.ACES_INPUT_MAT.device != x.device:
            self.ACES_INPUT_MAT = self.ACES_INPUT_MAT.to(x.device, dtype=x.dtype)
            self.ACES_OUTPUT_MAT = self.ACES_OUTPUT_MAT.to(x.device, dtype=x.dtype)

        batch, channels, height, width = x.shape
        x_reshaped = x.view(batch, channels, -1)
        x_transformed = torch.einsum('ij,bjk->bik', self.ACES_INPUT_MAT, x_reshaped)
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        x_tonemapped = (x_transformed * (a * x_transformed + b)) / (x_transformed * (c * x_transformed + d) + e)
        x_output = torch.einsum('ij,bjk->bik', self.ACES_OUTPUT_MAT, x_tonemapped)
        return x_output.view(batch, channels, height, width).contiguous()

    def _tone_map_filmic(self, x):
        """Applies Filmic tone mapping."""
        x = torch.max(torch.zeros_like(x), x)
        A,B,C,D,E,F_=0.22,0.30,0.10,0.20,0.01,0.30
        def filmic_curve(v):
            return ((v*(A*v+C*B)+D*E)/(v*(A*v+B)+D*F_))-E/F_
        return filmic_curve(x)

    def _tone_map_uncharted2(self, x):
        """Applies Uncharted2 tone mapping."""
        A,B,C,D,E,F_=0.15,0.50,0.10,0.20,0.02,0.30
        def uncharted2_tonemap(v):
            return ((v*(A*v+C*B)+D*E)/(v*(A*v+B)+D*F_))-E/F_
        exposure_bias = 2.0
        curr = uncharted2_tonemap(x * exposure_bias)
        W = 11.2
        white_scale = uncharted2_tonemap(torch.tensor(W, device=x.device, dtype=x.dtype))
        return curr / white_scale

    def _tone_map_adaptive(self, x):
        """Applies adaptive tone mapping based on dynamic range."""
        L = 0.2126*x[:,0] + 0.7152*x[:,1] + 0.0722*x[:,2]
        avg_luminance = torch.mean(L)
        max_luminance = torch.max(L)
        dynamic_range = max_luminance / (avg_luminance + 1e-6)
        if dynamic_range > 100:
            return self._tone_map_aces(x)
        elif dynamic_range > 10:
            return self._tone_map_hable(x)
        elif avg_luminance < 0.1:
            return self._tone_map_uncharted2(x)
        else:
            return self._tone_map_reinhard(x)

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
        """Generates a preview tensor resized to target dimensions."""
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
        """Converts a tensor to a PIL Image."""
        try:
            if tensor.is_cuda:
                tensor = tensor.cpu()
            tensor = tensor.float()  # Convert back to float32 for PIL
            tensor = torch.clamp(tensor, 0.0, 1.0)
            tensor = (tensor * 255).byte()
            img_array = tensor.squeeze(0).permute(1, 2, 0).numpy()
            return Image.fromarray(img_array)
        except Exception as e:
            logging.error(f"Tensor to PIL conversion failed: {e}")
            return None


class HDRColorProcessor:
    """Processes HDR images with color and edge enhancements."""
    def __init__(self, device, jxr_loader):
        self.device = device
        self.jxr_loader = jxr_loader
        self.color_net = ColorCorrectionNet().eval().to(device)
        # Convert model to half if GPU is available for memory saving
        if device.type == 'cuda':
            self.color_net.half()
        self.edge_enhancement = EdgeEnhancementBlock().to(device)
        if device.type == 'cuda':
            self.edge_enhancement.half()

    def clear_gpu_memory(self):
        """Clears GPU memory cache."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def process_image(self, original_tensor, output_path, color_strength=0.0, edge_strength=0.0, use_enhancement=True):
        """
        Process an HDR image tensor with color and edge enhancements.

        Args:
            original_tensor (torch.Tensor): Input HDR image tensor
            output_path (str): Path to save the processed image
            color_strength (float): Strength of color enhancement (0-100)
            edge_strength (float): Strength of edge enhancement (0-100)
            use_enhancement (bool): Whether to apply AI enhancement

        Returns:
            PIL.Image or None: Processed image if successful, None if failed
        """
        try:
            # Move tensor to correct device and set precision
            tensor = original_tensor.to(self.device)
            if self.device.type == 'cuda':
                tensor = tensor.half()  # Use half precision only on GPU
            else:
                tensor = tensor.float()  # Use full precision on CPU

            # Add padding for processing
            pad_size = 32
            padded_tensor = F.pad(tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

            with torch.inference_mode():
                # Apply color enhancement if enabled
                if use_enhancement and color_strength > 0:
                    enhanced = self.color_net(padded_tensor)
                    normalized_strength = (color_strength / 100.0) * 1.5
                    enhanced_strength = pow(normalized_strength, 0.7)
                    enhanced = padded_tensor * (1 - enhanced_strength) + enhanced * enhanced_strength
                else:
                    enhanced = padded_tensor

                # Apply edge enhancement if enabled
                if edge_strength > 0:
                    enhanced = self.edge_enhancement(enhanced, edge_strength=edge_strength)

                # Remove padding and convert back to float32 for CPU processing
                enhanced = enhanced[:, :, pad_size:-pad_size, pad_size:-pad_size].float().cpu()

                # Convert to numpy array for saving
                array = enhanced.squeeze(0).permute(1, 2, 0).numpy()

                # Clean up GPU memory
                del enhanced, padded_tensor, tensor
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Normalize the array for saving as JPEG
                array_max = array.max(axis=(0, 1), keepdims=True)
                array_min = array.min(axis=(0, 1), keepdims=True)
                array = (array - array_min) / (array_max - array_min + 1e-8) * 255.0
                array = np.clip(array, 0, 255).astype(np.uint8)

                # Convert to PIL Image and save
                result_image = Image.fromarray(array)
                result_image.save(output_path, 'JPEG', quality=95, optimize=True)

                # Final cleanup
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

                return result_image

        except Exception as e:
            print(f"Processing failed: {str(e)}")
            traceback.print_exc()
            return None

    def switch_device(self, use_gpu):
        """Switches the processing device and updates model data types accordingly."""
        new_device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if new_device != self.device:
            self.clear_gpu_memory()
            self.device = new_device

            # Define desired data type based on the device
            desired_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32

            # Move color_net and edge_enhancement to the new device and dtype
            self.color_net = self.color_net.to(device=self.device, dtype=desired_dtype)
            self.edge_enhancement = self.edge_enhancement.to(device=self.device, dtype=desired_dtype)

            # Move pretrained models inside color_net
            for model in [self.color_net.vgg, self.color_net.resnet, self.color_net.densenet]:
                model.to(device=self.device, dtype=desired_dtype)

            # Move ACES matrices to the new device and dtype
            self.jxr_loader.ACES_INPUT_MAT = self.jxr_loader.ACES_INPUT_MAT.to(device=self.device, dtype=desired_dtype)
            self.jxr_loader.ACES_OUTPUT_MAT = self.jxr_loader.ACES_OUTPUT_MAT.to(device=self.device, dtype=desired_dtype)

            logging.info(f"Switched to {self.device}")
            return True
        else:
            logging.info(f"Already on {self.device}")
            return False


def validate_files(input_file: str, output_file: str) -> None:
    """Validates the existence of input and output file paths."""
    if not input_file or not os.path.exists(input_file):
        raise FileNotFoundError("Input file does not exist")
    if not output_file:
        raise ValueError("Output file path not specified")


def validate_parameters(tone_map: str, pre_gamma: str, auto_exposure: str) -> None:
    """Validates the tone map, pre-gamma, and auto-exposure parameters."""
    if tone_map not in SUPPORTED_TONE_MAPS:
        raise ValueError(f"Unsupported tone map: {tone_map}")
    try:
        float(pre_gamma)
        float(auto_exposure)
    except ValueError:
        raise ValueError("Pre-gamma and auto-exposure must be numeric")


class App(TKMT.ThemedTKinterFrame):
    """Main application class for the NVIDIA HDR Converter GUI."""

    def __init__(self, theme="park", mode="dark"):
        super().__init__("NVIDIA HDR Converter", theme, mode)
        self.device_manager = DeviceManager()
        self.jxr_loader = OptimizedJXRLoader(self.device_manager.get_device())
        self.color_processor = HDRColorProcessor(self.device_manager.get_device(), self.jxr_loader)

        self.current_before_image_path = None
        self.current_after_image_path = None
        self.original_input_tensor = None

        self.before_image_ref = None
        self.after_image_ref = None
        self.before_hist_ref = None
        self.after_hist_ref = None

        # Main Frame Setup
        main_frame = ttk.Frame(self.master, padding=(10, 10))
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Left Frame: Controls
        left_frame = ttk.Frame(main_frame, padding=(10, 10))
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        # Right Frame: Previews and Histograms
        right_frame = ttk.Frame(main_frame, padding=(10, 10))
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nw")

        # Setup various UI components
        self._setup_mode_selection(left_frame)
        self._setup_file_selection(left_frame)
        self._setup_device_selection(left_frame)
        self._setup_parameters(left_frame)
        self._setup_color_controls(left_frame)
        self._setup_conversion_button(left_frame)
        self._setup_progress_bar(left_frame)
        self._setup_status_label(left_frame)

        # Mode Selection Variables
        self.mode_size_var = tk.StringVar(value='small')
        self.current_mode = 'small'
        self.preview_width = MODES[self.current_mode]['preview_width']
        self.preview_height = MODES[self.current_mode]['preview_height']

        # Setup Previews and Histograms
        self._setup_previews(right_frame)
        self._setup_histograms(right_frame)
        self.update_mode_size()
        self.ui_lock = threading.Lock()

        # Mode Switch Frame
        mode_switch_frame = ttk.Frame(main_frame, padding=(10, 10))
        mode_switch_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(mode_switch_frame, text="UI Mode:").grid(row=0, column=0, sticky="e", padx=(0, 5))
        big_radio = ttk.Radiobutton(mode_switch_frame, text="Big", variable=self.mode_size_var, value='big',
                                    command=self.update_mode_size)
        big_radio.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        small_radio = ttk.Radiobutton(mode_switch_frame, text="Small", variable=self.mode_size_var, value='small',
                                      command=self.update_mode_size)
        small_radio.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Center and Initialize Window
        self.center_window(MODES[self.current_mode]['window_width'], MODES[self.current_mode]['window_height'])

        # Set initial state of AI Enhancement controls based on the device
        if self.device_manager.use_gpu:
            self.enhance_checkbox.config(state='normal')
            self.edge_scale.config(state='normal')
        else:
            self.enhance_checkbox.config(state='disabled')
            self.edge_scale.config(state='disabled')
            self.use_enhancement.set(False)
            self._update_enhancement_controls()

        logging.info(f"Application initialized on {self.device_manager.device}")

    def _setup_mode_selection(self, parent_frame):
        """Sets up the mode selection radio buttons (Single File or Folder)."""
        mode_frame = ttk.LabelFrame(parent_frame, text="Mode Selection", padding=(10, 10))
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.mode_var = tk.StringVar(value="single")
        single_radio = ttk.Radiobutton(mode_frame, text="Single File", variable=self.mode_var, value="single",
                                       command=self.update_mode)
        single_radio.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        folder_radio = ttk.Radiobutton(mode_frame, text="Folder", variable=self.mode_var, value="folder",
                                       command=self.update_mode)
        folder_radio.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    def update_mode(self):
        """Updates the UI based on the selected mode (Single File or Folder)."""
        mode = self.mode_var.get()
        if mode == "single":
            self.file_frame.config(text="File Selection")
            self.input_label.config(text="Input JXR:")
            self.output_label.grid()
            self.output_entry.grid()
            self.output_browse_button.grid()
            self.status_label.config(text="")
            self.convert_btn.config(state='normal')
        else:
            self.file_frame.config(text="Folder Selection")
            self.input_label.config(text="Input Folder:")
            self.output_label.grid_remove()
            self.output_entry.grid_remove()
            self.output_browse_button.grid_remove()
            self.output_entry.delete(0, tk.END)
            folder_path = self.input_entry.get().strip()
            if folder_path and os.path.isdir(folder_path):
                jxr_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jxr')]
                file_count = len(jxr_files)
                if file_count == 0:
                    self.status_label.config(text="No JXR files found in selected folder.", foreground="red")
                    self.convert_btn.config(state='disabled')
                else:
                    self.status_label.config(
                        text=f"Found {file_count} JXR files ready for conversion.",
                        foreground="#00FF00"
                    )
                    self.convert_btn.config(state='normal')
            else:
                self.status_label.config(text="Please select a folder containing JXR files.", foreground="#CCCCCC")
                self.convert_btn.config(state='disabled')

            # Clear previews and histograms
            self.before_label.config(image="", text="No Preview")
            self.before_image_ref = None
            self.after_label.config(image="", text="No Preview")
            self.after_image_ref = None
            self.before_hist_label.config(image="", text="No Histogram")
            self.before_hist_ref = None
            self.after_hist_label.config(image="", text="No Histogram")
            self.after_hist_ref = None

    def _setup_file_selection(self, parent_frame):
        """Sets up file or folder selection widgets."""
        self.file_frame = ttk.LabelFrame(parent_frame, text="File Selection", padding=(10, 10))
        self.file_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.input_label = ttk.Label(self.file_frame, text="Input JXR:")
        self.input_label.grid(row=0, column=0, sticky="e", padx=(0, 5), pady=5)
        self.input_entry = ttk.Entry(self.file_frame, width=40)
        self.input_entry.grid(row=0, column=1, padx=(0, 5), pady=5)
        self.browse_button = ttk.Button(self.file_frame, text="Browse...", command=self.browse_input)
        self.browse_button.grid(row=0, column=2, padx=(0, 5), pady=5)

        self.output_label = ttk.Label(self.file_frame, text="Output JPG:")
        self.output_label.grid(row=1, column=0, sticky="e", padx=(0, 5), pady=5)
        self.output_entry = ttk.Entry(self.file_frame, width=40)
        self.output_entry.grid(row=1, column=1, padx=(0, 5), pady=5)
        self.output_browse_button = ttk.Button(self.file_frame, text="Browse...", command=self.browse_output)
        self.output_browse_button.grid(row=1, column=2, padx=(0, 5), pady=5)

    def _setup_device_selection(self, parent_frame):
        """Sets up device selection radio buttons (GPU or CPU)."""
        device_frame = ttk.LabelFrame(parent_frame, text="Processing Device", padding=(10, 10))
        device_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.use_gpu_var = tk.BooleanVar(value=self.device_manager.use_gpu)
        gpu_state = 'normal' if torch.cuda.is_available() else 'disabled'
        gpu_label = "GPU (CUDA)" if torch.cuda.is_available() else "GPU (Not Available)"
        gpu_radio = ttk.Radiobutton(device_frame, text=gpu_label,
                                    variable=self.use_gpu_var, value=True,
                                    command=self.update_device,
                                    state=gpu_state)
        gpu_radio.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        cpu_radio = ttk.Radiobutton(device_frame, text="CPU", variable=self.use_gpu_var, value=False,
                                    command=self.update_device)
        cpu_radio.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    def _setup_parameters(self, parent_frame):
        """Sets up parameter input fields for tone mapping, gamma, and exposure."""
        self.params_frame = ttk.LabelFrame(parent_frame, text="Parameters", padding=(10, 10))
        self.params_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(self.params_frame, text="Tone Map:").grid(row=0, column=0, sticky="e", padx=(0, 5), pady=5)
        self.tonemap_var = tk.StringVar(value=DEFAULT_TONE_MAP)
        tone_map_values = sorted(list(SUPPORTED_TONE_MAPS))
        tone_map_dropdown = ttk.Combobox(self.params_frame, textvariable=self.tonemap_var, values=tone_map_values,
                                         state='readonly')
        tone_map_dropdown.grid(row=0, column=1, padx=(0, 5), pady=5)

        ttk.Label(self.params_frame, text="Gamma:").grid(row=1, column=0, sticky="e", padx=(0, 5), pady=5)
        self.pregamma_var = tk.StringVar(value=DEFAULT_PREGAMMA)
        pregamma_entry = ttk.Entry(self.params_frame, textvariable=self.pregamma_var, width=15)
        pregamma_entry.grid(row=1, column=1, padx=(0, 5), pady=5)

        ttk.Label(self.params_frame, text="Exposure:").grid(row=2, column=0, sticky="e", padx=(0, 5), pady=5)
        self.autoexposure_var = tk.StringVar(value=DEFAULT_AUTOEXPOSURE)
        autoexposure_entry = ttk.Entry(self.params_frame, textvariable=self.autoexposure_var, width=15)
        autoexposure_entry.grid(row=2, column=1, padx=(0, 5), pady=5)

    def _setup_conversion_button(self, parent_frame):
        """Sets up the conversion button."""
        convert_frame = ttk.Frame(parent_frame, padding=(0, 0))
        convert_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        convert_frame.grid_columnconfigure(0, weight=1)
        self.convert_btn = ttk.Button(convert_frame, text="Convert", command=self.convert_image, style="Accent.TButton")
        self.convert_btn.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        style = ttk.Style()
        try:
            if 'Accent.TButton' not in style.element_names():
                style.configure(
                    'Accent.TButton',
                    font=('Segoe UI', 10),
                    padding=5,
                    foreground='white',
                    background='#007ACC'
                )
                style.map('Accent.TButton',
                          background=[('active', '#005A9E')],
                          foreground=[('active', 'white')])
        except:
            pass

    def _setup_progress_bar(self, parent_frame):
        """Sets up the progress bar."""
        progress_frame = ttk.Frame(parent_frame, padding=(0, 0))
        progress_frame.grid(row=5, column=0, sticky="ew", pady=(5, 10))
        progress_frame.grid_columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew")

    def _setup_status_label(self, parent_frame):
        """Sets up the status label to display messages."""
        self.status_label = ttk.Label(parent_frame, text="", foreground="#CCCCCC")
        self.status_label.grid(row=6, column=0, sticky="w", pady=(0, 10))

    def _setup_previews(self, parent_frame):
        """Sets up the preview image canvases."""
        previews_frame = ttk.Frame(parent_frame, padding=(10, 10))
        previews_frame.grid(row=0, column=0, sticky="nsew")
        previews_frame.grid_columnconfigure(0, weight=1)
        previews_frame.grid_columnconfigure(1, weight=1)
        previews_frame.grid_rowconfigure(0, weight=1)

        before_frame = ttk.LabelFrame(previews_frame, text="Before Conversion", padding=(10, 10))
        before_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.before_canvas = tk.Canvas(before_frame, width=self.preview_width, height=self.preview_height)
        self.before_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.before_label = ttk.Label(self.before_canvas)
        self.before_canvas.create_window(self.preview_width // 2, self.preview_height // 2, window=self.before_label)

        after_frame = ttk.LabelFrame(previews_frame, text="After Conversion", padding=(10, 10))
        after_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.after_canvas = tk.Canvas(after_frame, width=self.preview_width, height=self.preview_height)
        self.after_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.after_label = ttk.Label(self.after_canvas)
        self.after_canvas.create_window(self.preview_width // 2, self.preview_height // 2, window=self.after_label)

    def _setup_histograms(self, parent_frame):
        """Sets up the histogram canvases for before and after images."""
        histograms_frame = ttk.Frame(parent_frame, padding=(10, 10))
        histograms_frame.grid(row=1, column=0, sticky="nsew")

        before_hist_frame = ttk.LabelFrame(histograms_frame, text="Before Conversion Histogram", padding=(10, 10))
        before_hist_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.before_hist_canvas = tk.Canvas(before_hist_frame, width=self.preview_width, height=150)
        self.before_hist_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.before_hist_label = ttk.Label(self.before_hist_canvas)
        self.before_hist_canvas.create_window(self.preview_width // 2, 75, window=self.before_hist_label)

        after_hist_frame = ttk.LabelFrame(histograms_frame, text="After Conversion Histogram", padding=(10, 10))
        after_hist_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.after_hist_canvas = tk.Canvas(after_hist_frame, width=self.preview_width, height=150)
        self.after_hist_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.after_hist_label = ttk.Label(self.after_hist_canvas)
        self.after_hist_canvas.create_window(self.preview_width // 2, 75, window=self.after_hist_label)

    def update_mode_size(self):
        """Updates the window and preview sizes based on selected mode."""
        mode = self.mode_size_var.get()
        logging.info(f"Switching to mode: {mode}")
        self.current_mode = mode
        self.preview_width = MODES[mode]['preview_width']
        self.preview_height = MODES[mode]['preview_height']
        self.root.minsize(MODES[mode]['window_width'], MODES[mode]['window_height'])
        self.root.geometry(f"{MODES[mode]['window_width']}x{MODES[mode]['window_height']}")

        self.adjust_preview_canvases()
        self.adjust_histogram_canvases()

        if self.original_input_tensor is not None:
            self.before_hist_label.config(image="", text="")
            self.before_hist_ref = None
            preview_tensor = self.jxr_loader.process_preview(
                self.original_input_tensor,
                self.preview_width,
                self.preview_height
            )
            self._update_preview_ui(preview_tensor)
            self.show_color_spectrum_from_tensor(self.original_input_tensor, is_before=True)

        if getattr(self, 'current_after_image_path', None):
            self.show_preview_from_file(self.current_after_image_path, is_before=False)
            self.show_color_spectrum(self.current_after_image_path, is_before=False)

        self.center_window(MODES[mode]['window_width'], MODES[mode]['window_height'])
        self.status_label.config(text=f"Switched to {mode} mode", foreground="#CCCCCC")

    def adjust_preview_canvases(self):
        """Adjusts the size of preview canvases based on the current mode."""
        self.before_canvas.config(width=self.preview_width, height=self.preview_height)
        self.before_canvas.delete("all")
        self.before_canvas.create_window(self.preview_width // 2, self.preview_height // 2, window=self.before_label)

        self.after_canvas.config(width=self.preview_width, height=self.preview_height)
        self.after_canvas.delete("all")
        self.after_canvas.create_window(self.preview_width // 2, self.preview_height // 2, window=self.after_label)

    def adjust_histogram_canvases(self):
        """Adjusts the size of histogram canvases based on the current mode."""
        self.before_hist_canvas.config(width=self.preview_width, height=150)
        self.before_hist_canvas.delete("all")
        self.before_hist_canvas.create_window(self.preview_width // 2, 75, window=self.before_hist_label)

        self.after_hist_canvas.config(width=self.preview_width, height=150)
        self.after_hist_canvas.delete("all")
        self.after_hist_canvas.create_window(self.preview_width // 2, 75, window=self.after_hist_label)

    def center_window(self, width, height):
        """Centers the application window on the screen."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def update_device(self):
        """Handles the device switching logic based on user selection."""
        use_gpu = self.use_gpu_var.get()
        try:
            self.device_manager.switch_device(use_gpu)
            success = self.color_processor.switch_device(use_gpu)
            if success:
                device_name = "GPU" if use_gpu and torch.cuda.is_available() else "CPU"
                self.status_label.config(text=f"Switched to {device_name}", foreground="#CCCCCC")

                # Enable or disable AI Enhancement controls based on the device
                if self.device_manager.use_gpu:
                    self.enhance_checkbox.config(state='normal')
                    self.edge_scale.config(state='normal')
                else:
                    self.enhance_checkbox.config(state='disabled')
                    self.edge_scale.config(state='disabled')
                    self.use_enhancement.set(False)
                    self._update_enhancement_controls()
            else:
                raise RuntimeError("Failed to switch devices")
        except Exception as e:
            error_msg = f"Device switching failed: {str(e)}"
            self.status_label.config(text=error_msg, foreground="red")
            logging.error(error_msg)
            self.use_gpu_var.set(not use_gpu)

    def browse_input(self):
        """Handles the input file or folder browsing."""
        mode = self.mode_var.get()
        if mode == "single":
            filename = filedialog.askopenfilename(
                title="Select Input JXR File",
                filetypes=[("JXR files", "*.jxr"), ("All files", "*.*")]
            )
            if filename:
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, filename)
                self.status_label.config(text="Loading preview...", foreground="#CCCCCC")
                self.master.update_idletasks()
                self.create_preview_from_jxr(filename)
        else:
            foldername = filedialog.askdirectory(title="Select Folder Containing JXR Files")
            if foldername:
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, foldername)
                jxr_files = [f for f in os.listdir(foldername) if f.lower().endswith('.jxr')]
                file_count = len(jxr_files)
                if file_count == 0:
                    self.status_label.config(text="No JXR files found in selected folder.", foreground="red")
                    self.convert_btn.config(state='disabled')
                else:
                    self.status_label.config(
                        text=f"Found {file_count} JXR files ready for conversion.",
                        foreground="#00FF00"
                    )
                    self.convert_btn.config(state='normal')

                # Clear previews and histograms
                self.before_label.config(image="", text="No Preview")
                self.before_image_ref = None
                self.after_label.config(image="", text="No Preview")
                self.after_image_ref = None
                self.before_hist_label.config(image="", text="No Histogram")
                self.before_hist_ref = None
                self.after_hist_label.config(image="", text="No Histogram")
                self.after_hist_ref = None

    def browse_output(self):
        """Handles the output file browsing."""
        filename = filedialog.asksaveasfilename(
            title="Select Output JPG File",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if filename:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, filename)

    def create_preview_from_jxr(self, jxr_file: str):
        """Generates a preview from a selected JXR file."""
        if not jxr_file or not jxr_file.lower().endswith('.jxr'):
            self._clear_preview()
            self.status_label.config(text="Invalid JXR file", foreground="red")
            return
        if not os.path.exists(jxr_file):
            self._clear_preview()
            self.status_label.config(text="File not found", foreground="red")
            return

        self.status_label.config(text="Loading preview...", foreground="#CCCCCC")
        self.master.update_idletasks()

        def process_jxr():
            try:
                tensor, error, metadata = self.jxr_loader.load_jxr(jxr_file, is_preview=True)
                if tensor is None:
                    raise RuntimeError("Failed to load JXR file.")
                self.original_input_tensor = tensor.clone()  # Store original always
                preview_tensor = self.jxr_loader.process_preview(
                    tensor,
                    self.preview_width,
                    self.preview_height
                )
                if preview_tensor is None:
                    raise RuntimeError("Failed to generate preview")

                self.master.after(0, lambda: self._update_preview_ui(preview_tensor))
            except Exception as e:
                error_msg = str(e)
                logging.error(f"Preview generation error: {error_msg}")
                self.master.after(0, lambda: self._handle_preview_error(error_msg))

        thread = threading.Thread(target=process_jxr, daemon=True)
        thread.start()

    def _handle_preview_error(self, error_msg: str):
        """Handles errors during preview generation."""
        self.status_label.config(text=f"Preview failed: {error_msg}", foreground="red")
        self.before_label.config(image="", text="Preview Failed")
        self.before_image_ref = None
        self.before_hist_label.config(image="", text="No Histogram")
        self.before_hist_ref = None

    def _clear_preview(self):
        """Clears the preview and histogram displays."""
        self.before_label.config(image="", text="No Preview")
        self.before_image_ref = None
        self.before_hist_label.config(image="", text="No Histogram")
        self.before_hist_ref = None

    def show_preview_from_file(self, filepath: str, is_before: bool):
        """Displays a preview image from a file."""
        label = self.before_label if is_before else self.after_label
        if is_before:
            if self.before_image_ref:
                self.before_label.config(image="")
                self.before_image_ref = None
            self.current_before_image_path = filepath
        else:
            if self.after_image_ref:
                self.after_label.config(image="")
                self.after_image_ref = None
            self.current_after_image_path = filepath

        try:
            img = Image.open(filepath).convert('RGB')
            img = self._resize_image(img, self.preview_width, self.preview_height)
            img_tk = ImageTk.PhotoImage(img)
            label.config(image=img_tk, text="")
            if is_before:
                self.before_image_ref = img_tk
            else:
                self.after_image_ref = img_tk
            img.close()
        except Exception as e:
            label.config(image="", text=f"Preview Error: {str(e)}")
            if is_before:
                self.before_image_ref = None
            else:
                self.after_image_ref = None
            logging.error(f"Error loading preview: {str(e)}")

    def show_color_spectrum(self, filepath: str, is_before: bool):
        """Generates and displays a color spectrum histogram from an image file."""
        label = self.before_hist_label if is_before else self.after_hist_label
        if is_before and self.before_hist_ref:
            self.before_hist_label.config(image="", text="")
            self.before_hist_ref = None
        elif not is_before and self.after_hist_ref:
            self.after_hist_label.config(image="", text="")
            self.after_hist_ref = None

        try:
            img = Image.open(filepath).convert('RGB')
            vis_img = self._create_histogram_image(np.array(img))
            vis_tk = ImageTk.PhotoImage(vis_img)
            def update_visualization():
                label.config(image=vis_tk, text="")
                if is_before:
                    self.before_hist_ref = vis_tk
                else:
                    self.after_hist_ref = vis_tk
            self.master.after(0, update_visualization)
        except Exception as e:
            label.config(image="", text=f"Visualization Error: {str(e)}")
            if is_before:
                self.before_hist_ref = None
            else:
                self.after_hist_ref = None
            logging.error(f"Error generating color spectrum: {str(e)}")

    def show_color_spectrum_from_tensor(self, tensor: torch.Tensor, is_before: bool):
        """Generates and displays a color spectrum histogram from a tensor."""
        label = self.before_hist_label if is_before else self.after_hist_label
        if is_before and self.before_hist_ref:
            self.before_hist_label.config(image="", text="")
            self.before_hist_ref = None
        elif not is_before and self.after_hist_ref:
            self.after_hist_label.config(image="", text="")
            self.after_hist_ref = None

        try:
            img_tensor = tensor.squeeze(0)
            if img_tensor.shape[0] != 3:
                img_tensor = img_tensor[:3, :, :]
            img_data = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            vis_img = self._create_histogram_image(img_data)
            vis_tk = ImageTk.PhotoImage(vis_img)

            def update_visualization():
                label.config(image=vis_tk, text="")
                if is_before:
                    self.before_hist_ref = vis_tk
                else:
                    self.after_hist_ref = vis_tk

            self.master.after(0, update_visualization)
        except Exception as e:
            label.config(image="", text=f"Visualization Error: {str(e)}")
            if is_before:
                self.before_hist_ref = None
            else:
                self.after_hist_ref = None
            logging.error(f"Error generating color spectrum from tensor: {str(e)}")

    def _create_histogram_image(self, img_array: np.ndarray) -> Image.Image:
        """Creates a histogram visualization from an image array."""
        vis_img = Image.new('RGB', (self.preview_width, 150), '#1e1e1e')
        draw = ImageDraw.Draw(vis_img, 'RGBA')
        draw.rectangle([0,0,self.preview_width,150], fill='#2b2b2b')
        grid_spacing = 150//4
        for i in range(5):
            y = i*grid_spacing
            draw.line([(0,y),(self.preview_width,y)], fill='#40404040', width=1)
        channels = [
            (img_array[:,:,0], '#ff000066'),
            (img_array[:,:,1], '#00ff0066'),
            (img_array[:,:,2], '#0000ff66')
        ]
        for channel_data, color in channels:
            hist, _ = np.histogram(channel_data, bins=self.preview_width, range=(0,255))
            if hist.max() > 0:
                hist = hist / hist.max() * (150 - 10)
            points = [(0,150)]
            for x in range(self.preview_width):
                y = 150 - hist[x]
                points.append((x,y))
            points.append((self.preview_width,150))
            draw.polygon(points, fill=color)
        return vis_img

    def _resize_image(self, image: Image.Image, max_width: int, max_height: int) -> Image.Image:
        """Resizes an image while maintaining aspect ratio."""
        original_width, original_height = image.size
        ratio = min(max_width / original_width, max_height / original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _update_preview_ui(self, tensor: torch.Tensor):
        """Updates the UI with the generated preview image."""
        try:
            preview_image = self.jxr_loader.tensor_to_pil(tensor)
            if preview_image is None:
                raise RuntimeError("Failed to convert preview to image")

            orig_width, orig_height = preview_image.size
            width_ratio = self.preview_width / orig_width
            height_ratio = self.preview_height / orig_height
            scale_factor = min(width_ratio, height_ratio)
            new_width = int(orig_width * scale_factor)
            new_height = int(orig_height * scale_factor)

            preview_image = preview_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            if new_width < self.preview_width or new_height < self.preview_height:
                bg = Image.new('RGB', (self.preview_width, self.preview_height), (46, 46, 46))
                offset_x = (self.preview_width - new_width) // 2
                offset_y = (self.preview_height - new_height) // 2
                bg.paste(preview_image, (offset_x, offset_y))
                preview_image = bg

            buffer = io.BytesIO()
            preview_image.save(buffer, format='PNG')
            buffer.seek(0)
            preview_tk = ImageTk.PhotoImage(Image.open(buffer))
            self.before_label.config(image=preview_tk, text="")
            self.before_image_ref = preview_tk
            self.show_color_spectrum_from_tensor(tensor, is_before=True)
            self.status_label.config(text="Preview loaded successfully", foreground="#00FF00")

        except Exception as e:
            self._handle_preview_error(str(e))

    def _update_enhancement_controls(self):
        """Enables or disables enhancement controls based on user selection."""
        state = 'normal' if self.use_enhancement.get() else 'disabled'
        self.edge_scale.configure(state=state)

    def _setup_color_controls(self, parent_frame):
        """Sets up image enhancement controls (AI Enhancement and Edge Strength)."""
        enhance_frame = ttk.LabelFrame(self.params_frame, text="Image Enhancement", padding=(10, 10))
        enhance_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.use_enhancement = tk.BooleanVar(value=True)

        # Store reference to the Enhance Checkbox
        self.enhance_checkbox = ttk.Checkbutton(enhance_frame, text="Enable AI Enhancement",
                                                variable=self.use_enhancement,
                                                command=self._update_enhancement_controls)
        self.enhance_checkbox.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky="w")

        enhance_frame.grid_columnconfigure(0, minsize=120)
        enhance_frame.grid_columnconfigure(1, weight=1)
        enhance_frame.grid_columnconfigure(2, minsize=50)

        # Edge Enhancement Controls
        ttk.Label(enhance_frame, text="Strength:").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
        self.edge_strength = tk.DoubleVar(value=0.0)
        self.edge_scale = ttk.Scale(enhance_frame, from_=0.0, to=100.0,
                                    variable=self.edge_strength, orient="horizontal")
        self.edge_scale.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(10, 0))
        self.edge_label = ttk.Label(enhance_frame, text="0%", width=4, anchor="e")
        self.edge_label.grid(row=1, column=2, sticky="e", pady=(10, 0))

        def update_edge_label(*args):
            value = self.edge_strength.get()
            self.edge_label.config(text=f"{int(value)}%")

        self.edge_strength.trace_add("write", update_edge_label)
        self._update_enhancement_controls()

    def convert_image(self):
        """Initiates the image conversion process based on selected mode."""
        mode = self.mode_var.get()
        if mode == "single":
            self._convert_single_file()
        else:
            self._convert_folder()

    def _convert_single_file(self):
        """Converts a single JXR file to JPEG."""
        input_file = self.input_entry.get().strip()
        original_output_file = self.output_entry.get().strip()
        tone_map = self.tonemap_var.get().strip()
        pre_gamma_str = self.pregamma_var.get().strip()
        auto_exposure_str = self.autoexposure_var.get().strip()
        use_enhancement = self.use_enhancement.get()

        # Set color_strength to 100 if enhancement is enabled, else 0
        color_strength = 100.0 if use_enhancement else 0.0
        edge_strength = self.edge_strength.get() if use_enhancement else 0.0

        try:
            validate_files(input_file, original_output_file)
            validate_parameters(tone_map, pre_gamma_str, auto_exposure_str)
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text=str(e), foreground="red")
            return

        pre_gamma = float(pre_gamma_str)
        auto_exposure = float(auto_exposure_str)
        self.jxr_loader.selected_tone_map = tone_map
        self.jxr_loader.selected_pre_gamma = pre_gamma
        self.jxr_loader.selected_auto_exposure = auto_exposure

        self.status_label.config(text="Converting...", foreground="#CCCCCC")
        self.progress['value'] = 0
        self.master.update_idletasks()
        self.convert_btn.config(state='disabled')

        def process_task():
            try:
                # Always reload original tensor to avoid cumulative effects
                tensor, error, metadata = self.jxr_loader.load_jxr(input_file)
                if tensor is None:
                    raise RuntimeError("Failed to load JXR file.")

                # Use the original tensor for processing
                result = self.color_processor.process_image(
                    tensor.clone(),
                    original_output_file,
                    color_strength=color_strength,
                    edge_strength=edge_strength,
                    use_enhancement=use_enhancement
                )

                if result:
                    self.master.after(0, lambda: self.show_color_spectrum(original_output_file, is_before=False))
                    self.safe_update_ui("Conversion successful!", "#00FF00")
                    self.master.after(0, lambda: self.show_preview_from_file(original_output_file, is_before=False))
                else:
                    raise RuntimeError("Image processing failed to produce output")
            except Exception as e:
                error_msg = str(e)
                logging.error(f"Conversion failed with error: {error_msg}")
                self.safe_update_ui(f"Conversion failed: {error_msg}", "red")
                messagebox.showerror("Error", error_msg)
            finally:
                self.safe_enable_convert_button()

        threading.Thread(target=process_task, daemon=True).start()

    def _convert_folder(self):
        """Converts all JXR files in a selected folder to JPEG."""
        folder_path = self.input_entry.get().strip()
        if not folder_path or not os.path.isdir(folder_path):
            error_msg = "Please select a valid folder."
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg, foreground="red")
            return

        jxr_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jxr')]
        if not jxr_files:
            error_msg = "No JXR files found in the selected folder."
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg, foreground="red")
            return

        output_folder = os.path.join(folder_path, "Converted_JPGs")
        os.makedirs(output_folder, exist_ok=True)

        use_enhancement = self.use_enhancement.get()
        color_strength = 100.0 if use_enhancement else 0.0
        edge_strength = self.edge_strength.get() if use_enhancement else 0.0

        self.status_label.config(text=f"Converting {len(jxr_files)} files...", foreground="#CCCCCC")
        self.progress['maximum'] = len(jxr_files)
        self.progress['value'] = 0
        self.convert_btn.config(state='disabled')

        def process_files():
            successful_conversions = 0
            failed_conversions = 0
            failed_files = []
            try:
                for jxr_file in jxr_files:
                    try:
                        input_path = os.path.join(folder_path, jxr_file)
                        final_output = os.path.join(output_folder, f"{os.path.splitext(jxr_file)[0]}.jpg")
                        self.safe_update_ui(f"Processing: {jxr_file}", "#CCCCCC")
                        tensor, error, metadata = self.jxr_loader.load_jxr(input_path)
                        if tensor is None:
                            raise RuntimeError(f"Failed to load {jxr_file}")

                        self.color_processor.process_image(
                            tensor.clone(),
                            final_output,
                            color_strength=color_strength,
                            edge_strength=edge_strength,
                            use_enhancement=use_enhancement
                        )
                        successful_conversions += 1
                        self.safe_increment_progress()
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        logging.error(f"Failed to process {jxr_file}: {e}")
                        failed_conversions += 1
                        failed_files.append(jxr_file)
                        self.safe_increment_progress()

                if failed_conversions == 0:
                    self.safe_update_ui(
                        f"Batch conversion completed! All {successful_conversions} files processed successfully.",
                        "#00FF00"
                    )
                else:
                    error_msg = (f"Batch conversion completed with errors.\n"
                                 f"Successful: {successful_conversions}, Failed: {failed_conversions}\n"
                                 f"Failed files: {', '.join(failed_files)}")
                    self.safe_update_ui(error_msg, "orange" if successful_conversions > 0 else "red")
                    logging.error(error_msg)
            except Exception as e:
                logging.exception("Batch processing error.")
                self.safe_update_ui(f"Batch processing error: {str(e)}", "red")
            finally:
                self.safe_enable_convert_button()

        threading.Thread(target=process_files, daemon=True).start()

    def safe_update_ui(self, message, color):
        """Safely updates the UI status label from any thread."""
        with self.ui_lock:
            self.status_label.config(text=message, foreground=color)

    def safe_increment_progress(self):
        """Safely increments the progress bar from any thread."""
        with self.ui_lock:
            self.progress['value'] += 1
            self.master.update_idletasks()

    def safe_enable_convert_button(self):
        """Safely enables the convert button from any thread."""
        with self.ui_lock:
            self.convert_btn.config(state='normal')


if __name__ == "__main__":
    try:
        app = App("park", "dark")
        root = app.master
        root.title("NVIDIA HDR Converter")
        root.minsize(1760, 840)
        window_width = 1760
        window_height = 840
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int((screen_width - window_width) / 2)
        center_y = int((screen_height - window_height) / 2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        logging.info("Starting main event loop...")
        root.mainloop()
    except Exception as e:
        logging.exception("Error starting application.")
        messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n{str(e)}")
        sys.exit(1)

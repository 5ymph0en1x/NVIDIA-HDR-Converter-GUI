import os
import sys
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import TKinterModernThemes as TKMT
import concurrent.futures
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import requests
from tqdm import tqdm
import time
import traceback

# ============================================================
# CONFIGURATION AND CONSTANTS
# ============================================================
DEFAULT_TONE_MAP = "hable"
DEFAULT_PREGAMMA = "1.2"
DEFAULT_AUTOEXPOSURE = "0.9"
DEFAULT_COLOR_STRENGTH = 0.25

SUPPORTED_TONE_MAPS = {"hable", "reinhard", "filmic", "aces", "uncharted2"}

PREVIEW_WIDTH = 512
PREVIEW_HEIGHT = 288

# Model configuration
PRETRAINED_MODELS = {
    'vgg': models.vgg16,
    'resnet': models.resnet34,
    'densenet': models.densenet121
}


class AttentionBlock(nn.Module):
    """Spatial and channel attention for feature enhancement"""

    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.spatial_pool = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        self.channel_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spatial_att = self.spatial_pool(x)
        channel_att = self.channel_pool(x)
        return x * spatial_att * channel_att


class EdgeEnhancementBlock(nn.Module):
    """Edge detection and enhancement using learned filters"""

    def __init__(self, in_channels):
        super(EdgeEnhancementBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)
        self.edge_weights = nn.Parameter(torch.randn(8, 1, 3, 3))

    def forward(self, x):
        print(f"EdgeEnhancementBlock Input Shape: {x.shape}")  # Debug
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, but got {x.dim()}D tensor")

        # Extract edges using learned filters
        edges = x.mean(dim=1, keepdim=True)  # [batch, 1, H, W]
        print(f"After Mean Shape: {edges.shape}")  # Debug

        edges = F.conv2d(edges, self.edge_weights, padding=1)  # [batch, 8, H, W]
        print(f"After Conv2d Shape: {edges.shape}")  # Debug

        edges = torch.sigmoid(edges)
        print(f"After Sigmoid Shape: {edges.shape}")  # Debug

        if edges.dim() < 2:
            raise ValueError(f"Expected tensor with at least 2 dimensions, but got {edges.dim()}")

        try:
            edges = edges.max(dim=1, keepdim=True)[0]  # [batch, 1, H, W]
            print(f"After Max Shape: {edges.shape}")  # Debug
        except Exception as e:
            print(f"Error during torch.max: {e}")
            raise

        # Enhance features near edges
        features = F.relu(self.conv1(x))
        print(f"After Conv1 Shape: {features.shape}")  # Debug

        enhanced = self.conv2(features)
        print(f"After Conv2 Shape: {enhanced.shape}")  # Debug

        return x + enhanced * edges


class ColorCorrectionNet(nn.Module):
    """Neural network for HDR-aware color correction"""

    def __init__(self):
        super(ColorCorrectionNet, self).__init__()

        # Load pre-trained backbone models
        self.vgg = PRETRAINED_MODELS['vgg'](weights=models.VGG16_Weights.DEFAULT)
        self.resnet = PRETRAINED_MODELS['resnet'](weights=models.ResNet34_Weights.DEFAULT)
        self.densenet = PRETRAINED_MODELS['densenet'](weights=models.DenseNet121_Weights.DEFAULT)

        # Freeze backbone models
        for model in [self.vgg, self.resnet, self.densenet]:
            for param in model.parameters():
                param.requires_grad = False

        # Initial feature extraction and dimension reduction
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Adaptation layers for each network's features
        self.vgg_adapt = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.resnet_adapt = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.densenet_adapt = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, 1),  # 32*3 = 96 channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Edge enhancement blocks
        self.edge_blocks = nn.ModuleList([
            EdgeEnhancementBlock(64) for _ in range(2)
        ])

        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(64) for _ in range(2)
        ])

        # Final color transformation
        self.color_transform = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def extract_vgg_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg.features):
            x = layer(x)
            if i == 4:  # After second conv layer
                features.append(x)
                break
        return features[0]

    def extract_resnet_features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        return x

    def extract_densenet_features(self, x):
        x = self.densenet.features[:4](x)  # First dense block
        return x

    def forward(self, x):
        # Initial feature extraction
        init_features = self.init_conv(x)

        # Extract features from each backbone
        vgg_features = self.extract_vgg_features(x)
        resnet_features = self.extract_resnet_features(x)
        densenet_features = self.extract_densenet_features(x)

        # Ensure all features have the same spatial dimensions
        target_size = (x.shape[2] // 2, x.shape[3] // 2)
        vgg_features = F.interpolate(vgg_features, size=target_size, mode='bilinear', align_corners=True)
        resnet_features = F.interpolate(resnet_features, size=target_size, mode='bilinear', align_corners=True)
        densenet_features = F.interpolate(densenet_features, size=target_size, mode='bilinear', align_corners=True)

        # Adapt features
        vgg_adapted = self.vgg_adapt(vgg_features)
        resnet_adapted = self.resnet_adapt(resnet_features)
        densenet_adapted = self.densenet_adapt(densenet_features)

        # Concatenate features
        combined = torch.cat([vgg_adapted, resnet_adapted, densenet_adapted], dim=1)

        # Fuse features
        fused = self.fusion(combined)

        # Apply edge enhancement and attention
        features = fused
        for edge_block, attention_block in zip(self.edge_blocks, self.attention_blocks):
            features = edge_block(features)
            features = attention_block(features)

        # Final color transformation
        residual = self.color_transform(features)

        # Upsample residual to match input size
        residual = F.interpolate(residual, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        # Add residual and clamp
        enhanced = x + residual
        return torch.clamp(enhanced, -1, 1)


class VibrancyEnhancer:
    """Handles color vibrance enhancement while preserving natural look"""

    def __init__(self):
        self.saturation_boost = 1.2
        self.vibrance_threshold = 0.5

    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV color space"""
        print(f"VibrancyEnhancer.rgb_to_hsv: Input RGB shape: {rgb.shape}")  # Debug
        # Ensure input is in range [0, 1]
        rgb = torch.clamp(rgb, 0, 1)
        print(f"VibrancyEnhancer.rgb_to_hsv: After clamp RGB shape: {rgb.shape}")  # Debug

        # Get max and min values across channels
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        print(f"VibrancyEnhancer.rgb_to_hsv: cmax shape: {cmax.shape}, cmin shape: {cmin.shape}, delta shape: {delta.shape}")  # Debug

        # Calculate Hue
        hue = torch.zeros_like(cmax)
        print(f"VibrancyEnhancer.rgb_to_hsv: Initialized hue shape: {hue.shape}")  # Debug

        # Red is max
        red_mask = (cmax_idx == 0) & (delta != 0)
        hue[red_mask] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[red_mask]
        print(f"VibrancyEnhancer.rgb_to_hsv: After red_mask hue shape: {hue.shape}")  # Debug

        # Green is max
        green_mask = (cmax_idx == 1) & (delta != 0)
        hue[green_mask] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[green_mask]
        print(f"VibrancyEnhancer.rgb_to_hsv: After green_mask hue shape: {hue.shape}")  # Debug

        # Blue is max
        blue_mask = (cmax_idx == 2) & (delta != 0)
        hue[blue_mask] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[blue_mask]
        print(f"VibrancyEnhancer.rgb_to_hsv: After blue_mask hue shape: {hue.shape}")  # Debug

        hue = hue / 6
        print(f"VibrancyEnhancer.rgb_to_hsv: After normalization hue shape: {hue.shape}")  # Debug

        # Calculate Saturation
        saturation = torch.where(cmax != 0, delta / cmax, torch.zeros_like(delta))
        print(f"VibrancyEnhancer.rgb_to_hsv: Saturation shape: {saturation.shape}")  # Debug

        # Value is maximum of RGB channels
        value = cmax
        print(f"VibrancyEnhancer.rgb_to_hsv: Value shape: {value.shape}")  # Debug

        hsv = torch.cat([hue, saturation, value], dim=1)
        print(f"VibrancyEnhancer.rgb_to_hsv: Concatenated HSV shape: {hsv.shape}")  # Debug
        return hsv

    def hsv_to_rgb(self, hsv):
        """Convert HSV to RGB color space"""
        print(f"VibrancyEnhancer.hsv_to_rgb: Input HSV shape: {hsv.shape}")  # Debug
        h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
        print(f"VibrancyEnhancer.hsv_to_rgb: Split H, S, V shapes: {h.shape}, {s.shape}, {v.shape}")  # Debug
        h = h * 6
        print(f"VibrancyEnhancer.hsv_to_rgb: H scaled shape: {h.shape}")  # Debug

        c = v * s
        x = c * (1 - torch.abs(h % 2 - 1))
        m = v - c
        print(f"VibrancyEnhancer.hsv_to_rgb: C, X, M shapes: {c.shape}, {x.shape}, {m.shape}")  # Debug

        rgb = torch.zeros_like(hsv)
        print(f"VibrancyEnhancer.hsv_to_rgb: Initialized RGB shape: {rgb.shape}")  # Debug

        # Masks for different hue ranges
        mask1 = (h < 1)
        mask2 = (h >= 1) & (h < 2)
        mask3 = (h >= 2) & (h < 3)
        mask4 = (h >= 3) & (h < 4)
        mask5 = (h >= 4) & (h < 5)
        mask6 = (h >= 5)

        # Apply masks to assign RGB values
        # For mask1: R = C, G = X, B = 0
        rgb[:, 0][mask1[:, 0]] = c[mask1]
        rgb[:, 1][mask1[:, 0]] = x[mask1]
        rgb[:, 2][mask1[:, 0]] = torch.zeros_like(x[mask1])

        # For mask2: R = X, G = C, B = 0
        rgb[:, 0][mask2[:, 0]] = x[mask2]
        rgb[:, 1][mask2[:, 0]] = c[mask2]
        rgb[:, 2][mask2[:, 0]] = torch.zeros_like(x[mask2])

        # For mask3: R = 0, G = C, B = X
        rgb[:, 0][mask3[:, 0]] = torch.zeros_like(x[mask3])
        rgb[:, 1][mask3[:, 0]] = c[mask3]
        rgb[:, 2][mask3[:, 0]] = x[mask3]

        # For mask4: R = 0, G = X, B = C
        rgb[:, 0][mask4[:, 0]] = torch.zeros_like(x[mask4])
        rgb[:, 1][mask4[:, 0]] = x[mask4]
        rgb[:, 2][mask4[:, 0]] = c[mask4]

        # For mask5: R = X, G = 0, B = C
        rgb[:, 0][mask5[:, 0]] = x[mask5]
        rgb[:, 1][mask5[:, 0]] = torch.zeros_like(x[mask5])
        rgb[:, 2][mask5[:, 0]] = c[mask5]

        # For mask6: R = C, G = 0, B = X
        rgb[:, 0][mask6[:, 0]] = c[mask6]
        rgb[:, 1][mask6[:, 0]] = torch.zeros_like(x[mask6])
        rgb[:, 2][mask6[:, 0]] = x[mask6]

        rgb = rgb + m
        print(f"VibrancyEnhancer.hsv_to_rgb: After adding M RGB shape: {rgb.shape}")  # Debug
        return rgb

    def enhance(self, image_tensor):
        """Apply vibrance enhancement to image tensor"""
        try:
            print(f"VibrancyEnhancer.enhance: Input tensor shape: {image_tensor.shape}")  # Debug
            # Convert to HSV space
            rgb_scaled = (image_tensor + 1) / 2
            print(f"VibrancyEnhancer.enhance: RGB_scaled shape: {rgb_scaled.shape}")  # Debug
            hsv = self.rgb_to_hsv(rgb_scaled)
            print(f"VibrancyEnhancer.enhance: HSV shape: {hsv.shape}")  # Debug

            # Selectively boost saturation based on current levels
            mask = hsv[:, 1:2, :, :] < self.vibrance_threshold
            print(f"VibrancyEnhancer.enhance: Mask shape: {mask.shape}")  # Debug
            hsv[:, 1:2, :, :] = torch.where(
                mask,
                hsv[:, 1:2, :, :] * self.saturation_boost,
                hsv[:, 1:2, :, :]
            )
            print("VibrancyEnhancer.enhance: Saturation boosted based on mask")  # Debug

            # Convert back to RGB
            rgb = self.hsv_to_rgb(hsv)
            print(f"VibrancyEnhancer.enhance: RGB shape after conversion: {rgb.shape}")  # Debug
            return rgb * 2 - 1

        except Exception as e:
            print(f"VibrancyEnhancer.enhance: Failed with error: {e}")  # Debug
            raise


# ============================================================
# COLOR PROCESSING CLASSES
# ============================================================
class HDRColorProcessor:
    """Handles HDR color processing and correction"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing color processor on {self.device}")

        print("Checking and downloading pre-trained models...")
        self.download_models()

        print("Loading models...")
        self.color_net = ColorCorrectionNet().to(self.device)
        self.vibrance_enhancer = VibrancyEnhancer()

        # Set model to evaluation mode
        self.color_net.eval()
        print("Models loaded successfully")

    def download_models(self):
        """Download pre-trained models if not present"""
        model_configs = {
            'vgg16': models.VGG16_Weights.DEFAULT,
            'resnet34': models.ResNet34_Weights.DEFAULT,
            'densenet121': models.DenseNet121_Weights.DEFAULT
        }

        for model_name, weights in model_configs.items():
            print(f"\nChecking {model_name}...")
            try:
                # Get the download URL
                url = weights.url

                # Get the cached file path
                filename = url.split('/')[-1]
                cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
                os.makedirs(cache_dir, exist_ok=True)
                cached_file = os.path.join(cache_dir, filename)

                if os.path.exists(cached_file):
                    print(f"{model_name} already cached at {cached_file}")
                    continue

                # Download the file with progress bar
                print(f"Downloading {model_name} from {url}")
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))

                with open(cached_file, 'wb') as f, tqdm(
                        desc=f"Downloading {model_name}",
                        total=total_size,
                        unit='iB',
                        unit_scale=True
                ) as pbar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        pbar.update(size)

                print(f"Successfully downloaded {model_name}")

            except Exception as e:
                print(f"Warning: Error downloading {model_name}: {str(e)}")
                print("Will attempt to download during model initialization")

    def _preprocess_image(self, image):
        """Preprocess image for the neural network"""
        try:
            # Ensure the image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            tensor = transform(image).unsqueeze(0).to(self.device)
            print(f"DEBUG: Preprocessed tensor shape: {tensor.shape}")  # Debug
            return tensor
        except Exception as e:
            print(f"DEBUG: Preprocessing failed: {str(e)}")  # Debug
            raise

    def _postprocess_image(self, tensor):
        """Convert tensor back to PIL Image"""
        tensor = (tensor + 1) / 2.0
        tensor = tensor.squeeze(0).cpu()
        return transforms.ToPILImage()(tensor)

    def process_image(self, input_path, output_path, color_features=None, strength=0.7):
        """Process image with HDR-aware color correction"""
        try:
            print(f"DEBUG: Opening input file: {input_path}")  # Debug
            # Open image without specifying format
            image = Image.open(input_path)

            print(f"DEBUG: Image format: {image.format}")  # Debug
            print(f"DEBUG: Image mode: {image.mode}")  # Debug
            print(f"DEBUG: Image size: {image.size}")  # Debug

            if image.mode != 'RGB':
                print(f"DEBUG: Converting image from {image.mode} to RGB")  # Debug
                image = image.convert('RGB')

            # Transform to tensor
            image_tensor = self._preprocess_image(image)

            with torch.no_grad():
                # Apply color correction
                enhanced = self.color_net(image_tensor)

                # Enhance vibrance
                enhanced = self.vibrance_enhancer.enhance(enhanced)

                # Blend with original based on strength
                blend_factor = strength
                blended = (
                        blend_factor * enhanced +
                        (1.0 - blend_factor) * image_tensor
                )

                # Convert to image and save
                result_image = self._postprocess_image(blended)
                result_image.save(output_path, 'JPEG', quality=95)

                return result_image

        except Exception as e:
            traceback.print_exc()  # Print full traceback
            print(f"DEBUG: Color correction failed: {str(e)}")  # Debug
            print(f"DEBUG: Attempting fallback conversion")  # Debug

            # Fallback: direct conversion if AI processing fails
            try:
                image = Image.open(input_path)
                print(f"DEBUG: Fallback - Image format: {image.format}, mode: {image.mode}")  # Debug
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(output_path, 'JPEG', quality=95)
                print("DEBUG: Fallback conversion successful")  # Debug
                return image
            except Exception as fallback_error:
                print(f"DEBUG: Fallback conversion also failed: {str(fallback_error)}")  # Debug
                raise

    def apply_color_preservation(self, image_tensor, reference_tensor, strength=0.7):
        """Apply color preservation using color transfer in LAB space"""
        # Convert to LAB space using approximate conversion
        image_lab = self._rgb_to_lab(image_tensor)
        reference_lab = self._rgb_to_lab(reference_tensor)

        # Compute statistics
        image_mean = torch.mean(image_lab, dim=(2, 3), keepdim=True)
        image_std = torch.std(image_lab, dim=(2, 3), keepdim=True)
        ref_mean = torch.mean(reference_lab, dim=(2, 3), keepdim=True)
        ref_std = torch.std(reference_lab, dim=(2, 3), keepdim=True)

        # Apply color transfer with strength parameter
        normalized = (image_lab - image_mean) / (image_std + 1e-6)
        matched = normalized * (ref_std * strength + image_std * (1 - strength)) + \
                  (ref_mean * strength + image_mean * (1 - strength))

        # Convert back to RGB
        return self._lab_to_rgb(matched)

    def _rgb_to_lab(self, rgb):
        """Convert RGB to LAB color space"""
        # RGB to XYZ
        rgb = (rgb + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
        print(f"HDRColorProcessor._rgb_to_lab: Input RGB shape: {rgb.shape}")  # Debug

        # Define conversion matrices
        rgb_to_xyz = torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ]).to(rgb.device)

        # Convert to XYZ
        xyz = torch.einsum('ci,bchw->bihw', rgb_to_xyz, rgb)
        print(f"HDRColorProcessor._rgb_to_lab: After RGB to XYZ shape: {xyz.shape}")  # Debug

        # XYZ to LAB
        # Reference white point (D65)
        xyz_ref = torch.tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1).to(rgb.device)

        # Scale XYZ relative to reference white
        xyz_scaled = xyz / (xyz_ref + 1e-10)
        print(f"HDRColorProcessor._rgb_to_lab: Scaled XYZ shape: {xyz_scaled.shape}")  # Debug

        # Apply cube root transformation
        mask = xyz_scaled > 0.008856
        xyz_scaled[mask] = torch.pow(xyz_scaled[mask], 1 / 3)
        xyz_scaled[~mask] = 7.787 * xyz_scaled[~mask] + 16 / 116
        print(f"HDRColorProcessor._rgb_to_lab: After cube root transformation shape: {xyz_scaled.shape}")  # Debug

        # Calculate LAB channels
        L = (116 * xyz_scaled[:, 1:2] - 16)
        a = 500 * (xyz_scaled[:, 0:1] - xyz_scaled[:, 1:2])
        b = 200 * (xyz_scaled[:, 1:2] - xyz_scaled[:, 2:3])

        lab = torch.cat([L, a, b], dim=1)
        print(f"HDRColorProcessor._rgb_to_lab: LAB shape: {lab.shape}")  # Debug
        return lab

    def _lab_to_rgb(self, lab):
        """Convert LAB to RGB color space"""
        # LAB to XYZ
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
        print(f"HDRColorProcessor._lab_to_rgb: Split L, a, b shapes: {L.shape}, {a.shape}, {b.shape}")  # Debug

        # Reference white point (D65)
        xyz_ref = torch.tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1).to(lab.device)

        # Calculate intermediate values
        y = (L + 16) / 116
        x = a / 500 + y
        z = y - b / 200
        print(f"HDRColorProcessor._lab_to_rgb: Intermediate x, y, z shapes: {x.shape}, {y.shape}, {z.shape}")  # Debug

        # Apply inverse cube root transformation
        xyz = torch.cat([x, y, z], dim=1)
        mask = xyz > 0.206893
        xyz[mask] = torch.pow(xyz[mask], 3)
        xyz[~mask] = (xyz[~mask] - 16 / 116) / 7.787
        print(f"HDRColorProcessor._lab_to_rgb: After inverse transformation XYZ shape: {xyz.shape}")  # Debug

        # Scale by reference white
        xyz = xyz * xyz_ref
        print(f"HDRColorProcessor._lab_to_rgb: Scaled XYZ shape: {xyz.shape}")  # Debug

        # XYZ to RGB
        xyz_to_rgb = torch.tensor([
            [3.240479, -1.537150, -0.498535],
            [-0.969256, 1.875992, 0.041556],
            [0.055648, -0.204043, 1.057311]
        ]).to(lab.device)

        rgb = torch.einsum('ci,bihw->bchw', xyz_to_rgb, xyz)
        print(f"HDRColorProcessor._lab_to_rgb: After XYZ to RGB shape: {rgb.shape}")  # Debug

        # Clip and scale to [-1, 1]
        rgb = torch.clamp(rgb, 0, 1) * 2 - 1
        print(f"HDRColorProcessor._lab_to_rgb: Final RGB shape: {rgb.shape}")  # Debug
        return rgb


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def resource_path(relative_path):
    """Get absolute path to resource, works for development and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def decode_jxr_to_tiff(jxr_path: str) -> str:
    """Decode JXR file to TIFF format"""
    JXRDEC_PATH = resource_path("JXRDecApp.exe")
    if not os.path.exists(JXRDEC_PATH):
        raise FileNotFoundError("JXRDecApp.exe not found")

    if not os.path.exists(jxr_path):
        raise FileNotFoundError(f"Input JXR file not found: {jxr_path}")

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tiff_path = tmp.name

    cmd = [JXRDEC_PATH, "-i", jxr_path, "-o", tiff_path]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)

    if not os.path.exists(tiff_path) or os.path.getsize(tiff_path) == 0:
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=cmd,
            stderr="Failed to create valid TIFF file"
        )
    return tiff_path


def convert_tiff_to_png(tiff_path: str) -> str:
    """Convert TIFF to PNG format"""
    temp_png = tiff_path + ".png"
    try:
        subprocess.run(['magick', 'convert', tiff_path, temp_png],
                       check=True, capture_output=True, text=True)
        return temp_png
    except:
        try:
            img = Image.open(tiff_path)
            img.save(temp_png, 'PNG')
            return temp_png
        except Exception as e:
            raise RuntimeError(f"Failed to convert TIFF to PNG: {str(e)}")


def run_hdrfix(input_file: str, output_file: str, tone_map: str,
               pre_gamma: str, auto_exposure: str) -> str:
    """Run HDR conversion with specified parameters and verify output"""
    HDRFIX_PATH = resource_path("hdrfix.exe")
    if not os.path.exists(HDRFIX_PATH):
        raise FileNotFoundError("hdrfix.exe not found")

    # Force PNG output regardless of the extension provided
    output_png = os.path.splitext(output_file)[0] + '.png'

    print(f"DEBUG: Running HDRfix command with PNG output: {output_png}")  # Debug
    cmd = [
        HDRFIX_PATH, input_file, output_png,
        "--tone-map", tone_map,
        "--pre-gamma", pre_gamma,
        "--auto-exposure", auto_exposure
    ]

    # Run HDRfix and capture output
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"DEBUG: HDRfix stdout: {result.stdout}")  # Debug
    print(f"DEBUG: HDRfix stderr: {result.stderr}")  # Debug

    # Verify output file exists and is valid
    if not os.path.exists(output_png):
        raise FileNotFoundError(f"HDRfix failed to create output file: {output_png}")

    # Return the actual output file path
    return output_png


def validate_files(input_file: str, output_file: str) -> None:
    """Validate input and output file paths"""
    if not input_file or not os.path.exists(input_file):
        raise FileNotFoundError("Input file does not exist")
    if not output_file:
        raise ValueError("Output file path not specified")


def validate_parameters(tone_map: str, pre_gamma: str, auto_exposure: str) -> None:
    """Validate conversion parameters"""
    if tone_map not in SUPPORTED_TONE_MAPS:
        raise ValueError(f"Unsupported tone map: {tone_map}")
    try:
        float(pre_gamma)
        float(auto_exposure)
    except ValueError:
        raise ValueError("Pre-gamma and auto-exposure must be numeric")


# ============================================================
# MAIN APP CLASS
# ============================================================
class App(TKMT.ThemedTKinterFrame):
    def __init__(self, theme="park", mode="dark"):
        super().__init__("NVIDIA HDR Converter", theme, mode)

        # Initialize color processing components
        self.color_processor = HDRColorProcessor()

        # Create main container frame
        main_frame = ttk.Frame(self.master, padding=(10, 10))
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Create left and right frames
        left_frame = ttk.Frame(main_frame, padding=(10, 10))
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        right_frame = ttk.Frame(main_frame, padding=(10, 10))
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nw")

        # Setup GUI components
        self._setup_mode_selection(left_frame)
        self._setup_file_selection(left_frame)
        self._setup_parameters(left_frame)
        self._setup_color_controls()
        self._setup_conversion_button(left_frame)
        self._setup_progress_bar(left_frame)
        self._setup_status_label(left_frame)
        self._setup_previews(right_frame)

        self.before_image_ref = None
        self.after_image_ref = None

        # Thread-safe UI updates
        self.ui_lock = threading.Lock()

        # Run the application
        self.run()

    def _setup_color_controls(self):
        """Setup color enhancement controls"""
        color_frame = ttk.LabelFrame(self.params_frame, text="Color Enhancement", padding=(10, 10))
        color_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        # Enable/disable color correction
        self.use_color_correction = tk.BooleanVar(value=True)
        color_checkbox = ttk.Checkbutton(
            color_frame,
            text="Enable AI Color Enhancement",
            variable=self.use_color_correction
        )
        color_checkbox.grid(row=0, column=0, pady=5)

        # Color preservation strength
        strength_frame = ttk.Frame(color_frame)
        strength_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)

        ttk.Label(strength_frame, text="Color Preservation:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.color_strength = tk.DoubleVar(value=DEFAULT_COLOR_STRENGTH)
        color_scale = ttk.Scale(
            strength_frame,
            from_=0.0,
            to=1.0,
            variable=self.color_strength,
            orient="horizontal",
            length=200
        )
        color_scale.grid(row=0, column=1, sticky="ew")

        # Add value label with fixed width to prevent resizing, initialized with the default value
        self.strength_label = ttk.Label(strength_frame, text=f"{DEFAULT_COLOR_STRENGTH * 100:.0f}%", width=5,
                                        anchor="e")
        self.strength_label.grid(row=0, column=2, padx=(10, 0))

        # Update label when slider changes
        def update_strength_label(*args):
            value = self.color_strength.get() * 100
            self.strength_label.config(text=f"{value:.0f}%")

        self.color_strength.trace_add("write", update_strength_label)

    def _setup_mode_selection(self, parent_frame):
        """Setup conversion mode selection"""
        mode_frame = ttk.LabelFrame(parent_frame, text="Mode Selection", padding=(10, 10))
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.mode_var = tk.StringVar(value="single")

        single_radio = ttk.Radiobutton(
            mode_frame,
            text="Single File",
            variable=self.mode_var,
            value="single",
            command=self.update_mode
        )
        single_radio.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        folder_radio = ttk.Radiobutton(
            mode_frame,
            text="Folder",
            variable=self.mode_var,
            value="folder",
            command=self.update_mode
        )
        folder_radio.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    def _setup_file_selection(self, parent_frame):
        """Setup file selection controls"""
        self.file_frame = ttk.LabelFrame(parent_frame, text="File Selection", padding=(10, 10))
        self.file_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        # Input selection
        self.input_label = ttk.Label(self.file_frame, text="Input JXR:")
        self.input_label.grid(row=0, column=0, sticky="e", padx=(0, 5), pady=5)
        self.input_entry = ttk.Entry(self.file_frame, width=40)
        self.input_entry.grid(row=0, column=1, padx=(0, 5), pady=5)
        self.browse_button = ttk.Button(
            self.file_frame,
            text="Browse...",
            command=self.browse_input
        )
        self.browse_button.grid(row=0, column=2, padx=(0, 5), pady=5)

        # Output selection
        self.output_label = ttk.Label(self.file_frame, text="Output JPG:")
        self.output_label.grid(row=1, column=0, sticky="e", padx=(0, 5), pady=5)
        self.output_entry = ttk.Entry(self.file_frame, width=40)
        self.output_entry.grid(row=1, column=1, padx=(0, 5), pady=5)
        self.output_browse_button = ttk.Button(
            self.file_frame,
            text="Browse...",
            command=self.browse_output
        )
        self.output_browse_button.grid(row=1, column=2, padx=(0, 5), pady=5)

    def _setup_parameters(self, parent_frame):
        """Setup conversion parameters controls"""
        self.params_frame = ttk.LabelFrame(parent_frame, text="Parameters", padding=(10, 10))
        self.params_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        # Tone Map
        ttk.Label(self.params_frame, text="Tone Map:").grid(
            row=0, column=0, sticky="e", padx=(0, 5), pady=5
        )
        self.tonemap_var = tk.StringVar(value=DEFAULT_TONE_MAP)
        tone_map_values = sorted(list(SUPPORTED_TONE_MAPS))
        tone_map_dropdown = ttk.Combobox(
            self.params_frame,
            textvariable=self.tonemap_var,
            values=tone_map_values,
            state='readonly'
        )
        tone_map_dropdown.grid(row=0, column=1, padx=(0, 5), pady=5)

        # Pre-gamma
        ttk.Label(self.params_frame, text="Pre-gamma:").grid(
            row=1, column=0, sticky="e", padx=(0, 5), pady=5
        )
        self.pregamma_var = tk.StringVar(value=DEFAULT_PREGAMMA)
        ttk.Entry(
            self.params_frame,
            textvariable=self.pregamma_var,
            width=15
        ).grid(row=1, column=1, padx=(0, 5), pady=5)

        # Auto-exposure
        ttk.Label(self.params_frame, text="Auto-exposure:").grid(
            row=2, column=0, sticky="e", padx=(0, 5), pady=5
        )
        self.autoexposure_var = tk.StringVar(value=DEFAULT_AUTOEXPOSURE)
        ttk.Entry(
            self.params_frame,
            textvariable=self.autoexposure_var,
            width=15
        ).grid(row=2, column=1, padx=(0, 5), pady=5)

    def _setup_conversion_button(self, parent_frame):
        """Setup conversion button"""
        self.convert_btn = ttk.Button(
            parent_frame,
            text="Convert",
            command=self.convert_image
        )
        self.convert_btn.grid(row=3, column=0, pady=(10, 10), sticky="ew")

    def _setup_progress_bar(self, parent_frame):
        """Setup progress bar"""
        self.progress = ttk.Progressbar(
            parent_frame,
            orient="horizontal",
            mode="determinate"
        )
        self.progress.grid(row=4, column=0, sticky="ew", pady=(0, 10))

    def _setup_status_label(self, parent_frame):
        """Setup status label"""
        self.status_label = ttk.Label(parent_frame, text="", foreground="#CCCCCC")
        self.status_label.grid(row=5, column=0, sticky="w", pady=(0, 10))

    def _setup_previews(self, parent_frame):
        """Setup preview panels"""
        previews_frame = ttk.Frame(parent_frame, padding=(10, 10))
        previews_frame.grid(row=0, column=0, sticky="nsew")

        # Before preview
        before_frame = ttk.LabelFrame(
            previews_frame,
            text="Before Conversion",
            padding=(10, 10)
        )
        before_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        before_canvas = tk.Canvas(
            before_frame,
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT
        )
        before_canvas.pack(padx=10, pady=10)
        self.before_label = ttk.Label(before_canvas)
        before_canvas.create_window(
            PREVIEW_WIDTH // 2,
            PREVIEW_HEIGHT // 2,
            window=self.before_label
        )

        # After preview
        after_frame = ttk.LabelFrame(
            previews_frame,
            text="After Conversion",
            padding=(10, 10)
        )
        after_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        after_canvas = tk.Canvas(
            after_frame,
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT
        )
        after_canvas.pack(padx=10, pady=10)
        self.after_label = ttk.Label(after_canvas)
        after_canvas.create_window(
            PREVIEW_WIDTH // 2,
            PREVIEW_HEIGHT // 2,
            window=self.after_label
        )

    def update_mode(self):
        """Handle mode selection changes"""
        mode = self.mode_var.get()
        if mode == "single":
            self.file_frame.config(text="File Selection")
            self.input_label.config(text="Input JXR:")
            self.output_label.grid()
            self.output_entry.grid()
            self.output_browse_button.grid()
        else:
            self.file_frame.config(text="Folder Selection")
            self.input_label.config(text="Input Folder:")
            self.output_label.grid_remove()
            self.output_entry.grid_remove()
            self.output_browse_button.grid_remove()
            self.output_entry.delete(0, tk.END)
            self.before_label.config(image="", text="No Preview")
            self.before_image_ref = None
            self.after_label.config(image="", text="No Preview")
            self.after_image_ref = None

    def browse_input(self):
        """Handle input file/folder selection"""
        mode = self.mode_var.get()
        if mode == "single":
            filename = filedialog.askopenfilename(
                title="Select Input JXR File",
                filetypes=[("JXR files", "*.jxr"), ("All files", "*.*")]
            )
            if filename:
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, filename)
                self.status_label.config(
                    text="Loading preview...",
                    foreground="#CCCCCC"
                )
                self.master.update_idletasks()
                self.create_preview_from_jxr(filename)
        else:
            foldername = filedialog.askdirectory(
                title="Select Folder Containing JXR Files"
            )
            if foldername:
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, foldername)
                self.status_label.config(
                    text="Folder selected. Ready to convert all JXR files.",
                    foreground="#CCCCCC"
                )
                self.before_label.config(image="", text="No Preview")
                self.before_image_ref = None
                self.after_label.config(image="", text="No Preview")
                self.after_image_ref = None

    def browse_output(self):
        """Handle output file selection"""
        filename = filedialog.asksaveasfilename(
            title="Select Output JPG File",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if filename:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, filename)

    def convert_image(self):
        """Handle image conversion based on selected mode"""
        mode = self.mode_var.get()
        if mode == "single":
            self._convert_single_file()
        else:
            self._convert_folder()

    def _convert_single_file(self):
        """Convert a single JXR file"""
        input_file = self.input_entry.get().strip()
        original_output_file = self.output_entry.get().strip()

        # Validate parameters
        try:
            validate_files(input_file, original_output_file)
            validate_parameters(
                self.tonemap_var.get().strip(),
                self.pregamma_var.get().strip(),
                self.autoexposure_var.get().strip()
            )
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text=str(e), foreground="red")
            return

        # Prepare for conversion
        self.status_label.config(text="Converting...", foreground="#CCCCCC")
        self.progress['value'] = 0
        self.master.update_idletasks()
        self.convert_btn.config(state='disabled')

        def process_task():
            # Create a local copy of the output path that we can modify if needed
            output_file = original_output_file
            temp_hdr_output = None

            try:
                # Create temporary file for HDR conversion result
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_hdr_output = tmp.name

                print(f"DEBUG: Running HDRfix on {input_file}")  # Debug
                # Run HDR conversion to PNG first
                actual_output = run_hdrfix(
                    input_file,
                    temp_hdr_output,
                    self.tonemap_var.get().strip(),
                    self.pregamma_var.get().strip(),
                    self.autoexposure_var.get().strip()
                )
                temp_hdr_output = actual_output  # Use the actual PNG output path

                print(f"DEBUG: HDRfix completed, checking output file: {temp_hdr_output}")  # Debug
                if not os.path.exists(temp_hdr_output):
                    raise FileNotFoundError(f"HDRfix failed to create output file: {temp_hdr_output}")

                # Verify PNG file is valid
                try:
                    with Image.open(temp_hdr_output) as test_img:
                        print(
                            f"DEBUG: Temporary PNG properties - Format: {test_img.format}, Mode: {test_img.mode}, Size: {test_img.size}")  # Debug
                        # Force load the image data to verify it's valid
                        test_img.load()
                except Exception as png_error:
                    raise RuntimeError(f"HDRfix created invalid PNG file: {str(png_error)}")

                try:
                    # Try to remove existing output file
                    if os.path.exists(output_file):
                        os.remove(output_file)
                except PermissionError:
                    # If file is locked or can't be removed, create new filename
                    base, ext = os.path.splitext(output_file)
                    output_file = f"{base}_{int(time.time())}{ext}"

                if self.use_color_correction.get():
                    print("DEBUG: Applying color correction")  # Debug
                    try:
                        # Process with AI color enhancement
                        self.color_processor.process_image(
                            temp_hdr_output,
                            output_file,
                            strength=self.color_strength.get()
                        )
                    except Exception as e:
                        print(f"DEBUG: Color correction failed: {str(e)}, attempting direct conversion")  # Debug
                        # Fallback to direct conversion
                        img = Image.open(temp_hdr_output)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(output_file, 'JPEG', quality=95)
                        img.close()
                else:
                    print("DEBUG: Converting directly to JPG")  # Debug
                    # Convert PNG to JPG directly without AI processing
                    img = Image.open(temp_hdr_output)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(output_file, 'JPEG', quality=95)
                    img.close()

                print(f"DEBUG: Checking final output: {output_file}")  # Debug
                if not os.path.exists(output_file):
                    raise FileNotFoundError(f"Failed to create final output file: {output_file}")

                self.safe_update_ui(
                    "Conversion successful!",
                    "#00FF00"
                )

                # Update preview
                if os.path.exists(output_file):
                    self.show_preview_from_file(output_file, is_before=False)

            except Exception as e:
                print(f"DEBUG: Conversion failed with error: {str(e)}")  # Debug
                self.safe_update_ui(
                    f"Conversion failed: {str(e)}",
                    "red"
                )
                messagebox.showerror("Error", str(e))

            finally:
                # Clean up temporary file
                if temp_hdr_output and os.path.exists(temp_hdr_output):
                    try:
                        os.remove(temp_hdr_output)
                    except:
                        pass
                self.safe_enable_convert_button()

        # Start processing in a separate thread
        threading.Thread(target=process_task, daemon=True).start()

    def _convert_folder(self):
        """Convert all JXR files in a folder"""
        folder_path = self.input_entry.get().strip()

        # Validate folder
        if not folder_path or not os.path.isdir(folder_path):
            error_msg = "Please select a valid folder."
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg, foreground="red")
            return

        # Find JXR files
        jxr_files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith('.jxr')]
        if not jxr_files:
            error_msg = "No JXR files found in the selected folder."
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg, foreground="red")
            return

        # Create output folder
        output_folder = os.path.join(folder_path, "Converted_JPGs")
        os.makedirs(output_folder, exist_ok=True)

        # Prepare for conversion
        self.status_label.config(
            text=f"Converting {len(jxr_files)} files...",
            foreground="#CCCCCC"
        )
        self.progress['maximum'] = len(jxr_files)
        self.progress['value'] = 0
        self.convert_btn.config(state='disabled')

        def process_file(jxr_file):
            """Process a single JXR file in the batch conversion"""
            print(f"\nDEBUG: Processing {jxr_file}")  # Debug
            input_path = os.path.join(folder_path, jxr_file)
            temp_hdr_output = os.path.join(
                output_folder,
                f"temp_hdr_{os.path.splitext(jxr_file)[0]}.png"  # Changed to PNG
            )
            final_output = os.path.join(
                output_folder,
                f"{os.path.splitext(jxr_file)[0]}.jpg"
            )

            try:
                print(f"DEBUG: Running HDRfix on {input_path}")  # Debug
                # Run HDR conversion to PNG
                actual_output = run_hdrfix(
                    input_path,
                    temp_hdr_output,
                    self.tonemap_var.get().strip(),
                    self.pregamma_var.get().strip(),
                    self.autoexposure_var.get().strip()
                )
                temp_hdr_output = actual_output  # Use the actual PNG output path

                if not os.path.exists(temp_hdr_output):
                    raise FileNotFoundError(f"HDRfix failed to create output file: {temp_hdr_output}")

                # Verify PNG file is valid
                try:
                    with Image.open(temp_hdr_output) as test_img:
                        print(
                            f"DEBUG: Temporary PNG properties - Format: {test_img.format}, Mode: {test_img.mode}, Size: {test_img.size}")  # Debug
                        # Force load the image data to verify it's valid
                        test_img.load()
                except Exception as png_error:
                    raise RuntimeError(f"HDRfix created invalid PNG file: {str(png_error)}")

                if self.use_color_correction.get():
                    print(f"DEBUG: Applying color correction for {jxr_file}")  # Debug
                    try:
                        # Process with AI color enhancement
                        self.color_processor.process_image(
                            temp_hdr_output,
                            final_output,
                            strength=self.color_strength.get()
                        )
                    except Exception as e:
                        print(f"DEBUG: Color correction failed for {jxr_file}: {str(e)}, attempting direct conversion")  # Debug
                        # Fallback to direct conversion
                        img = Image.open(temp_hdr_output)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(final_output, 'JPEG', quality=95)
                        img.close()
                else:
                    print(f"DEBUG: Converting directly to JPG for {jxr_file}")  # Debug
                    # Convert PNG to JPG directly without AI processing
                    img = Image.open(temp_hdr_output)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(final_output, 'JPEG', quality=95)
                    img.close()

                print(f"DEBUG: Checking final output: {final_output}")  # Debug
                if not os.path.exists(final_output):
                    raise FileNotFoundError(f"Failed to create final output file: {final_output}")

                self.safe_increment_progress()
                print(f"DEBUG: Successfully processed {jxr_file}")  # Debug
                return True

            except Exception as e:
                print(f"DEBUG: Error processing {jxr_file}: {str(e)}")  # Debug
                return False

            finally:
                # Clean up temporary file
                if os.path.exists(temp_hdr_output):
                    try:
                        os.remove(temp_hdr_output)
                    except:
                        pass

        def batch_process():
            """Handle the batch processing of all files"""
            successful_conversions = 0
            failed_conversions = 0
            failed_files = []

            try:
                # Determine optimal number of worker threads
                cpu_cores = os.cpu_count() or 1
                max_workers = max(1, cpu_cores // 2)  # Use half of available cores
                print(f"DEBUG: Using {max_workers} worker threads for batch processing")  # Debug

                # Process files in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all files for processing
                    future_to_file = {executor.submit(process_file, f): f for f in jxr_files}

                    # Process results as they complete
                    for future in concurrent.futures.as_completed(future_to_file):
                        jxr_file = future_to_file[future]
                        try:
                            success = future.result()
                            if success:
                                successful_conversions += 1
                            else:
                                failed_conversions += 1
                                failed_files.append(jxr_file)
                        except Exception as e:
                            print(f"DEBUG: Failed to process {jxr_file}: {str(e)}")  # Debug
                            failed_conversions += 1
                            failed_files.append(jxr_file)

                # Update UI with final status
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
                    print(f"DEBUG: {error_msg}")  # Debug

            except Exception as e:
                print(f"DEBUG: Batch processing error: {str(e)}")  # Debug
                self.safe_update_ui(
                    f"Batch processing error: {str(e)}",
                    "red"
                )

            finally:
                self.safe_enable_convert_button()

        # Start batch processing in a separate thread
        threading.Thread(target=batch_process, daemon=True).start()

    def create_preview_from_jxr(self, jxr_file: str):
        """Create preview from JXR file"""
        if not jxr_file.lower().endswith('.jxr'):
            self.status_label.config(text="Not a JXR file", foreground="red")
            self.before_label.config(image="", text="No Preview")
            self.before_image_ref = None
            return

        tiff_path = None
        try:
            tiff_path = decode_jxr_to_tiff(jxr_file)
            self.show_preview_from_file(tiff_path, is_before=True)
            self.status_label.config(
                text="Preview loaded successfully",
                foreground="#00FF00"
            )
        except FileNotFoundError as e:
            self.status_label.config(text=str(e), foreground="red")
            messagebox.showerror("Error", str(e))
            self.before_label.config(image="", text="No Preview")
            self.before_image_ref = None
        except subprocess.CalledProcessError:
            self.status_label.config(
                text="Preview failed: JXRDecApp error",
                foreground="red"
            )
            self.before_label.config(text="Preview Failed")
            self.before_image_ref = None
        except Exception as e:
            self.status_label.config(
                text=f"Preview failed: {str(e)}",
                foreground="red"
            )
            self.before_label.config(text="Preview Failed")
            self.before_image_ref = None
        finally:
            if tiff_path and os.path.exists(tiff_path):
                try:
                    os.remove(tiff_path)
                except:
                    pass

    def show_preview_from_file(self, filepath: str, is_before: bool):
        """Show image preview"""
        temp_png = None
        label = self.before_label if is_before else self.after_label

        # Clear existing image reference first
        if is_before:
            if self.before_image_ref:
                self.before_label.config(image="")
                self.before_image_ref = None
        else:
            if self.after_image_ref:
                self.after_label.config(image="")
                self.after_image_ref = None

        try:
            if filepath.lower().endswith(('.tif', '.tiff')):
                temp_png = convert_tiff_to_png(filepath)
                filepath = temp_png

            # Load and convert image
            img = Image.open(filepath).convert('RGB')
            img_ratio = img.width / img.height
            target_ratio = PREVIEW_WIDTH / PREVIEW_HEIGHT

            # Calculate new size maintaining aspect ratio
            if img_ratio > target_ratio:
                new_size = (PREVIEW_WIDTH, int(PREVIEW_WIDTH / img_ratio))
            else:
                new_size = (int(PREVIEW_HEIGHT * img_ratio), PREVIEW_HEIGHT)

            # Resize and create PhotoImage
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            # Update label with new image
            label.config(image=img_tk, text="")
            if is_before:
                self.before_image_ref = img_tk
            else:
                self.after_image_ref = img_tk

            # Force garbage collection of the old image
            img.close()

        except Exception as e:
            label_text = f"Preview Error: {str(e)}"
            label.config(image="", text=label_text)
            if is_before:
                self.before_image_ref = None
            else:
                self.after_image_ref = None

        finally:
            if temp_png and os.path.exists(temp_png):
                try:
                    os.remove(temp_png)
                except:
                    pass

    def safe_update_ui(self, message, color):
        """Thread-safe UI update"""
        with self.ui_lock:
            self.status_label.config(text=message, foreground=color)

    def safe_increment_progress(self):
        """Thread-safe progress bar update"""
        with self.ui_lock:
            self.progress['value'] += 1
            self.master.update_idletasks()

    def safe_enable_convert_button(self):
        """Thread-safe convert button enable"""
        with self.ui_lock:
            self.convert_btn.config(state='normal')


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    App("park", "dark")

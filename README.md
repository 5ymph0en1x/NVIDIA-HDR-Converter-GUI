# NVIDIA HDR Converter GUI
A powerful GUI application for converting NVIDIA JXR (HDR) screenshots to high-quality JPEG images with AI-powered color enhancement.

<p align="center">
  <img width="1024" src="interface.jpg">
</p>

## Features

### Core Functionality
- Convert NVIDIA JXR (HDR) screenshots to JPEG format
- Advanced HDR tone mapping with multiple algorithms
- AI-powered color enhancement using deep learning models
- Batch processing support
- Live preview functionality

### AI Color Enhancement
- Neural network-based color correction using pretrained models
- Edge preservation and enhancement
- Intelligent contrast adjustment
- Vibrance enhancement with natural look preservation
- Automatic model download and caching

### Supported Tone Mapping Algorithms
- Hable (Default)
- Reinhard
- Filmic
- ACES
- Uncharted 2

### User Interface
- Modern, dark-themed GUI
- Real-time preview
- Progress tracking
- Batch processing support
- Customizable parameters

## Requirements

### Python Dependencies
- PyTorch
- torchvision
- Pillow
- TKinterModernThemes
- tqdm
- requests

### External Dependencies
- JXRDecApp.exe (for JXR decoding)
- hdrfix.exe (for HDR processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-hdr-converter.git
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Place required executables (JXRDecApp.exe and hdrfix.exe) in the same directory as the script.

## Usage

### Single File Conversion
1. Launch the application
2. Select "Single File" mode
3. Choose input JXR file
4. Select output JPEG location
5. Adjust parameters if needed
6. Click "Convert"

### Batch Processing
1. Launch the application
2. Select "Folder" mode
3. Choose folder containing JXR files
4. Adjust parameters if needed
5. Click "Convert"
- Converted files will be saved in a "Converted_JPGs" subfolder

### Parameters
- **Tone Map**: Select HDR tone mapping algorithm
- **Pre-gamma**: Adjust gamma before tone mapping
- **Auto-exposure**: Control exposure adjustment
- **Color Enhancement**: Toggle AI-powered color enhancement
- **Color Preservation**: Adjust strength of color enhancement

## Technical Details

### AI Architecture
- Uses VGG16 backbone for feature extraction
- Custom edge enhancement and attention mechanisms
- Adaptive color transformation
- Multi-scale feature processing

### Performance Optimization
- Efficient memory usage
- Multi-threaded batch processing
- Automatic CPU/GPU detection
- Progressive image loading
- Cached model storage

## Acknowledgments
- Uses PyTorch pretrained models
- GUI theme based on TKinterModernThemes

# build.spec
# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import imagecodecs
import napari.plugins
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None
print(dir(imagecodecs))
napari.plugins.plugin_manager.discover()

# Définir les chemins vers les DLL supplémentaires requises par imagecodecs et MKL
additional_dlls = [
    # DLL Intel MKL
    ("C:\\Users\\*****\\anaconda3\\pkgs\\mkl-2024.2.2-h66d3029_15\\Library\\bin\\mkl_intel_thread.2.dll", "."),
    ("C:\\Users\\*****\\anaconda3\\pkgs\\mkl-2024.2.2-h66d3029_15\\Library\\bin\\mkl_core.2.dll", "."),
    ("C:\\Users\\*****\\anaconda3\\pkgs\\mkl-2024.2.2-h66d3029_15\\Library\\bin\\mkl_def.2.dll", ".")
]

# Collect all TKinterModernThemes data files
tkmt_datas = collect_data_files('TKinterModernThemes')

# Collect metadata for torch and torchvision
datas = tkmt_datas
datas += copy_metadata('torch') + copy_metadata('torchvision')

# Define hidden imports
hidden_imports = (
    collect_submodules('imagecodecs') +
    ["imagecodecs._shared"] + [x.__name__ for x in napari.plugins.plugin_manager.plugins.values()] +
    collect_submodules('TKinterModernThemes') +
    collect_submodules('torch') +
    collect_submodules('torchvision') +
    collect_submodules('imagecodecs') +
    [
        'tensorboard',                        # Manually add missing hidden imports
        'tzdata',
        'scipy.special._cdflib',
        # Add any other hidden imports that are missing
    ]
)

# Optionally exclude unnecessary torch.distributed modules if not used
excludes = [
    'torch.distributed._shard.checkpoint',
    'torch.distributed._sharded_tensor',
    # Add other torch.distributed modules to exclude
]

a = Analysis(
    ['app.py'],  # Replace with your main script
    pathex=[os.path.abspath('.')],
    binaries=additional_dlls,  # Include additional DLLs
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],  # Add custom hook paths if you have any
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='NVIDIA_HDR_Converter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging purposes
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app.ico'  # Ensure 'app.ico' exists in the specified path
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NVIDIA HDR Converter',
)

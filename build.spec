# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

# Collect TKinterModernThemes data files
tkmt_datas = collect_data_files('TKinterModernThemes')

# Collect imagecodecs data files and dynamic libraries
imagecodecs_datas = collect_data_files('imagecodecs')
imagecodecs_binaries = collect_dynamic_libs('imagecodecs')

# Additional data files
datas = [
    ('splash.jpg', '.'),
    ('app.ico', '.'),
] + tkmt_datas + imagecodecs_datas

# Hidden imports for TKinterModernThemes and imagecodecs
hiddenimports = [
    'TKinterModernThemes',
    'TKinterModernThemes.WidgetStyling',
    'TKinterModernThemes.ColorPalette',
    'imagecodecs',
    'imagecodecs._imagecodecs',
    'imagecodecs.imagecodecs',
] + collect_submodules('TKinterModernThemes') + collect_submodules('imagecodecs')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=imagecodecs_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='HDRConverter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app.ico',
)

# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Collect all data files, binaries, and hidden imports for mediapipe
mp_datas, mp_binaries, mp_hidden = collect_all('mediapipe')

datas = mp_datas
binaries = mp_binaries
hiddenimports = [
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'cv2',
    'numpy',
    'mediapipe',
    'mediapipe.python',
    'mediapipe.python.solutions',
    'mediapipe.python.solutions.face_mesh',
    'PIL',
    'scipy',
    'scipy.spatial',
    'deepfeik',
    'deepfeik.core',
    'deepfeik.core.engine',
    'deepfeik.core.blender',
    'deepfeik.core.color',
    'deepfeik.core.face_mesh',
    'deepfeik.core.landmarks',
    'deepfeik.core.warper',
    'deepfeik.pipeline',
    'deepfeik.pipeline.video_source',
    'deepfeik.pipeline.frame_buffer',
    'deepfeik.gui',
    'deepfeik.gui.main_window',
    'deepfeik.gui.camera_thread',
    'deepfeik.gui.clipboard',
] + mp_hidden

a = Analysis(
    ['src/deepfeik/__main__.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='deepfeik',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

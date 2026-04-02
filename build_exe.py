# -*- coding: utf-8 -*-
"""PyInstaller build script for 치아 정합 파이프라인."""
import PyInstaller.__main__
import sys
import os

# kornia LoFTR pretrained weights path
import kornia
kornia_data = os.path.join(os.path.dirname(kornia.__file__))

# sam2 package path
import sam2
sam2_path = os.path.dirname(sam2.__file__)

args = [
    'main_gui.py',
    '--onedir',
    '--windowed',
    '--name', 'DentalReg',

    # local modules
    '--hidden-import', 'register',
    '--hidden-import', 'matching',
    '--hidden-import', 'preprocess',
    '--hidden-import', 'transform',
    '--hidden-import', 'refine',
    '--hidden-import', 'sam2_mask',
    '--hidden-import', 'legacy_of',

    # torch / kornia
    '--hidden-import', 'torch',
    '--hidden-import', 'torchvision',
    '--hidden-import', 'kornia',
    '--hidden-import', 'kornia.feature',
    '--hidden-import', 'kornia.feature.loftr',

    # sam2
    '--hidden-import', 'sam2',
    '--hidden-import', 'sam2.sam2_image_predictor',
    '--hidden-import', 'sam2.modeling',
    '--hidden-import', 'huggingface_hub',

    # SimpleITK / skimage (optional but include)
    '--hidden-import', 'SimpleITK',
    '--hidden-import', 'skimage',
    '--hidden-import', 'skimage.registration',
    '--hidden-import', 'skimage.exposure',
    '--hidden-import', 'skimage.color',
    '--hidden-import', 'skimage.transform',

    # matplotlib backend
    '--hidden-import', 'matplotlib',
    '--hidden-import', 'matplotlib.backends.backend_agg',

    # PIL
    '--hidden-import', 'PIL',
    '--hidden-import', 'PIL.Image',
    '--hidden-import', 'PIL.ImageTk',

    # collect all kornia and sam2 data files
    '--collect-all', 'kornia',
    '--collect-all', 'sam2',

    # torch needs its DLLs
    '--collect-all', 'torch',

    # exclude unnecessary heavy modules
    '--exclude-module', 'tensorboard',
    '--exclude-module', 'IPython',
    '--exclude-module', 'jupyter',
    '--exclude-module', 'notebook',

    # add local .py as data (not strictly needed with hidden-import but safe)
    '--add-data', f'register.py{os.pathsep}.',
    '--add-data', f'matching.py{os.pathsep}.',
    '--add-data', f'preprocess.py{os.pathsep}.',
    '--add-data', f'transform.py{os.pathsep}.',
    '--add-data', f'refine.py{os.pathsep}.',
    '--add-data', f'sam2_mask.py{os.pathsep}.',
    '--add-data', f'legacy_of.py{os.pathsep}.',

    '--noconfirm',
]

print("=== Building DentalReg.exe ===")
print(f"Python: {sys.executable}")
PyInstaller.__main__.run(args)
print("=== Build complete ===")
print("Output: dist/DentalReg/")

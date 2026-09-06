"""Build the current web workspace into a native CPU-capable application."""
import os
import subprocess
import sys
import importlib.util
from pathlib import Path

args = [sys.executable, '-m', 'PyInstaller', '--onedir', '--windowed', '--name',
        'DKPregistrator', '--noconfirm', '--hidden-import', 'main_gui',
        '--hidden-import', 'sam2.sam2_image_predictor', '--hidden-import', 'sam2.modeling',
        '--hidden-import', 'huggingface_hub', '--collect-all', 'kornia',
        '--collect-all', 'sam2', '--collect-all', 'uvicorn',
        '--add-data', f'webapp/frontend/dist{os.pathsep}webapp/frontend/dist',
        '--add-data', f'webapp/folder_dialog.ps1{os.pathsep}webapp',
        '--exclude-module', 'tensorboard', '--exclude-module', 'IPython',
        '--exclude-module', 'jupyter', '--exclude-module', 'notebook']
if sys.platform == 'win32':
    args += ['--icon', 'app_icon.ico']
if sys.platform == 'darwin':
    args += ['--collect-all', 'torch', '--osx-bundle-identifier', 'com.perioahn.registrator']
# New torchvision wheels load _C_stable through torch.ops, outside Python's
# import graph. Older PyInstaller hooks only collect the former _C module.
vision_root = Path(importlib.util.find_spec('torchvision').origin).parent
vision_libraries = [p for p in vision_root.rglob('*') if p.suffix in {'.so', '.pyd', '.dll', '.dylib'}]
if not vision_libraries:
    raise RuntimeError('torchvision native libraries are missing before packaging')
for library in vision_libraries:
    destination = Path('torchvision') / library.parent.relative_to(vision_root)
    args += ['--add-binary', f'{library}{os.pathsep}{destination}']
subprocess.run(args + ['launcher.py'], check=True)

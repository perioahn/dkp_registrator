"""Build the current web workspace into a native CPU-capable application."""
import os
import subprocess
import sys

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
subprocess.run(args + ['launcher.py'], check=True)

# AgNW_Analyser.spec
# PyInstaller spec file that explicitly bundles tkinter for conda environments.
#
# Build with:
#   & "C:\Users\sambi\.julia\conda\3\x86_64\python.exe" -m PyInstaller AgNW_Analyser.spec

import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

# ── Conda environment root ────────────────────────────────────────────────────
CONDA = r"C:\Users\sambi\.julia\conda\3\x86_64"

# ── Collect all submodules for the scientific packages ────────────────────────
all_datas    = []
all_binaries = []
all_hidden   = []

for pkg in ['tifffile', 'PIL', 'numpy', 'scipy', 'skimage', 'matplotlib', 'cv2']:
    try:
        d, b, h = collect_all(pkg)
        all_datas    += d
        all_binaries += b
        all_hidden   += h
    except Exception:
        pass

# ---- Bundle the analysis scripts alongside the exe ---------------------------
# These are loaded at runtime via importlib so PyInstaller won't detect them
# automatically. The '.' destination means they land next to the exe.
import os as _os
for _script in ['nanowire_analysis.py', 'run_batch_analysis.py',
                 'AgNW_Analysis_Guide.html']:
    if _os.path.isfile(_script):
        all_datas.append((_script, '.'))

# ── Tkinter DLLs (exact paths found on this machine) ─────────────────────────
# _tkinter.pyd  — the Python extension module for tkinter
all_binaries.append((
    r"C:\Users\sambi\.julia\conda\3\x86_64\DLLs\_tkinter.pyd", "."
))

# tcl86t.dll and tk86t.dll — the tcl/tk runtime libraries
all_binaries.append((
    r"C:\Users\sambi\.julia\conda\3\x86_64\Library\bin\tcl86t.dll", "."
))
all_binaries.append((
    r"C:\Users\sambi\.julia\conda\3\x86_64\Library\bin\tk86t.dll", "."
))

# tcl8.6 and tk8.6 data directories — scripts, encoding tables, etc.
# These must land in the root of the exe's working directory.
_tcl_lib = r"C:\Users\sambi\.julia\conda\3\x86_64\Library\lib"
for _pkg in ["tcl8.6", "tk8.6"]:
    _src = os.path.join(_tcl_lib, _pkg)
    if os.path.isdir(_src):
        all_datas.append((_src, _pkg))

# Also search the standard conda tcl location as fallback
_tcl_alt = r"C:\Users\sambi\.julia\conda\3\x86_64\tcl"
for _pkg in ["tcl8.6", "tk8.6"]:
    _src = os.path.join(_tcl_alt, _pkg)
    if os.path.isdir(_src) and (_src, _pkg) not in all_datas:
        all_datas.append((_src, _pkg))

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ['AgNW_Analyser_GUI.py'],
    pathex=['.'],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hidden + [
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        'tkinter.font',
        '_tkinter',
        'multiprocessing',
        'multiprocessing.freeze_support',
        'multiprocessing.spawn',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hook_tcl.py'],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AgNW_Analyser',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX can break DLL loading on some Windows setups
    console=False,      # no terminal window (GUI only)
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,          # add an .ico file path here if you have one
)

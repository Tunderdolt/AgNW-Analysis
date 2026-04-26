# hook_tcl.py
# PyInstaller runtime hook — runs inside the exe before any imports.
# Sets TCL_LIBRARY and TK_LIBRARY so tkinter can find its data files
# regardless of where the exe is run from.
import os
import sys

# When frozen, sys._MEIPASS is the temp folder where PyInstaller
# extracts everything. The tcl8.6 and tk8.6 folders land there.
if getattr(sys, 'frozen', False):
    base = sys._MEIPASS
    tcl_path = os.path.join(base, 'tcl8.6')
    tk_path  = os.path.join(base, 'tk8.6')
    if os.path.isdir(tcl_path):
        os.environ['TCL_LIBRARY'] = tcl_path
    if os.path.isdir(tk_path):
        os.environ['TK_LIBRARY'] = tk_path

# -*- coding: utf-8 -*-
"""
rt_hook_nhc.py — PyInstaller runtime hook (runs before the application).

Two jobs:
  1. Windowed (--noconsole) safety: with no console, sys.stdout/stderr are None,
     so a stray print()/flush() or a logging StreamHandler emit can raise. We
     give them a real sink next to the .exe so the app survives and stays
     debuggable (nhc_console.log).
  2. Make the bundled CUDA runtime (torch/lib) discoverable. PyInstaller +
     torch usually handle this, but adding the directory explicitly is a cheap
     belt-and-suspenders against DLL resolution issues on the target machine.
"""
import os
import sys

_base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))

# --- 1. valid stdout/stderr in windowed mode -------------------------------
try:
    _logdir = os.path.dirname(sys.executable)
    _sink = open(os.path.join(_logdir, "nhc_console.log"), "w",
                 buffering=1, encoding="utf-8", errors="replace")
    if sys.stdout is None:
        sys.stdout = _sink
    if sys.stderr is None:
        sys.stderr = _sink
    print(f"[rt_hook] start. _MEIPASS={_base}")
except OSError:
    pass

# --- 2. bundled CUDA libs discoverable -------------------------------------
for _sub in (os.path.join(_base, "torch", "lib"), _base):
    if os.path.isdir(_sub):
        try:
            os.add_dll_directory(_sub)
        except (OSError, AttributeError):
            pass

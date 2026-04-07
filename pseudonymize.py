"""Thin wrapper — imports everything from the shared ~/RWM/pseudonymize.py.

All client alias logic lives in one place so every project uses the same map.
"""
import importlib.util, os, sys

_shared_path = os.path.expanduser('~/RWM/pseudonymize.py')
_spec = importlib.util.spec_from_file_location('_pseudonymize_shared', _shared_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export everything from the shared module into this module's namespace
for _name in _mod.__all__:
    globals()[_name] = getattr(_mod, _name)

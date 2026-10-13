"""
PulseGuard-Z Backend Package
"""

import sys
import typing

# Compatibility shim for Python 3.14 pre-release typing._eval_type changes
if sys.version_info >= (3, 14):
    _orig_eval_type = getattr(typing, "_eval_type", None)
    if _orig_eval_type is not None:
        def _compat_eval_type(*args, **kwargs):
            kwargs.pop("prefer_fwd_module", None)
            return _orig_eval_type(*args, **kwargs)
        typing._eval_type = _compat_eval_type

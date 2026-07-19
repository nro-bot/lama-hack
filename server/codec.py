"""
msgpack codec — copied VERBATIM from newt/_client/robot.py:26-53.

Do not re-derive this. The ndarray envelope keys are `bytes`, not `str`:
_pack_array emits Python bytes keys, msgpack encodes them as `bin`, and they
decode back as bytes even under the default raw=False. A reimplementation using
str keys packs and unpacks fine against itself and then silently fails to
round-trip against the real client.
"""
from __future__ import annotations

import functools
from typing import Any

import msgpack
import numpy as np


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


pack = functools.partial(msgpack.packb, default=_pack_array)
unpack = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ---------------------------------------------------------------------------
# Key access helpers.
#
# Frame keys decode as `str` (packed as msgpack str) but the ndarray envelopes
# nested inside them use `bytes`. Rather than guess per call site, every lookup
# goes through get_key, which tries both. Mirrors robot.py:1296.
# ---------------------------------------------------------------------------


def get_key(d: dict, key: str, default: Any = None) -> Any:
    """Fetch `key` from a decoded msgpack dict, tolerating str or bytes keys."""
    if key in d:
        return d[key]
    encoded = key.encode()
    if encoded in d:
        return d[encoded]
    return default


def get_str(d: dict, key: str, default: str = "") -> str:
    """Fetch a string field, decoding bytes. Mirrors robot.py's _str_field."""
    val = get_key(d, key)
    if isinstance(val, bytes):
        return val.decode()
    return val or default

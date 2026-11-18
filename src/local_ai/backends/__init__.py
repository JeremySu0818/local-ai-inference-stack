"""Inference backends.

Each backend is a self-contained package that declares one :class:`Backend`;
this module gathers those declarations into ``BACKENDS``.  The router looks a
name up here and never knows how an engine is built, started or fed, so an
engine can be added or replaced without touching the router.
"""
from __future__ import annotations

from .base import Backend, DEFAULT_BASE_URL, ENDPOINTS, configure_http

from .llama_cpp import BACKEND as LLAMA_CPP
from .qwen_image import BACKEND as QWEN_IMAGE

HTTP = Backend(name="http", base_url="", configure=configure_http)

BACKENDS = {b.name: b for b in (HTTP, LLAMA_CPP, QWEN_IMAGE)}


def get_backend(name: str) -> Backend:
    """The declaration for a backend name, or a message naming the names that exist."""
    backend = BACKENDS.get(name)
    if backend is None:
        raise ValueError(f"unsupported backend: {name}")
    return backend


__all__ = ("BACKENDS", "Backend", "DEFAULT_BASE_URL", "ENDPOINTS", "get_backend")

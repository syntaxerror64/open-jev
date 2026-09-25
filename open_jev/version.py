"""Package version of open-jev (stage 5).

A dependency-free single source of truth for the release number; the on-disk
checkpoint format carries its own separate ``format_version`` (see
``open_jev.checkpoint``).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

#!/usr/bin/env python3
"""Small IO helpers for standalone scripts."""
import gzip

try:
    from xopen import xopen
except ImportError:

    def xopen(path, mode="r", *args, **kwargs):
        path = str(path)
        if path.endswith(".gz"):
            gzip_mode = mode
            if "b" not in gzip_mode and "t" not in gzip_mode:
                gzip_mode = f"{gzip_mode}t"
            return gzip.open(path, gzip_mode, *args, **kwargs)
        return open(path, mode, *args, **kwargs)

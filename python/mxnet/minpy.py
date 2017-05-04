#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MinPy Python interface."""
from __future__ import absolute_import
from __future__ import print_function
import contextlib
from . import base

_enabled = False


def enable_jit():
    """Enable JIT."""
    global _enabled
    if _enabled:
        return
    base.check_call(base._LIB.MXEnableJIT())
    _enabled = True


def disable_jit():
    """Disable JIT."""
    global _enabled
    if not _enabled:
        return
    base.check_call(base._LIB.MXDisableJIT())
    _enabled = False


def _set_jit_context(ctx):
    if not _enabled:
        return
    base.check_call(
        base._LIB.MXSetJITContext(
            ctypes.c_int(ctx.device_typeid), ctypes.c_int(ctx.device_id)))


class JITContext():
    def mark_as_output(self, arr):
        if not _enabled:
            return
        base.check_call(base._LIB.MXJITMarkAsOutput(arr.handle))


@contextlib.contextmanager
def jit(ctx=None):
    enable_jit()
    if ctx is not None:
        _set_jit_context(ctx)
    yield JITContext()
    disable_jit()

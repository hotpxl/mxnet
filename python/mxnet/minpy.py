#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MinPy Python interface."""
from __future__ import absolute_import
from __future__ import print_function
import contextlib
from . import base


def enable_jit():
    """Enable JIT."""
    base.check_call(base._LIB.MXEnableJIT())


def disable_jit():
    """Disable JIT."""
    base.check_call(base._LIB.MXDisableJIT())


def set_jit_context(ctx):
    base.check_call(
        base._LIB.MXSetJITContext(
            ctypes.c_int(ctx.device_typeid), ctypes.c_int(ctx.device_id)))


class JITContext():
    def mark_as_output(self, arr):
        base.check_call(base._LIB.MXJITMarkAsOutput(arr.handle))


@contextlib.contextmanager
def jit(ctx=None):
    enable_jit()
    if ctx is not None:
        set_jit_context(ctx)
    yield JITContext()
    disable_jit()

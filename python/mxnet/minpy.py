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


class JITContext():
    def mark_as_output(arr):
        base.check_call(base._LIB.MXJITMarkAsOutput(arr.handle))


@contextlib.contextmanager
def jit():
    enable_jit()
    yield JITContext()
    disable_jit()

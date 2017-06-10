"""Test on extra marked outputs."""
import mxnet as mx
import time

for i in range(10):
    mx.minpy.enable_jit()
    a = mx.nd.ones(2) * i
    b = a * 4
    mx.minpy.JITContext().mark_as_output(b)
    c = b + b
    mx.minpy.JITContext().mark_as_output(c)
    mx.minpy.disable_jit()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(b.asnumpy())
    print(c.asnumpy())

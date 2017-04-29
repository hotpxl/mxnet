"""Test on data-independent control flow."""
import mxnet as mx
import time

for i in range(10):
    a = mx.nd.ones(2) * i
    mx.minpy.enable_jit()
    a = a + 1
    if a.asnumpy()[0] % 2 == 0:
        b = a + a
    else:
        b = a * a
    b = b + 1
    mx.minpy.disable_jit()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(b.asnumpy())

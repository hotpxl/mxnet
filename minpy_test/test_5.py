"""Test on data-independent control flow."""
import mxnet as mx
import time

switch = False
for i in range(10):
    a = mx.nd.ones(2) * i
    mx.minpy.enable_jit()
    if switch:
        b = a + a
    else:
        b = a * a
    b = b + 1
    mx.minpy.disable_jit()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(b.asnumpy())
    switch = not switch

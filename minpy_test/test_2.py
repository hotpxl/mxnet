import mxnet as mx
import time
mx.minpy.enable_jit()

a = mx.nd.ones(2)
b = mx.nd.ones(2)
c = a + b
mx.minpy.disable_jit()
print(c.asnumpy())
time.sleep(1)

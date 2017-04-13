import mxnet as mx
import time
mx.minpy.enable_jit()

a = mx.nd.ones(2)
b = mx.nd.ones(2)
print((a + b).asnumpy())
mx.minpy.disable_jit()
time.sleep(1)

import mxnet as mx
mx.minpy.enable_jit()

a = mx.nd.ones(2)
b = mx.nd.ones(2)
(a + b).asnumpy()
mx.minpy.disable_jit()

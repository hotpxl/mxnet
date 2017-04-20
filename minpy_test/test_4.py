"""Test JIT Graph Equal Feature"""
import mxnet as mx
import time

mx.minpy.enable_jit()

print("Build G1")
a = mx.nd.ones(2)
b = mx.nd.ones(2)
c = a + b
print(c.asnumpy())
time.sleep(1)

print("Build G2, which is identical to G1")
a = mx.nd.ones(2)
b = mx.nd.ones(2)
c = a + b
print(c.asnumpy())
time.sleep(1)

print("Build G3, which uses diff op than G1 & G2")
a = mx.nd.ones(2)
b = mx.nd.ones(2)
c = a - b
print(c.asnumpy())
time.sleep(1)

print("Build G4, which uses diff compute order than G3")
a = mx.nd.ones(2)
b = mx.nd.ones(2)
c = b - a
print(c.asnumpy())
time.sleep(1)

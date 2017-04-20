"""Test JIT Graph Equal Feature"""
import mxnet as mx
import time

mx.minpy.enable_jit()

print("Run G1")
a = mx.nd.ones(2)
b = mx.nd.ones(2)
c = a + b
print(c.asnumpy())
print

print("Run G2, which is identical to G1")
a = mx.nd.ones(2)
b = mx.nd.ones(2)
c = a + b
print(c.asnumpy())
print

print("Run G3, which uses diff op than G1 & G2")
a = mx.nd.ones(2)
b = mx.nd.ones(2)
c = a - b
print(c.asnumpy())
print

print("Run G4, which uses diff compute order than G3")
a = mx.nd.ones(2)
b = mx.nd.ones(2)
c = b - a
print(c.asnumpy())
print

print("Run G5, which size is different from G1")
a = mx.nd.ones(3)
b = mx.nd.ones(3)
c = a + b
print(c.asnumpy())
print

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import functools
import mxnet as mx
import mxnet.ndarray as nd

context = mx.cpu()


def linear(X, W, bias):
    return nd.dot(X, W) + bias


def sigmoid(x):
    return .5 * (nd.tanh(.5 * x) + 1)


def gaussian(shape):
    return nd.random_normal(shape=shape).as_in_context(context)


D = 1024

WX_SHAPE = (7, D)
Wxi = gaussian(shape=WX_SHAPE)
Wxf = gaussian(shape=WX_SHAPE)
Wxo = gaussian(shape=WX_SHAPE)
Wxg = gaussian(shape=WX_SHAPE)

bX_SHAPE = (D, )
bxi = nd.zeros(shape=bX_SHAPE)
bxf = nd.zeros(shape=bX_SHAPE)
bxo = nd.zeros(shape=bX_SHAPE)
bxg = nd.zeros(shape=bX_SHAPE)

WH_SHAPE = (D, D)
Whi = gaussian(shape=WH_SHAPE)
Whf = gaussian(shape=WH_SHAPE)
Who = gaussian(shape=WH_SHAPE)
Whg = gaussian(shape=WH_SHAPE)

bH_SHAPE = (D, )
bhi = nd.zeros(shape=bH_SHAPE)
bhf = nd.zeros(shape=bH_SHAPE)
bho = nd.zeros(shape=bH_SHAPE)
bhg = nd.zeros(shape=bH_SHAPE)

W = gaussian(shape=(D, 10))
b = nd.zeros(shape=(10, ))

N = 128
X = gaussian(shape=(N, 784 // 7, 7))


def pure(func):
    '''
    Assumptions:
    1. Closed (pure) function.
    2. Arguments are of type NDArray.
    3. Return value is either NDArray of an iterable of NDArrays.
    4. No side effect.
    '''
    closure_payload = {
        'segment_id': None,
        'output_shapes': None,
    }

    # fmap :: (a -> b) -> Either a [a] -> Either b [b]
    def fmap(f, value):
        if isinstance(value, mx.nd.NDArray):
            return f(value)
        else:
            return list(map(f, value))

    @functools.wraps(func)
    def wrapper(*args):
        if closure_payload['segment_id'] is None:
            mx.minpy.start_trace()
            ret = func(*args)
            fmap(lambda x: mx.minpy.mark_as_output(x), ret)
            segment_id = mx.minpy.end_trace()
            closure_payload['segment_id'] = segment_id
            closure_payload['output_shapes'] = fmap(lambda x: x.shape, ret)
        ret_value = fmap(lambda x: mx.nd.zeros(shape=x.shape), closure_payload['output_shapes'])
        run_segment(segment_id, args, closure_payload, ret_value)
        return ret_value

    return wrapper


@pure
def foo(h, c, patch, Wxi, Wxf, Wxo, Wxg, bxi, bxf, bxo, bxg, Whi, Whf, Who,
        Whg, bhi, bhf, bho, bhg):
    i = sigmoid(linear(patch, Wxi, bxi) + linear(h, Whi, bhi))
    f = sigmoid(linear(patch, Wxf, bxf) + linear(h, Whf, bhf))
    o = sigmoid(linear(patch, Wxo, bxo) + linear(h, Who, bho))
    g = nd.tanh(linear(patch, Wxg, bxg) + linear(h, Whg, bhg))
    c = f * c + i * g
    h = o * mx.nd.tanh(c)
    return h, c, linear(h, W, b)


for index in range(10):
    h = nd.zeros((N, D))
    c = nd.zeros((N, D))

    for i in range(784 // 7):
        patch = mx.nd.slice_axis(X, axis=1, begin=i, end=(i + 1))
        t0 = time.time()
        # Segment start.
        # i = sigmoid(linear(patch, Wxi, bxi) + linear(h, Whi, bhi))
        # f = sigmoid(linear(patch, Wxf, bxf) + linear(h, Whf, bhf))
        # o = sigmoid(linear(patch, Wxo, bxo) + linear(h, Who, bho))
        # g = nd.tanh(linear(patch, Wxg, bxg) + linear(h, Whg, bhg))
        # c = f * c + i * g
        # h = o * mx.nd.tanh(c)
        # linear(h, W, b).asnumpy()
        # Segment end.
        h_new, c_new, res = foo(h, c, patch, Wxi, Wxf, Wxo, Wxg, bxi, bxf, bxo,
                                bxg, Whi, Whf, Who, Whg, bhi, bhf, bho, bhg)
        # res.asnumpy()
        h = h_new
        c = c_new
        print(time.time() - t0)

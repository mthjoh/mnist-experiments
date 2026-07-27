import numpy as np
from conv_layers import Convolution, im2col, im2col_batch
import time

x = np.arange(75).reshape(3,5,5).astype(np.float64)
w = np.ones((3,3,3))/27.0
s = 1

# x_batch = np.stack([x, x])
# W_layer = w.reshape(1, 3, 3, 3)
# b = np.zeros(1)
# layer = Convolution(W_layer, b)

fake_batch = np.random.randn(100, 1, 28, 28)
W = np.random.randn(16, 1, 3, 3) * 0.01
b = np.zeros(16)
layer = Convolution(W, b)

# start = time.perf_counter()
# fast = layer.forward(fake_batch)
# elapsed = time.perf_counter() - start
# print(f'Fast Time: {elapsed}s')

# start2 = time.perf_counter()
# slow = layer.forward_naive(fake_batch)
# elapsed2 = time.perf_counter() - start2
# print(f'Slow Time: {elapsed2}s')
# print(f'Shape of each and are they roughly the same value?:{fast.shape, slow.shape},{np.allclose(fast, slow)}')

out = layer.forward(fake_batch)
layer.backward(np.ones_like(out))
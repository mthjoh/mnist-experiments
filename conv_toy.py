import numpy as np
from conv_layers import Convolution, im2col, im2col_batch, MaxPooling, Flatten
import time

pool = MaxPooling(2, 2)
fake = np.random.randn(100, 16, 26, 26)
out = pool.forward(fake)
# print(pool.backward(np.ones_like(out)).shape)

flat = Flatten()
dout = flat.forward(out)
print(flat.backward(dout).shape)
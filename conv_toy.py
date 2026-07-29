import numpy as np
from conv_layers import Convolution, im2col, im2col_batch, MaxPooling, Flatten
import time
from conv_network import SimpleConvNet

net = SimpleConvNet()
x = np.random.randn(10, 1, 28, 28)
t = np.zeros((10, 10))
t[np.arange(10), np.random.randint(0, 10, 10)] = 1

print(net.loss(x, t))
print(net.accuracy(x, t))

g = net.gradient(x, t)
for j in range(3):
    print(net.W[j].shape, g['W'][j].shape, net.b[j].shape, g['b'][j].shape)
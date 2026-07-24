import numpy as np
from conv_layers import Convolution

x = np.arange(75).reshape(3,5,5).astype(np.float64)
w = np.ones((3,3,3))/27.0
s = 1

out_size = (x.shape[1] - w.shape[1])//s + 1
out = np.zeros((out_size, out_size))

for i in range(out_size):
    for j in range(out_size):
        out[i,j] = np.sum(x[:,i*s:i*s+3,j*s:j*s+3] * w)



print(out[0,0])
print(out)




x_batch = x.reshape(1, 3, 5, 5)
W_layer = w.reshape(1, 3, 3, 3)
b = np.zeros(1)

print(Convolution(W_layer,b).forward(x_batch))
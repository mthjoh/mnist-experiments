import numpy as np
from conv_layers import Convolution
import time

x = np.arange(75).reshape(3,5,5).astype(np.float64)
w = np.ones((3,3,3))/27.0
s = 1

out_size = (x.shape[1] - w.shape[1])//s + 1
out = np.zeros((out_size, out_size))

for i in range(out_size):
    for j in range(out_size):
        out[i,j] = np.sum(x[:,i*s:i*s+3,j*s:j*s+3] * w)



def im2col(x, F, stride):
    channels, H, W_in = x.shape
    out_size = (H-F)//stride + 1
    col = np.zeros((out_size*out_size, channels*F*F))
    
    for i in range(out_size):
        for j in range(out_size):
            col[i*out_size+j] = x[:,i*stride:i*stride+F,j*stride:j*stride+F].flatten()
    return col

col = im2col(x, 3, 1)

result = np.matmul(col,w.flatten())
print(result)
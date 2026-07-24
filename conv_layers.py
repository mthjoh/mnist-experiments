import numpy as np

class Convolution:
    def __init__(self, W, b, stride=1):
        self.W=W
        self.b=b
        self.stride=stride

    def forward(self, x):
        batch, channels, H, W_in = x.shape
        filters = self.W.shape[0]
        F = self.W.shape[2]
        out_size = (H-F)//self.stride + 1
        out = np.zeros((batch, filters, out_size, out_size))
        
        for image in range(batch):
            for filter in range(filters):
                for i in range(out_size):
                    for j in range(out_size):
                        out[image,filter,i,j] = np.sum(x[image,:,i*self.stride:i*self.stride+F,j*self.stride:j*self.stride+F] * self.W[filter])+self.b[filter]
        return out
    



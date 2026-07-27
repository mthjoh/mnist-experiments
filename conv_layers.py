import numpy as np

class Convolution:
    def __init__(self, W, b, stride=1):
        self.W=W
        self.b=b
        self.stride=stride

    def forward_naive(self, x):
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
    
    def forward(self, x):
        batch, channels, H, W_in = x.shape
        filters = self.W.shape[0]
        F = self.W.shape[2]
        out_size = (H-F)//self.stride + 1
        col = im2col_batch(x,F,self.stride)
        col_W = self.W.reshape(filters,-1).T
        out = col @ col_W + self.b
        out = out.reshape(batch, out_size, out_size, filters)
        out = out.transpose(0,3,1,2)
        return out


    
def im2col(x, F, stride):
    channels, H, W_in = x.shape
    out_size = (H-F)//stride + 1
    col = np.zeros((out_size*out_size, channels*F*F))
    
    for i in range(out_size):
        for j in range(out_size):
            col[i*out_size+j] = x[:,i*stride:i*stride+F,j*stride:j*stride+F].flatten()
    return col

def im2col_batch(x, F, stride):
    images, channels, H, W_in = x.shape
    out_size = (H-F)//stride + 1
    patches = out_size*out_size
    col = np.zeros((patches*images, channels*F*F))

    for image in range(images):
        col[image*patches:image*patches+patches]=im2col(x[image], F, stride)
    return col

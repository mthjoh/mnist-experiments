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

        self.col = col
        self.col_W = col_W
        self.x_shape = x.shape
        return out


    # x        (100, 1, 28, 28)
    # col      (67600, 9)
    # col_W    (9, 16)
    # out      (100, 16, 26, 26)
    # dout 2D  (67600, 16)
    # dW       (16,1,3,3)
    # db       (16,)
    # dcol     (67600,9)
    # dx       (100, 1, 28, 28)

    def backward(self, dout):
        filters = self.W.shape[0]
        F = self.W.shape[2]
        channels = self.W.shape[1]
        dout = dout.transpose(0,2,3,1).reshape(-1, filters)

        dW = self.col.T @ dout
        self.dW = dW.T.reshape(filters,channels,F,F)

        self.db = np.sum(dout, axis = 0)

        dcol = dout@self.col_W.T

        dx=col2im(dcol, self.x_shape, F, self.stride)
        return dx

class MaxPooling:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.arg_max = None
        self.x_shape = None

    def forward(self,x):
        batch, channels, H, W_in = x.shape
        out_size = (H-self.pool_size)//self.stride + 1
        self.x_shape = x.shape
        x_folded = x.reshape(batch*channels, -1,H, W_in)
        col = im2col_batch(x_folded, self.pool_size, self.stride)
        out=np.max(col, axis=1)
        self.arg_max = np.argmax(col, axis=1)
        out = out.reshape(batch, channels, out_size, out_size)
        return out

    def backward(self, dout):
        dout_flat = dout.flatten()

        dcol = np.zeros((dout_flat.size, self.pool_size*self.pool_size))

        rows = np.arange(dout_flat.size)
        dcol[rows, self.arg_max] = dout_flat

        batch, channels, H, W_in = self.x_shape
        dx_folded = col2im(dcol,(batch*channels, 1, H, W_in),self.pool_size,self.stride)

        dx = dx_folded.reshape(batch,channels,H,W_in)
        return dx

class Flatten:
    def __init__(self):
        self.x_shape = None
    def forward(self, x):
        batch = x.shape[0]
        self.x_shape = x.shape
        return x.reshape(batch, -1)
    def backward(self, dout):
        return dout.reshape(self.x_shape)


        
    
def col2im(dcol, x_shape, F, stride):
    batch, channels, H, W_in = x_shape
    out_size = (H-F)//stride + 1
    patches = out_size*out_size
    pixels = np.zeros((batch, channels, H, W_in))

    for n in range(batch):
        for i in range(out_size):
            for j in range(out_size):
                row = n*patches + i*out_size + j
                patch_grad = dcol[row].reshape(channels, F, F)
                pixels[n, :,i*stride:i*stride+F,j*stride:j*stride+F] += patch_grad
    return pixels

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

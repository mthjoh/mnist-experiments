import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        out = dout.copy()
        out[self.mask] = 0
        return out


class Affine:
    def __init__(self, W, b):
        self.W=W
        self.b=b
        self.x=None
        self.dW=None
        self.db=None

    def forward(self, x):
        self.x=x
        return (np.dot(x,self.W)+self.b)

    def backward(self, dout):
        dx = np.dot(dout,np.transpose(self.W))
        self.dW = np.dot(np.transpose(self.x),dout)
        self.db = np.sum(dout, axis=0)
        return dx


if __name__ == '__main__':
    x = np.array([-2,0,3])
    dout = np.array([1.0,1.0,1.0])
    relu = Relu()
    forward_test=relu.forward(x)
    print(forward_test)
    backward_test=relu.backward(dout)
    print(backward_test)
    

def softmax(x):
    xmax = np.max(x)
    xtamed= x-xmax
    expx = np.exp(xtamed)
    return expx/np.sum(expx)

def cross_entropy_error(y,t):
    return -np.log(np.sum(t*y)+1e-7)

print(cross_entropy_error(np.array(([1.0, 0.0])),np.array(([1.0, 0.0]))))
print(cross_entropy_error(np.array(([1.0, 0.0])),np.array(([0.0, 1.0]))))

print(softmax(np.array([2.0, 1.0, 0.1])))
print(softmax(np.array([1002.0, 1001.0, 1000.1]))) 

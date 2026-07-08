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

if __name__ == '__main__':
    x = np.array([-2,0,3])
    dout = np.array([1.0,1.0,1.0])
    relu = Relu()
    forward_test=relu.forward(x)
    print(forward_test)
    backward_test=relu.backward(dout)
    print(backward_test)
    

 
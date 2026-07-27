import numpy as np
from conv_layers import Convolution

x = np.random.randn(2,1,6,6)
W = np.random.randn(2,1,3,3)*0.1
b = np.zeros(2)
layer = Convolution(W, b)

out = layer.forward(x)
dx_analytic = layer.backward(np.ones_like(out)).copy()
dW_analytic = layer.dW.copy()
db_analytic = layer.db.copy()

# print(dW_analytic.shape)
# print(db_analytic.shape)
print(dx_analytic.shape)


def loss(layer, x):
    return np.sum(layer.forward(x))

def check(layer, arr, idx, grad_entry):
    original_value = arr[idx]
    arr[idx] = original_value + 1e-4
    loss_up=loss(layer,x)
    arr[idx] = original_value - 1e-4
    loss_down=loss(layer,x)
    arr[idx]=original_value
    numerical_gradient = (loss_up-loss_down)/(2*1e-4)
    print(f'Numerical: {numerical_gradient}, Backprops: {grad_entry}')
    

# for idx in np.ndindex(W.shape):
#     check(layer, layer.W, idx, dW_analytic[idx])

# for idx in np.ndindex(b.shape):
#     check(layer, layer.b, idx, db_analytic[idx])

for idx in np.ndindex(x.shape):
    check(layer, x, idx, dx_analytic[idx])

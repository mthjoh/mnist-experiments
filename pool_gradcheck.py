import numpy as np
from conv_layers import MaxPooling

x = np.random.randn(1, 2, 4, 4)
pool = MaxPooling(2,2)

out = pool.forward(x)
dx_analytic = pool.backward(np.ones_like(out)).copy()

def loss(pool, x):
    return np.sum(pool.forward(x))

def check(pool, arr, idx, grad_entry):
    original_value = arr[idx]
    arr[idx] = original_value + 1e-4
    loss_up=loss(pool,x)
    arr[idx] = original_value - 1e-4
    loss_down=loss(pool,x)
    arr[idx]=original_value
    numerical_gradient = (loss_up-loss_down)/(2*1e-4)
    print(f'Numerical: {numerical_gradient}, Backprops: {grad_entry}')

for idx in np.ndindex(x.shape):
    check(pool, x, idx, dx_analytic[idx])
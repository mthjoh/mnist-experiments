from dataset.mnist import load_mnist
import numpy as np
import network

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
net = network.MultiLayerNet([784, 10, 10])
x_portion = x_train[:5]
t_portion = t_train[:5]

grads = net.gradient(x_portion,t_portion)

def check(net, arr, idx, grad_entry):
    original_value = arr[idx]
    arr[idx] = original_value + 1e-4
    loss_up=net.loss(x_portion,t_portion)
    arr[idx] = original_value - 1e-4
    loss_down=net.loss(x_portion,t_portion)
    arr[idx]=original_value
    numerical_gradient = (loss_up-loss_down)/(2*1e-4)
    print(f'Numerical: {numerical_gradient}, Backprops: {grad_entry}')


check(net, net.b[0], 3, grads['b'][0][3])
check(net, net.b[1], 7, grads['b'][1][7])
check(net, net.W[0], (406,0), grads['W'][0][406,0])
check(net, net.W[1], (5,5), grads['W'][1][5,5])
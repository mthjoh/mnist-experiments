import numpy as np
import layers
import network
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import optimizers

#optimzer = optimizers.SGD(lr=0.1)
#optimzer = optimizers.Momentum(lr=0.1, mom=0.9)
optimzer = optimizers.Adam(lr=0.001, beta1=0.9, beta2=0.999)

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

net = network.MultiLayerNet([784,1000,1000,10])

for i in range(10001):
    choice = np.random.choice(60000,100)
    x_batch = x_train[choice]
    t_batch = t_train[choice]
    grads = net.gradient(x_batch,t_batch)

    params={'W':net.W, 'b':net.b}
    optimzer.update(params,grads)

    if i%2000==0:
        print(f"Iteration {i}, Loss is: {net.loss(x_batch, t_batch)}")
        print(f"Iteration {i}, Accurcy is: {net.accuracy(x_test,t_test)}")

np.savez('model.npz', *net.W, *net.b, sizes=net.sizes)
import numpy as np
import layers
import network
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

net = network.MultiLayerNet()

for i in range(100000):
    choice = np.random.choice(60000,100)
    x_batch = x_train[choice]
    t_batch = t_train[choice]
    grads = net.gradient(x_batch,t_batch)
    net.W1 -= 0.1*grads['W1']
    net.W2 -= 0.1*grads['W2']
    net.b1 -= 0.1*grads['b1']
    net.b2 -= 0.1*grads['b2']
    print(net.loss(x_batch, t_batch))

print(f"Accurcy is: {net.accuracy(x_test,t_test)}")

# print(x_batch.shape, t_batch.shape)
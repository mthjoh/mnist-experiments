import numpy as np
import layers
import network
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

net = network.MultiLayerNet([784,100,10])

for i in range(2001):
    choice = np.random.choice(60000,100)
    x_batch = x_train[choice]
    t_batch = t_train[choice]
    grads = net.gradient(x_batch,t_batch)

    for j in range(len(net.W)):
        net.W[j] -= 0.1*grads['W'][j]
        net.b[j] -= 0.1*grads['b'][j]


    # net.W1 -= 0.1*grads['W1']
    # net.W2 -= 0.1*grads['W2']
    # net.b1 -= 0.1*grads['b1']
    # net.b2 -= 0.1*grads['b2']
    if i%100==0:
        print(f"Iteration {i}, Loss is: {net.loss(x_batch, t_batch)}")
        print(f"Iteration {i}, Accurcy is: {net.accuracy(x_test,t_test)}")

np.savez('model.npz', *net.W, *net.b, sizes=net.sizes)

# print(x_batch.shape, t_batch.shape)

# print(np.abs(grads['W'][0]).mean())
# print(np.abs(grads['W'][-1]).mean())

# img = x_test[0]
# img = img.reshape(28,28)
# plt.imshow(img, cmap='gray')
# plt.show()


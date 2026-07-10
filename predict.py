import numpy as np
import network
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

net = network.MultiLayerNet([784,100,10])
data = np.load('model.npz')

n1=len(net.W)
n2=len(net.b)
for i in range(n1):
    net.W[i][:]=data[f'arr_{i}']
for i in range(n2):
    net.b[i][:]=data[f'arr_{n1+i}']

print(net.accuracy(x_test, t_test))


idx=100
img = x_test[idx]
img_net = img.reshape(1,784)
img_plot = img.reshape(28,28)
print(f'The prediction of the network: {np.argmax(net.predict(img_net))}')
print(f'The actual answer: {np.argmax(t_test[idx])}')
plt.imshow(img_plot, cmap='gray')
plt.show()

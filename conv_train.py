import numpy as np
from dataset.mnist import load_mnist
from conv_network import SimpleConvNet
import optimizers
import time

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=True, one_hot_label=True)

net = SimpleConvNet()
optimzer = optimizers.Momentum(lr=0.1, mom=0.9)

x_eval = x_test[:1000]
t_eval = t_test[:1000]

start = time.perf_counter()

for i in range(3001):
    choice = np.random.choice(60000,100)
    x_batch = x_train[choice]
    t_batch = t_train[choice]

    if i%200==0:
        print(f"Iteration {i}, Loss is: {net.loss(x_batch, t_batch)}")
        print(f"Iteration {i}, Accurcy is: {net.accuracy(x_eval, t_eval)}")

    grads = net.gradient(x_batch,t_batch)

    params={'W':net.W, 'b':net.b}
    optimzer.update(params,grads)

elapsed = time.perf_counter() - start
print(f'Time: {elapsed}s')

np.savez('conv_model.npz', *net.W, *net.b)
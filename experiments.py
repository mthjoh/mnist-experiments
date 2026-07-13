import numpy as np
import layers
import network
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import optimizers
import csv

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

def run_experiments(sizes, optimizer, iters, batch_size):
    loss_hist = []
    acc_hist = []
    iter_hist = []
    optimizer = optimizer
    net = network.MultiLayerNet(sizes)
    for i in range(iters):
        choice = np.random.choice(x_train.shape[0],batch_size)
        x_batch = x_train[choice]
        t_batch = t_train[choice]
        grads = net.gradient(x_batch,t_batch)

        params={'W':net.W, 'b':net.b}
        optimizer.update(params,grads)

        if i%500==0:
            loss_hist.append(net.loss(x_batch, t_batch))
            acc_hist.append(net.accuracy(x_test,t_test))
            iter_hist.append(i)

    return loss_hist, acc_hist, iter_hist

results = {}
results['SGD'] = run_experiments([784, 100, 10], optimizers.SGD(lr=0.1), 10001, 100)
results['Momentum'] = run_experiments([784, 100, 10], optimizers.Momentum(0.1, 0.9), 10001, 100)
results['Adam'] = run_experiments([784, 100, 10], optimizers.Adam(0.001, 0.9, 0.999), 10001, 100)

for name, (loss, acc, it) in results.items():
    plt.plot(it, acc, label=name)

plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Optimizer Accuracy Comparison')
plt.legend()
plt.show()

plt.figure()

for name, (loss, acc, it) in results.items():
    plt.plot(it, loss, label=name)

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Optimizer Loss Comparison')
plt.legend()
plt.show()

# I am going to log the data into a CSV file
with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['optimizer', 'iteration', 'loss', 'accuracy'])
    for name, (loss, acc, it) in results.items():
        for j in range(len(loss)):
            writer.writerow([name, it[j],loss[j],acc[j]])
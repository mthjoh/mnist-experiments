import time
import numpy as np
import optimizers
import network
from dataset.mnist import load_mnist
import csv
import layers

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

dtype=np.float32

x_train = x_train.astype(dtype)
t_train = t_train.astype(dtype)

iters=20001
batch=500

sizes = [784, 1000, 1000, 1000, 10]
flops = 0

for i in range(len(sizes)-1):
    flops += 2*batch*sizes[i] *sizes[i+1]
total = flops*3

def run_experiments(sizes, optimizer, iters, batch_size):
    loss_hist = []
    acc_hist = []
    iter_hist = []
    net = network.MultiLayerNet(sizes)

    for j in range(len(sizes)-1):
        net.W[j] = net.W[j].astype(dtype)
        net.layers[2*j].W = net.W[j]
        net.b[j] = net.b[j].astype(dtype)
        net.layers[2*j].b = net.b[j]

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
            print(f'Accuracy:{acc_hist[-1]}, Loss:{loss_hist[-1]}')

    return loss_hist, acc_hist, iter_hist

start = time.perf_counter()
loss_hist, acc_hist, iter_hist = run_experiments(sizes, optimizers.Momentum(0.1, 0.9), iters, batch)
elapsed = time.perf_counter() - start
print(f'Time: {elapsed}s, iters/sec:{iters/elapsed}, images/sec: {batch*iters/elapsed} GFLOPS: {total*iters/elapsed/1e9}')

with open('results_exp_99.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['iteration', 'loss', 'accuracy'])
    for j in range(len(loss_hist)):
        writer.writerow([iter_hist[j],loss_hist[j],acc_hist[j]])

print(f'Accuracy:{acc_hist[-1]}, Loss:{loss_hist[-1]}')
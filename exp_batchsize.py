from experiments import run_experiments
import numpy as np
import matplotlib.pyplot as plt
import optimizers
import csv

results = {}
results['ten_batch'] = run_experiments([784, 100, 10], optimizers.Momentum(0.1, 0.9), 10001, 10)
results['hundred_batch'] = run_experiments([784, 100, 10], optimizers.Momentum(0.1, 0.9), 10001, 100)
results['thousand_batch'] = run_experiments([784, 100, 10], optimizers.Momentum(0.1, 0.9), 10001, 1000)

for name, (loss, acc, it) in results.items():
    plt.plot(it, acc, label=name)

plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Batch Size Accuracy Comparison')
plt.legend()
plt.show()

plt.figure()

for name, (loss, acc, it) in results.items():
    plt.plot(it, loss, label=name)

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Batch Size Loss Comparison')
plt.legend()
plt.show()

# I am going to log the data into a CSV file
with open('results_for_batchsize.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['batch_size', 'iteration', 'loss', 'accuracy'])
    for name, (loss, acc, it) in results.items():
        for j in range(len(loss)):
            writer.writerow([name, it[j],loss[j],acc[j]])
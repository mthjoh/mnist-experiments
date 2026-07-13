from experiments import run_experiments
import numpy as np
import matplotlib.pyplot as plt
import optimizers
import csv

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
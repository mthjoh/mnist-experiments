import numpy as np
from conv_network import SimpleConvNet

net = SimpleConvNet()
x = np.random.randn(2, 1, 28, 28)
t = np.zeros((2,10))
t[0,3] = 1
t[1,5] = 1
grads = net.gradient(x, t)

dW_analytic = [g.copy() for g in grads['W']]
db_analytic = [g.copy() for g in grads['b']]

def check(arr, idx, grad_entry):
    original_value = arr[idx]
    arr[idx] = original_value + 1e-4
    loss_up=net.loss(x,t)
    arr[idx] = original_value - 1e-4
    loss_down=net.loss(x,t)
    arr[idx]=original_value
    numerical_gradient = (loss_up-loss_down)/(2*1e-4)
    rel = abs(numerical_gradient - grad_entry) / max(abs(numerical_gradient), abs(grad_entry), 1e-12)
    return rel

rels = []
for idx in np.ndindex(net.W[0].shape):
    rels.append(check(net.W[0], idx, dW_analytic[0][idx]))

rels_b = []
for idx in np.ndindex(net.b[0].shape):
    rels_b.append(check(net.b[0], idx, db_analytic[0][idx]))

print(np.median(rels), np.max(rels))
print(np.median(rels_b), np.max(rels_b))

rels_W1 = []
for _ in range(100):
    idx = (np.random.randint(2704), np.random.randint(100))
    rels_W1.append(check(net.W[1], idx, dW_analytic[1][idx]))

print(np.median(rels_W1), np.max(rels_W1))
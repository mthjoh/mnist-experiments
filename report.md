# [Title — you write it; make it say what was built and found]

## 1. Introduction
   ← your draft's paragraphs 1–2, plus the headline: 99.11%
## 2. Network Implementation

    I created the layers.py file to simulate the different layers in the neural network: Affine, Relu, and SoftmaxWithLoss. The Affine class's forward method returns the dot product of the input and the weights with the biases added in. Its backward method stores the derivative of the weights and biases in relationship with the loss and returns the gradient with respect to its input.  The Relu class's forward method returns the array with all the negative values zeroed out. The backward pass returns the backprop with all the masked values set to 0. The SoftmaxWithLoss's forward method applies the softmax and cross entropy error functions to the received data and returns it. Its backward method returns the difference between the prediction and the true label divided by the batchsize. I created network.py to simulate forward passes and backpropagation. MultiLayerNet builds the layer chain from a sizes list such as [784, 100, 10]. The train.py file handles the training of the network.  

    While briefly comparing wide nets and deep nets, I discovered that deep layer nets failed after a certain amount of depth - the loss was stuck at 2.302 (equal to -log(0.1) the loss of uniform guessing among 10 choices). With naive init (weight = small random * 0.01), the signal shrank at every layer(~400:1 by the last layer). This resulted in the signals reaching the softmax with loss function being too small. The gradients came back as zeros. This taught me the importance of He initialization: scaling each layer's weights by the root of (2/n_inputs) specifically designed for ReLU networks. Immediately the same deep architecture was able to be trained; a 14 layer net reached its full accuracy.

    Backprop needed double checking to make sure the network was working correctly. Gradient checking was used to compare its results with the results of backprop. The formula was (loss up by ε − loss down by ε) / 2ε, ε being a small number (in this case 1e-4). ε could not be too small such as 1e-300 because python uses float64 setting a limit to precision. The result of the comparison was that the backprop was accurate up to ~7 decimal points, checked on both W and b across multiple layers. 

    The first working network — plain SGD, [784,100,10] — reached 97.99% after 100k iterations. 
   
## 3. Optimization Experiments
   ← optimizer trio; batch×lr coupling INCLUDING the new lr-0.5 NaN finding
## 4. Performance Engineering
   ← throughput, FLOPs, GFLOPS/utilization; the dtype saga (fp32 win, fp16 collapse)
## 5. The 99% Campaign
   ← exp_99 memorization diagnosis → augmentation v1 → v2 → 99.11%
## 6. Conclusions and Next Steps
   ← MLP ceiling case for CNNs; validation discipline; one sentence on transformers/papers
## Appendix: Repository Map
   ← file-by-file one-liners; how to reproduce
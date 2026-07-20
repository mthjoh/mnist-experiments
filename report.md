# [Title — you write it; make it say what was built and found]

## 1. Introduction
   ← your draft's paragraphs 1–2, plus the headline: 99.11%
## 2. Network Implementation

    I created the layers.py file to simulate the different layers in the neural network: Affine, Relu, and SoftmaxWithLoss. The Affine class's forward method returns the dot product of the input and the weights with the biases added in. Its backward method stores the derivative of the weights and biases in relationship with the loss and returns the gradient with respect to its input.  The Relu class's forward method returns the array with all the negative values zeroed out. The backward pass returns the backprop with all the masked values set to 0. The SoftmaxWithLoss's forward method applies the softmax and cross entropy error functions to the received data and returns it. Its backward method returns the difference between the prediction and the true label divided by the batchsize. I created network.py to simulate forward passes and backpropagation. The MultiLayerNet builds the layer chain from a sizes list such as [784, 100, 10]. The train.py file handles the training of the network.  

   ← layers/network/train description; gradient checking (~7 decimals); He init story
   
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
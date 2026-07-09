import numpy as np
import layers

class MultiLayerNet:
    def __init__(self):
        self.W1 = np.random.randn(784, 100) * 0.01 
        self.b1 = np.zeros(100)
        self.W2 = np.random.randn(100,10) * 0.01 
        self.b2 = np.zeros(10)
        self.layers=[layers.Affine(self.W1,self.b1),layers.Relu(),layers.Affine(self.W2,self.b2),]
        self.last_layer = layers.SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers: x = layer.forward(x)
        return x

    def loss(self, x, t):
        score = self.predict(x)
        return self.last_layer.forward(score,t)

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward()
        for rlayer in reversed(self.layers): dout=rlayer.backward(dout)
        dictionary = {'W1':self.layers[0].dW,'b1':self.layers[0].db,'W2':self.layers[2].dW,'b2':self.layers[2].db}
        return dictionary

    def accuracy(self, x, t):
        prediction=np.argmax(self.predict(x),axis=1)
        answer=np.argmax(t,axis=1)
        return np.mean(prediction==answer)

test = MultiLayerNet()
print(test.accuracy(np.random.randn(784).reshape(1,784),np.array([0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).reshape(1,10)))

# instance = MultiLayerNet()
# print(instance.W1.shape)
# print(instance.W2.shape)
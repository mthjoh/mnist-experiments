import numpy as np
import layers
import conv_layers

class SimpleConvNet:
    def __init__(self):
        self.W=[]
        self.b=[]
        self.layers=[]
        
        self.W.append(np.random.randn(16, 1, 3, 3) * np.sqrt(2/9))
        self.b.append(np.zeros(16))
        self.W.append(np.random.randn(2704, 100) * np.sqrt(2/2704))
        self.b.append(np.zeros(100))
        self.W.append(np.random.randn(100,10) * np.sqrt(2/100))
        self.b.append(np.zeros(10))

        self.layers.append(conv_layers.Convolution(self.W[0],self.b[0],1))
        self.layers.append(layers.Relu())
        self.layers.append(conv_layers.MaxPooling(2,2))
        self.layers.append(conv_layers.Flatten())
        self.layers.append(layers.Affine(self.W[1], self.b[1]))
        self.layers.append(layers.Relu())
        self.layers.append(layers.Affine(self.W[2], self.b[2]))

        self.last_layer = layers.SoftmaxWithLoss()
    
    def predict(self, x):
        for layer in self.layers: x = layer.forward(x)
        return x

    def loss(self, x, t):
        score = self.predict(x)
        return self.last_layer.forward(score, t)

    def accuracy(self, x, t):
        prediction = np.argmax(self.predict(x), axis=1)
        answer = np.argmax(t, axis=1)
        return np.mean(prediction == answer)

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward()
        for rlayer in reversed(self.layers): dout = rlayer.backward(dout)

        dW_list=[]
        db_list=[]

        for layer in self.layers:
            if isinstance(layer, (layers.Affine, conv_layers.Convolution)):
                dW_list.append(layer.dW)
                db_list.append(layer.db)
        return {'W': dW_list, 'b': db_list}
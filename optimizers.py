import numpy as np
class SGD:
    def __init__(self,lr):
        self.lr = lr
    
    def update(self,params,grads):
        for j in range(len(params['W'])):
            params['W'][j] -= self.lr*grads['W'][j]
            params['b'][j] -= self.lr*grads['b'][j]

class Momentum:
    def __init__(self, lr, mom):
        self.lr = lr
        self.v = None
        self.mom=mom


    def update(self,params,grads):
        if self.v is None: 
            self.v = {'W':[],'b':[]}
            for i in range(len(params['W'])):
                self.v['W'].append(np.zeros_like(params['W'][i]))
                self.v['b'].append(np.zeros_like(params['b'][i]))

            #self.v=np.zeros(params['W'].sizes, params['b'].sizes)

        for j in range(len(params['W'])):
            self.v['W'][j] = self.mom*self.v['W'][j] - self.lr*grads['W'][j]
            params['W'][j] += self.v['W'][j]
            self.v['b'][j] = self.mom*self.v['b'][j] - self.lr*grads['b'][j]
            params['b'][j] += self.v['b'][j]


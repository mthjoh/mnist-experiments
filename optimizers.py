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

class Adam:
    def __init__(self, lr, beta1, beta2):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = {'W':[], 'b':[]}
            for i in range(len(params['W'])):
                self.m['W'].append(np.zeros_like(params['W'][i]))
                self.m['b'].append(np.zeros_like(params['b'][i]))
        if self.v is None:
            self.v = {'W':[], 'b':[]}
            for i in range(len(params['W'])):
                self.v['W'].append(np.zeros_like(params['W'][i]))
                self.v['b'].append(np.zeros_like(params['b'][i]))
        self.t+=1
        for j in range(len(params['W'])):
            self.m['W'][j] = self.beta1*self.m['W'][j] + (1-self.beta1)*grads['W'][j]
            self.v['W'][j] = self.beta2*self.v['W'][j] + (1-self.beta2)*grads['W'][j]**2
            mup = self.m['W'][j]/(1-self.beta1**self.t)
            vup = self.v['W'][j]/(1-self.beta2**self.t)
            params['W'][j] -= self.lr*mup/(np.sqrt(vup)+1e-7)

            self.m['b'][j] = self.beta1*self.m['b'][j] + (1-self.beta1)*grads['b'][j]
            self.v['b'][j] = self.beta2*self.v['b'][j] + (1-self.beta2)*grads['b'][j]**2
            mup = self.m['b'][j]/(1-self.beta1**self.t)
            vup = self.v['b'][j]/(1-self.beta2**self.t)
            params['b'][j] -= self.lr*mup/(np.sqrt(vup)+1e-7)
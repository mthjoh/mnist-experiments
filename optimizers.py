class SGD:
    def __init__(self,lr):
        self.lr = lr
    
    def update(self,params,grads):
        for j in range(len(params['W'])):
            params['W'][j] -= self.lr*grads['W'][j]
            params['b'][j] -= self.lr*grads['b'][j]
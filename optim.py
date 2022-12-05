
from enum import Enum
from copy import deepcopy

import numpy as np

from nn import DenseNet


# optimizers
class Optimizer:
    class OptimType(Enum):
        GD = 1
        SGD = 2
        ProxGD = 3
        ProxSGD = 4
        Adam = 5

    def __init__(self, net: DenseNet, optim_type: OptimType, lr: float, lam: float) -> None:
        self.net = net
        self.optim_type = optim_type
        self.lr = lr
        self.lam = lam

    def grad(self, labels: np.ndarray) -> np.ndarray:
        """
        self.net.layers
        self.net.forward_data
        """
        out_layer = self.net.layers[-1]['result'] # bs x 10
        soft_pred: np.ndarray = self.softmax(out_layer)
        grad = []
        for i, layer in enumerate(self.net.layers):
            weight = np.zeros_like(layer['weight'])
            bias = np.zeros_like(layer['bias'])
            grad.append({'weight': weight, 'bias': bias})

        if self.optim_type == Optimizer.OptimType.GD \
                or self.optim_type == Optimizer.OptimType.SGD:
            dL_dz2: np.ndarray = soft_pred - labels # bs x 10
            a1 = self.net.layers[-2]['activation']
            dL_dW2 = np.dot(a1.T, dL_dz2)  # 64xbs * bsx10 = 64x10
            dL_db2: np.ndarray = np.sum(dL_dz2, axis=0, keepdims=True)  # 1x10
            grad[-1]['weight'] = dL_dW2
            grad[-1]['bias'] = dL_db2.T

            da1_dz1 = a1 * (1 - a1)  # bs x 64
            W2 = self.net.layers[-1]['weight'] # 784x64
            dL_dz1 = np.dot(dL_dz2, W2.T) * da1_dz1  # (bsx10 dot 10x64) mul bsx64 = bsx64
            dL_dW1 = np.dot(self.net.forward_data.T, dL_dz1) # 784xbs * bsx64 = 784x64
            dL_db1: np.ndarray = np.sum(dL_dz1, axis=0, keepdims=True) # 1x64
            grad[-2]['weight'] = dL_dW1
            grad[-2]['bias'] = dL_db1.T

        elif self.optim_type == Optimizer.OptimType.ProxGD:
            pass
        elif self.optim_type == Optimizer.OptimType.ProxSGD:
            pass
        elif self.optim_type == Optimizer.OptimType.Adam:
            pass

        return grad

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_sum = np.sum(np.exp(x), axis=1, keepdims=True)
        return np.exp(x) / exp_sum

    def backward(self, y: np.ndarray,) -> None:

        batch_size = y.shape[0]
        out_layer = self.net.layers[-1]['result'] # bs x 10
        # softmax
        soft_pred = self.softmax(out_layer)
        labels = np.zeros_like(soft_pred)
        labels[np.arange(batch_size), y] = 1 # bs x 10

        # cross entropy loss
        loss = np.multiply(labels, np.log(soft_pred)) # bs x 10
        loss = -np.sum(loss) / batch_size
        # regularization
        reg = 0
        for layer in self.net.layers:
            reg += np.sum(layer['weight'] ** 2)
        reg *= self.lam / 2
        loss += reg

        # calc gradient 
        grad = self.grad(labels)
        
        # update
        for i, layer in enumerate(grad):
            self.net.layers[i]['weight'] -= self.lr * layer['weight']
            self.net.layers[i]['bias'] -= self.lr * layer['bias']
        
        return loss

        

        
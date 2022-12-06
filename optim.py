
from abc import ABC, abstractmethod
from enum import Enum
from copy import deepcopy

import numpy as np

from nn import DenseNet

class OptimType(Enum):
    SGD = 1
    ProxSGD = 2
    ProxSGD_LR = 3
    Adam = 4


# optimizers
class Optimizer(ABC):
    # base class for optimizers

    def __init__(self, net: DenseNet, lr: float, ) -> None:
        self.net = net
        self.lr = lr

    def grad(self, labels: np.ndarray) -> 'list[dict[str, np.ndarray]]':
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

        return grad

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_sum = np.sum(np.exp(x), axis=1, keepdims=True)
        return np.exp(x) / exp_sum

    @abstractmethod
    def update(labels) -> float:
        # update = self.grad(labels)
        pass

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
        # regularization, not included in pytorch
        # reg = 0
        # for layer in self.net.layers:
        #     reg += np.sum(layer['weight'] ** 2)
        #     reg += np.sum(layer['bias'] ** 2)
        # reg *= self.lam / 2
        # loss += reg

        # calc gradient 
        grad = self.update(labels)
        
        # update
        for i, layer in enumerate(grad):
            self.net.layers[i]['weight'] -= self.lr * layer['weight']
            self.net.layers[i]['bias'] -= self.lr * layer['bias']
        
        return loss

        
class SGD(Optimizer):
    
        def __init__(self, net: DenseNet, lr: float, lam: float) -> None:
            super().__init__(net, lr)
            self.lam = lam
    
        def update(self, labels: np.ndarray) -> np.ndarray:
            grad = self.grad(labels)
            # grad from regularization
            for i, layer in enumerate(grad):
                grad[i]['weight'] += self.lam * self.net.layers[i]['weight']
                grad[i]['bias'] += self.lam * self.net.layers[i]['bias']
                
            return grad


class ProxSGD(Optimizer):
    
        def __init__(self, net: DenseNet, lr: float, lam: float) -> None:
            # set lam = 0 to cancel l2 norm regularization
            super().__init__(net, lr)
            self.lam = lam
    
        def update(self, labels: np.ndarray) -> np.ndarray:
            grad = self.grad(labels)
            # add l1 norm regularization
            for i, layer in enumerate(grad):
                layer['weight'] += self.lam * np.sign(self.net.layers[i]['weight'])
                layer['bias'] += self.lam * np.sign(self.net.layers[i]['bias'])

            return grad


class ProxSGD_LR(Optimizer):
        
            def __init__(self, net: DenseNet, lr: float, lam1: float=0.01, lam2: float=0.01) -> None:
                # set lam = 0 to cancel l2 norm regularization
                super().__init__(net, lr)
                self.lams = [lam1, lam2]
        
            def update(self, labels: np.ndarray) -> np.ndarray:
                grad = self.grad(labels)
                # add nuclear norm regularization to enforce low rank
                for i, layer in enumerate(grad):
                    weight = self.net.layers[i]['weight']
                    u, s, v = np.linalg.svd(weight, full_matrices=False)
                    threshold = self.lams[i] * self.lr * 1000
                    s = np.where(s > threshold, s - threshold, 0)
                    layer['weight'] += self.lams[i] * np.dot(u, np.dot(np.diag(s), v))
                    
                    # U, sigma, V = np.linalg.svd(L, 0)
                    # L_thr = lr_L * tau
                    # sigma = np.where(sigma > L_thr, sigma - L_thr, 0)
                    # sigma = np.diag(sigma)
                    # L = np.dot(np.dot(U, sigma), V)


                return grad


class Adam(SGD):
        
    def __init__(self, net: DenseNet, lr: float=0.01, lam: float=0.1, beta1: float=0.9, beta2: float=0.9) -> None:
        super().__init__(net, lr, lam)
        self.beta1 = beta1
        self.beta2 = beta2

        self.M = []
        self.V = []
        for layer in self.net.layers:
            weight = np.zeros_like(layer['weight'])
            bias = np.zeros_like(layer['bias'])
            self.M.append({'weight': weight, 'bias': bias})
            
            weight = np.zeros_like(layer['weight'])
            bias = np.zeros_like(layer['bias'])
            self.V.append({'weight': weight, 'bias': bias})

    def update(self, labels: np.ndarray) -> np.ndarray:
        grad = self.grad(labels)
        for i, layer in enumerate(grad):
            self.M[i]['weight'] = self.beta1 * self.M[i]['weight'] + (1 - self.beta1) * layer['weight']
            self.M[i]['bias'] = self.beta1 * self.M[i]['bias'] + (1 - self.beta1) * layer['bias']
            self.V[i]['weight'] = self.beta2 * self.V[i]['weight'] + (1 - self.beta2) * layer['weight'] ** 2
            self.V[i]['bias'] = self.beta2 * self.V[i]['bias'] + (1 - self.beta2) * layer['bias'] ** 2


        for i, layer in enumerate(grad):
            grad[i]['weight'] = self.M[i]['weight'] / (np.sqrt(self.V[i]['weight']) + 1e-8)
            grad[i]['bias'] = self.M[i]['bias'] / (np.sqrt(self.V[i]['bias']) + 1e-8)

        return grad


        

        
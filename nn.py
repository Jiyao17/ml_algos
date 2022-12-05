

# for building the model

import numpy as np


class DenseNet:
    def __init__(self, ):
        # 3-layer dense network for MNIST
        # include all intermediate variables
        self.W1 = np.random.normal(0, 0.01, size=(784, 64))
        self.b1 = np.random.normal(0, 0.01, size=(64, 1))
        self.z1: np.ndarray = None # bs x 64
        self.a1: np.ndarray = None # bs x 64
        self.W2 = np.random.normal(0, 0.01, size=(64, 10))
        self.b2 = np.random.normal(0, 0.01, size=(10, 1))
        self.z2: np.ndarray = None # bs x 10

        self.layer1 = { "weight": self.W1, "bias": self.b1, "result": self.z1, "activation": self.a1 }
        self.layer2 = { "weight": self.W2, "bias": self.b2, "result": self.z2 }
        self.layers: 'list[dict[str, np.ndarray]]' = [ self.layer1, self.layer2 ]

        self.forward_data: np.ndarray = None

    def sigmoid(self, x, out=None):
        if out is None:
            out = np.empty_like(x)
            
        return np.divide(1, 1 + np.exp(-x), out=out)

    def forward(self, X: np.ndarray,) -> np.ndarray:
        # flatten the input
        batch_size = X.shape[0]
        if self.z1 is None or self.z1.shape[0] != batch_size:
            self.z1 = np.random.normal(0, 0.01, size=(batch_size, 64))
            self.a1 = np.random.normal(0, 0.01, size=(batch_size, 64))
            self.z2 = np.random.normal(0, 0.01, size=(batch_size, 10))
            self.layer1["result"] = self.z1
            self.layer1["activation"] = self.a1
            self.layer2["result"] = self.z2

        X = X.reshape(batch_size, -1) # flatten to bs x 784
        self.forward_data = X
        np.dot(X, self.W1, out=self.z1) + self.b1.T # bs x 64
        self.sigmoid(self.z1, out=self.a1) # bs x 64
        np.dot(self.a1, self.W2, out=self.z2) + self.b2.T # bs x 10

        pred = np.argmax(self.z2, axis=1, keepdims=True) # bs x 1
        # accu = np.sum(pred == y) / batch_size
        # # cross entropy loss
        # loss = -np.sum(np.log(z2[np.arange(batch_size), y])) / batch_size
        return pred



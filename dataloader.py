

from torchvision.datasets import MNIST

import numpy as np


def get_MNIST_np(path, download=True):
    trainset = MNIST(path, train=True, download=download)
    testset = MNIST(path, train=False, download=download)
    
    traindata = trainset.data.numpy() / 255
    trainlabel = trainset.targets.numpy()
    testdata = testset.data.numpy() / 255
    testlabel = testset.targets.numpy()

    trainset = list(zip(traindata, trainlabel))
    testset = list(zip(testdata, testlabel))

    return trainset, testset


def dataloader(dataset, batch_size, shuffle=True, drop_last=True):
    """
    batch generator
    """
    if shuffle:
        np.random.shuffle(dataset)
        
    X, y = zip(*dataset)
    X = np.array(X)
    y = np.array(y)
    
    for i in range(0, len(X), batch_size):
        if i + batch_size > len(X):
            if drop_last:
                break
            else:
                yield X[i:], y[i:]
        else:
            yield X[i:i+batch_size], y[i:i+batch_size]
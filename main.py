

from dataloader import get_MNIST_np, dataloader
from nn import DenseNet
from optim import Optimizer

import numpy as np


def main():
    # load data
    trainset, testset = get_MNIST_np('./data', download=True)

    # build model
    net = DenseNet()
    optim = Optimizer(net, Optimizer.OptimType.SGD, lr=0.01, lam=0)

    # train
    epoch = 500
    training_loss = []
    testing_accu = []
    i = 0
    while i < epoch:
        for X, y in dataloader(trainset, batch_size=128, shuffle=True, drop_last=True):
            pred = net.forward(X)
            loss = optim.backward(y)

            # test
            correct = 0
            for X, y in dataloader(testset, batch_size=1000, shuffle=False, drop_last=False):
                pred = net.forward(X)
                correct += np.sum(pred.squeeze() == y)
            accu = correct / len(testset)

            training_loss.append(loss)
            testing_accu.append(accu)

            if (i + 1) % 10 == 0:
                print('Epoch: {}, Loss: {}, Accu: {}'.format(i, loss, accu))

            i += 1
            if i >= epoch:
                break


    return training_loss, testing_accu


if __name__ == '__main__':
    training_loss, testing_accu = main()
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(training_loss)
    plt.show()
    plt.figure()
    plt.plot(testing_accu)
    plt.show()



from dataloader import get_MNIST_np, dataloader
from nn import DenseNet
from optim import SGD, Adam, ProxSGD, ProxSGD_LR

import numpy as np


def main():
    # load data
    trainset, testset = get_MNIST_np('./data', download=True)

    # build model
    net = DenseNet(loc=0, scale=0.01)
    # optim = SGD(net, lr=0.01, lam=0.1)
    # optim = Adam(net, lr=0.01, lam=0.1, beta1=0.9, beta2=0.9)
    # optim = ProxSGD(net, lr=0.01, lam=0.1)
    optim = ProxSGD_LR(net, lr=0.01)



    # train
    epoch = 1000
    training_loss = []
    testing_accu = []
    norms = []
    ranks = []
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

            norm = net.get_norm()
            norms.append(norm)
            rank0, rank1 = net.get_ranks()
            ranks.append(rank0)

            if (i + 1) % 10 == 0:
                print('epoch: {}, loss: {}, accu: {}'.format(i + 1, loss, accu))
                print('norm: {}, rank0: {}, rank1: {}'.format(norm, rank0, rank1))

            i += 1
            if i >= epoch:
                break


    return training_loss, testing_accu, norms, ranks


if __name__ == '__main__':
    training_loss, testing_accu, norms, ranks = main()
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(training_loss)
    plt.show()
    plt.figure()
    plt.plot(testing_accu)
    plt.show()
    plt.figure()
    plt.plot(norms)
    plt.show()
    plt.figure()
    plt.plot(ranks)
    plt.show()


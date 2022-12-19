
import pickle

from src.dataloader import get_MNIST_np, dataloader
from src.nn import DenseNet
from src.optim import OptimType, SGD, Adam, ProxSGD, ProxSGD_LR, ADMM

import numpy as np
import matplotlib.pyplot as plt




def main(optim_type=OptimType.SGD):
    # load data
    trainset, testset = get_MNIST_np('./data', download=True)

    # build model
    net = DenseNet(loc=0, scale=0.01)
    if optim_type == OptimType.SGD:
        optim = SGD(net, lr=0.01, lam=0.1)
    elif optim_type == OptimType.Adam:
        optim = Adam(net, lr=0.01, lam=0.1, beta1=0.9, beta2=0.9)
    elif optim_type == OptimType.ProxSGD:
        optim = ProxSGD(net, lr=0.01, lam=0.1)
    elif optim_type == OptimType.ProxSGD_LR:
        optim = ProxSGD_LR(net, lr=0.01)
    elif optim_type == OptimType.ADMM:
        optim = ADMM(net, lr=0.01, lam=0.01, rho=0.01)
    else:
        raise NotImplementedError
    # optim = SGD(net, lr=0.01, lam=0.1)
    # optim = Adam(net, lr=0.01, lam=0.1, beta1=0.9, beta2=0.9)
    # optim = ProxSGD(net, lr=0.01, lam=0.1)
    # optim = ProxSGD_LR(net, lr=0.01)

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

    optimizers = [OptimType.SGD, OptimType.Adam, OptimType.ProxSGD, OptimType.ProxSGD_LR, OptimType.ADMM]
    opt = [OptimType.ADMM]
    for optim_type in opt:
        print('optim_type: {}'.format(optim_type))
        training_loss, testing_accu, norms, ranks = main(optim_type)

        pickle.dump(training_loss, open('result/data/training_loss_{}.pkl'.format(optim_type), 'wb'))
        pickle.dump(testing_accu, open('result/data/testing_accu_{}.pkl'.format(optim_type), 'wb'))

        plt.figure()
        plt.plot(training_loss)
        plt.savefig('result/fig/training_loss_{}.png'.format(optim_type))
        plt.figure()
        plt.plot(testing_accu)
        plt.savefig('result/fig/testing_accu_{}.png'.format(optim_type))
        plt.figure()
        plt.plot(norms)
        plt.savefig('result/fig/norms_{}.png'.format(optim_type))
        plt.figure()
        plt.plot(ranks)
        plt.savefig('result/fig/ranks_{}.png'.format(optim_type))


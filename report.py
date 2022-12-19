
import pickle
import os

from matplotlib import pyplot as plt

from src.optim import OptimType

result_dir = 'result/data'

accu_sgd = pickle.load(open(os.path.join(result_dir, 'testing_accu_{}.pkl'.format(OptimType.SGD)), 'rb'))
accu_adam = pickle.load(open(os.path.join(result_dir, 'testing_accu_{}.pkl'.format(OptimType.Adam)), 'rb'))
accu_proxsgd = pickle.load(open(os.path.join(result_dir, 'testing_accu_{}.pkl'.format(OptimType.ProxSGD)), 'rb'))
accu_proxsgd_lr = pickle.load(open(os.path.join(result_dir, 'testing_accu_{}.pkl'.format(OptimType.ProxSGD_LR)), 'rb'))
accu_admm = pickle.load(open(os.path.join(result_dir, 'testing_accu_{}.pkl'.format(OptimType.ADMM)), 'rb'))

loss_sgd = pickle.load(open(os.path.join(result_dir, 'training_loss_{}.pkl'.format(OptimType.SGD)), 'rb'))
loss_adam = pickle.load(open(os.path.join(result_dir, 'training_loss_{}.pkl'.format(OptimType.Adam)), 'rb'))
loss_proxsgd = pickle.load(open(os.path.join(result_dir, 'training_loss_{}.pkl'.format(OptimType.ProxSGD)), 'rb'))
loss_proxsgd_lr = pickle.load(open(os.path.join(result_dir, 'training_loss_{}.pkl'.format(OptimType.ProxSGD_LR)), 'rb'))
loss_admm = pickle.load(open(os.path.join(result_dir, 'training_loss_{}.pkl'.format(OptimType.ADMM)), 'rb'))

plt.figure()
plt.plot(accu_sgd, label='SGD')
plt.plot(accu_adam, label='Adam')
plt.legend()
plt.savefig('result/fig/accu_SGD_Adam.png')
plt.figure()
plt.plot(loss_sgd, label='SGD')
plt.plot(loss_adam, label='Adam')
plt.legend()
plt.savefig('result/fig/loss_SGD_Adam.png')


# print maximum accuracy
print('max accu of SGD: {}'.format(max(accu_sgd)))
print('max accu of Adam: {}'.format(max(accu_adam)))
print('max accu of ProxSGD: {}'.format(max(accu_proxsgd)))
print('max accu of ProxSGD_LR: {}'.format(max(accu_proxsgd_lr)))
print('max accu of ADMM: {}'.format(max(accu_admm)))


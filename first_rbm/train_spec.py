from __future__ import print_function
from __future__ import print_function
from random import randint

import datetime
import numpy as np
from matplotlib import animation
from numpy.random.mtrand import permutation

import alt_rbm
import import_data
import matplotlib.pyplot as plt
from logistic_sgd import load_data
import gzip
import six.moves.cPickle as pickle
import cv2

fig = plt.figure()
visual = None
im = plt.imshow(np.zeros((28, 28)), cmap=plt.get_cmap('gray'), animated=True)
i = 0
epochs = 200
hidden_layers = 100
lr = 0.1


def get_index():
    global i
    i += 1
    if i >= len(visual):
        i = 0
        print('restart')
    print(i)
    return i


def updatefig(*args):
    global visual
    mat = np.reshape(visual[get_index()], (28, 28))
    im.set_array(mat)
    return im,


if __name__ == '__main__':
    global visual
    img, val = import_data.load_dataset1()
    index = 0
    # new dataset
    data = np.zeros((100, 28, 28))
    for i in range(0, len(img)):
        if index < 100 and val[i] == 7:
            data[index] = img[i]
            index += 1
    data = np.resize(data, (len(data), 784))
    rbm = alt_rbm.RBM(784, hidden_layers, learning_rate=lr)
    rbm.train(data[0:200], epochs)
    f = open("./models/epochs{}_hidden{}_lr{}.pkl".format(epochs, hidden_layers, lr), 'wb')
    pickle.dump(rbm, f, protocol=2)
    f.close()
    # f = open('./models/epochs5000_hidden1000_lr0.01.pkl', 'rb')
    # rbm = pickle.load(f)
    initial = np.random.rand(101)
    im = plt.imshow(np.random.rand(28, 28), cmap=plt.get_cmap('gray'), animated=True)
    visual = rbm.daydream(15, initial)

    ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=True)
    plt.show()
    # for i in range(1, 10):
    #     initial = rbm.run_visible(initial)
    #     initial = rbm.run_hidden(initial)
    #     res_img = np.reshape(initial[1], (28, 28))
    #     original = data[52]
    #     original = np.reshape(original, (28, 28))
    #     cv2.imshow("orig", cv2.resize(original, (300, 300)))
    #     cv2.imshow("w", cv2.resize(res_img, (300, 300)))
    #     cv2.waitKey(0)



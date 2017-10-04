import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


class NeuralNet:
    def __init__(self, input, output):
        self.input = input
        self.output = output
        self.w0 = 2 * np.random.random((input.shape[1], input.shape[0])) - 1
        self.w1 = 2 * np.random.random((output.shape[0], output.shape[1])) - 1

    def train(self, epochs):
        for epoch in range(epochs):

            # Feed forward through layers 0, 1, and 2
            l0 = self.input
            l1 = sigmoid(np.dot(l0, self.w0))
            l2 = sigmoid(np.dot(l1, self.w1))

            # how much did we miss the target value?
            l2_error = y - l2

            if (epoch % 10000) == 0:
                print "Error:" + str(np.mean(np.abs(l2_error)))

            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            l2_delta = l2_error * sigmoid(l2, deriv=True)

            # how much did each l1 value contribute to the l2 error (according to the weights)?
            l1_error = l2_delta.dot(self.w1.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            l1_delta = l1_error * sigmoid(l1, deriv=True)

            self.w1 += l1.T.dot(l2_delta)
            self.w0 += l0.T.dot(l1_delta)

    def test(self, test_case):
        t1 = sigmoid(np.dot(test_case, self.w0))
        t2 = sigmoid(np.dot(t1, self.w1))
        print(t2)


if __name__ == '__main__':
    num_epochs = 50000
    x = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    net = NeuralNet(x, y)
    net.train(100000)
    net.test(x[0])

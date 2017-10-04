# simple perceptron that implements boolean functions
from random import choice
from numpy import array, dot, random
import matplotlib.pyplot as plt

unit_step = lambda x: 0 if x < 0 else 1

training_data = [
    (array([0, 0, 1]), 1),
    (array([0, 1, 0]), 0),
    (array([1, 0, 0]), 0),
    (array([1, 1, 1]), 1),
]

w = random.rand(3)
errors = []
eta = 0.1
n = 100

for i in xrange(n):
    # choose random element from training batch
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    # errors appended to see stability graph
    errors.append(error)
    # batch delta rule
    w += eta * error * x

for x, _ in training_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

plt.plot(errors)
plt.xlabel('epochs')
plt.ylabel('error')
plt.show()

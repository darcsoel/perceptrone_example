import numpy as np

test_dim = 2
test = np.random.rand(test_dim)


class Perceptron:
    bias = 1.0

    def __init__(self, inputs):
        self.weights = np.random.rand(inputs + 1)

    def run(self, x):
        sum_ = np.dot(np.append(x, self.bias), self.weights)
        return self._sigmoid(sum_)

    def set_weights(self, weights):
        self.weights = weights

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

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


def print_weights(perceptron):
    print('{0:.10f}'.format(perceptron.run([0, 0])))
    print('{0:.10f}'.format(perceptron.run([0, 1])))
    print('{0:.10f}'.format(perceptron.run([1, 0])))
    print('{0:.10f}'.format(perceptron.run([1, 1])))


if __name__ == '__main__':
    p = Perceptron(test_dim)
    print_weights(p)

    print()

    p.weights = [10, 10, -15]  # AND
    # p.weights = [15, 15, -10] #OR
    print_weights(p)

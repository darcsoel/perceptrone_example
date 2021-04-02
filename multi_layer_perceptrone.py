import numpy as np

from perceptron import Perceptron


class MultiLayersPerceptron:
    bias = 1.0

    def __init__(self, layers):
        self.layers = np.array(layers, dtype=object)
        network = []
        values = []

        for i, _ in enumerate(self.layers):
            network.append([])
            values.append([])
            values[i] = [0.0 for _ in range(self.layers[i])]

            if i > 0:
                for j in range(self.layers[i]):
                    perceptron = Perceptron(self.layers[i - 1])
                    perceptron.bias = self.bias
                    network[i].append(perceptron)

        self.network = np.array([np.array(x) for x in network], dtype=object)
        self.values = np.array([np.array(x) for x in values], dtype=object)

    def run(self, x):
        x = np.array(x, dtype=object)
        self.values[0] = x

        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i - 1])

        return self.values[-1]

    def set_weights(self, weights):
        for i, _ in enumerate(weights):
            for j, _ in enumerate(weights[i]):
                self.network[i + 1][j].weights = weights[i][j]


p = MultiLayersPerceptron([2, 2, 1])
p.set_weights([[[-10, -10, 15], [15, 15, -10]], [[10, 10, -15]]])

print('{0:.10f}'.format(p.run([0, 0])[0]))
print('{0:.10f}'.format(p.run([0, 1])[0]))
print('{0:.10f}'.format(p.run([1, 0])[0]))
print('{0:.10f}'.format(p.run([1, 1])[0]))

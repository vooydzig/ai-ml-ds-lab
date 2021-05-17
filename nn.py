import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x: float) -> float:
    fx = sigmoid(x)
    return fx * (1 - fx)


def cost(values: np.ndarray, predictions: np.ndarray) -> float:
    # ((v1-p1)^2 + (v2-p2)^2 + ... + (vn-pn)^2)/n
    return ((values - predictions) ** 2).mean()


class Neuron:
    def __init__(self, weights: np.ndarray, bias: float, name: str):
        self.name = name
        self.inputs = None
        self.weights = weights
        self.bias = bias
        self.value = 0
        self.activation = 0

    def feedforward(self, inputs: np.ndarray) -> float:
        self.inputs = inputs
        self.value = np.dot(self.weights, inputs) + self.bias
        self.activation = sigmoid(self.value)
        return self.activation

    def backpropagation(self, learn_rate: float, deriv: float, prediction: float) -> np.ndarray:
        deriv_bias = deriv_sigmoid(self.value)
        deriv_weights = np.multiply(self.inputs, deriv_sigmoid(self.value))
        deriv_previous = np.multiply(self.weights, deriv_sigmoid(self.value))
        self.weights = self.weights - learn_rate * deriv * prediction * deriv_weights
        self.bias = self.bias - learn_rate * deriv * prediction * deriv_bias
        return deriv_previous


class NeuralLayer:
    def __init__(self, neurons: int, weights: int = 2):
        self.neurons = []
        if neurons:
            self.neurons = [
                Neuron(
                    weights=np.array([
                        np.random.normal() for w in range(weights)
                    ]),
                    bias=np.random.normal(),
                    name=f'Neuron {n}'
                )
                for n in range(neurons)
            ]

    def backpropagation(self, learn_rate, deriv, prediction):
        deriv_previous = None
        for i, neuron in enumerate(self.neurons):
            pred = neuron.backpropagation(learn_rate, deriv, prediction[i])
            deriv_previous = pred if deriv_previous is None else (deriv_previous + pred)
        return deriv_previous


class NeuralNetwork:
    def __init__(self, inputs=4):
        self.layers = []
        self.losses = []
        self.inputs = inputs

    @property
    def output(self):
        if self.layers:
            return self.layers[-1]
        return None

    def add_layer(self, neurons):
        output = self.output
        if output is None:
            self.layers.append(NeuralLayer(neurons=neurons, weights=self.inputs))
        else:
            self.layers.append(NeuralLayer(neurons=neurons, weights=len(output.neurons)))

    def feedforward(self, x: np.ndarray):
        if not self.layers:
            return 0
        inputs = x
        for i, layer in enumerate(self.layers[:-1]):
            inputs = [n.feedforward(inputs) for n in layer.neurons]
        return np.array([o.feedforward(np.array(inputs)) for o in self.output.neurons])

    def backpropagation(self, learn_rate: float, deriv: float, prediction: np.ndarray):
        for layer in reversed(self.layers):
            prediction = layer.backpropagation(learn_rate, deriv, prediction)

    def train(self, samples: np.ndarray, labels: np.ndarray, learn_rate: float=0.1, epochs: int=1000):
        self.losses = []

        for epoch in range(epochs):
            for sample, label in zip(samples, labels):
                prediction = self.feedforward(sample)
                deriv = -2 * (label - prediction[0])
                self.backpropagation(learn_rate, deriv, np.array([1 for _ in range(len(self.output.neurons))]))

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(lambda x: self.feedforward(x)[0], 1, data)
                loss = cost(labels, y_preds)
                self.losses.append(loss)

    def draw_loss_plot(self):
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(range(len(self.losses)), self.losses)
        plt.show()

    def print_network(self):
        print('Network debug:')
        for i, l in enumerate(self.layers):
            print(f'\tLayer {i}')
            for j, n in enumerate(l.neurons):
                print(f'\t\t{n.name}: {n.weights}')

if __name__=='__main__':
    data = np.array([
        [-2, -1],  # Alice
        [25, 6],  # Bob
        [17, 4],  # Charlie
        [-15, -6],  # Diana
    ])
    labels = np.array([
        1,  # Alice
        0,  # Bob
        0,  # Charlie
        1,  # Diana
    ])

    network = NeuralNetwork(inputs=2)
    network.add_layer(neurons=2)
    network.add_layer(neurons=2)
    network.add_layer(neurons=1)

    # Make some predictions
    emily = np.array([-7, -3])
    frank = np.array([20, 2])
    print('Not trained network')
    print("Emily: %.3f" % network.feedforward(emily))
    print("Frank: %.3f" % network.feedforward(frank))

    network.train(data, labels)
    print('Trained network')
    print("Emily: %.3f" % network.feedforward(emily))  # ~ 1 - F
    print("Frank: %.3f" % network.feedforward(frank))  # ~ 0 - M
    network.draw_loss_plot()

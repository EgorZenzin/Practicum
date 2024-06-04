import numpy as np
import pickle

class MLP_regression:
    def __init__(self, layer_sizes, activation_functions):
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.weights = []
        self.biases = []
        self.activations = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(np.random.randn(layer_sizes[i + 1]))
            self.activations.append(np.zeros(layer_sizes[i + 1]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def derivative(self, x, activation_function):
        if activation_function == 'sigmoid':
            return x * (1 - x)
        elif activation_function == 'tanh':
            return 1 - np.power(x, 2)
        elif activation_function == 'relu':
            return (x > 0).astype(float)

    def forward(self, x):
        for i in range(len(self.weights)):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            if self.activation_functions[i] == 'sigmoid':
                x = self.sigmoid(x)
            elif self.activation_functions[i] == 'tanh':
                x = self.tanh(x)
            elif self.activation_functions[i] == 'relu':
                x = self.relu(x)
            self.activations[i] = x
        return x

    def backward(self, x, y, learning_rate):
        output = self.forward(x)
        deltas = []
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                error = output - y
            else:
                error = np.dot(deltas[-1], self.weights[i + 1].T)
            delta = error * self.derivative(self.activations[i], self.activation_functions[i])
            deltas.append(delta)
        deltas.reverse()
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(x.T if i == 0 else self.activations[i - 1].T, deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0)

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            self.backward(X, y, learning_rate)

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum(np.square(y_true - y_pred))
        ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
        return 1 - ss_res / ss_tot

    def save_parameters(self, filename):
        parameters = {"weights": self.weights, "biases": self.biases}
        with open(filename, "wb") as file:
            pickle.dump(parameters, file)

    def load_parameters(self, filename):
        with open(filename, "rb") as file:
            parameters = pickle.load(file)
        self.weights = parameters["weights"]
        self.biases = parameters["biases"]

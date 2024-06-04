import numpy as np
import pickle


class MLP_clasification:
    def __init__(self, layer_sizes, activation_functions):
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.weights = [
            np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0) * 1

    def forward_propagation(self, input_data):
        for b, w, activation in zip(
            self.biases, self.weights, self.activation_functions
        ):
            if activation == "sigmoid":
                input_data = self.sigmoid(np.dot(w, input_data) + b)
            elif activation == "tanh":
                input_data = self.tanh(np.dot(w, input_data) + b)
            elif activation == "relu":
                input_data = self.relu(np.dot(w, input_data) + b)
        return input_data

    def backpropagation(self, input_data, target):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        # Forward
        activation = input_data
        activations = [input_data]
        zs = []

        for b, w, activation_function in zip(
            self.biases, self.weights, self.activation_functions
        ):
            z = np.dot(w, activation) + b
            zs.append(z)
            if activation_function == "sigmoid":
                activation = self.sigmoid(z)
            elif activation_function == "tanh":
                activation = self.tanh(z)
            elif activation_function == "relu":
                activation = self.relu(z)
            activations.append(activation)

        # Backward
        delta = self.cost_derivative(
            activations[-1], target
        ) * self.get_activation_derivative(
            activations[-1], self.activation_functions[-1]
        )
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, len(self.layer_sizes)):
            z = zs[-l]
            activation_derivative = self.get_activation_derivative(
                z, self.activation_functions[-l]
            )
            delta = np.dot(self.weights[-l + 1].T, delta) * activation_derivative
            gradient_b[-l] = delta
            gradient_w[-l] = np.dot(delta, activations[-l - 1].T)

        return gradient_b, gradient_w

    def get_activation_derivative(self, z, activation_function):
        if activation_function == "sigmoid":
            return self.sigmoid_derivative(z)
        elif activation_function == "tanh":
            return self.tanh_derivative(z)
        elif activation_function == "relu":
            return self.relu_derivative(z)

    def cost_derivative(self, output_activations, target):
        return output_activations - target

    def update_parameters(self, mini_batch, learning_rate):
        sum_gradient_b = [np.zeros(b.shape) for b in self.biases]
        sum_gradient_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_gradient_b, delta_gradient_w = self.backpropagation(x, y)
            sum_gradient_b = [
                nb + dnb for nb, dnb in zip(sum_gradient_b, delta_gradient_b)
            ]
            sum_gradient_w = [
                nw + dnw for nw, dnw in zip(sum_gradient_w, delta_gradient_w)
            ]

        self.weights = [
            w - (learning_rate / len(mini_batch)) * nw
            for w, nw in zip(self.weights, sum_gradient_w)
        ]
        self.biases = [
            b - (learning_rate / len(mini_batch)) * nb
            for b, nb in zip(self.biases, sum_gradient_b)
        ]

    def save_parameters(self, filename):
        parameters = {"weights": self.weights, "biases": self.biases}
        with open(filename, "wb") as file:
            pickle.dump(parameters, file)

    def load_parameters(self, filename):
        with open(filename, "rb") as file:
            parameters = pickle.load(file)
        self.weights = parameters["weights"]
        self.biases = parameters["biases"]

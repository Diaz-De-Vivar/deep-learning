import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
import numpy as np
import nnfs


np.random.seed(42)
nnfs.init()

# Updated Layer_Dense class with activation function support
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, activation=None):
        """
        Initialize weights, biases, and activation function for the layer.
        :param n_inputs: Number of inputs to the layer
        :param n_neurons: Number of neurons in the layer
        :param activation: Activation function (e.g., relu, sigmoid, tanh, softmax)
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    def forward_prop(self, inputs):
        """
        Perform the forward pass for the layer.
        :param inputs: Input data
        """
        self.output_raw = inputs @ self.weights + self.biases
        self.output = self.activation(self.output_raw) if self.activation else self.output_raw

# Define activation functions as a dictionary for better modularity
activation_functions = {
    "linear": lambda x: x,
    "relu": lambda x: np.maximum(0, x),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh": lambda x: np.tanh(x),
    "softmax": lambda x: np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)
}        
import numpy as np
import torch
import torch.nn.functional as F
import nnfs
from nnfs.datasets import spiral_data, vertical_data

# Set seeds for reproducibility
np.random.seed(42)
nnfs.init()

class Backend:
    """Base class for different computation backends"""
    @staticmethod
    def get_backend(name):
        backends = {
            'numpy': NumPyBackend,
            'torch': TorchBackend
        }
        if name not in backends:
            raise ValueError(f"Unsupported backend '{name}'. Use one of: {list(backends.keys())}")
        return backends[name]()

class NumPyBackend(Backend):
    """NumPy implementation of computation backend"""
    def __init__(self):
        self.name = 'numpy'
        self.device = None

    def initialize_weights(self, n_inputs, n_neurons):
        return 0.01 * np.random.randn(n_inputs, n_neurons)

    def initialize_biases(self, n_neurons):
        return np.zeros((1, n_neurons))

    def forward(self, inputs, weights, biases):
        return inputs @ weights + biases

    # Activation functions
    def linear(self, x):
        return x

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # Loss function
    def categorical_cross_entropy(self, y_pred, y_true, epsilon=1e-15):
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred_clipped), axis=1, keepdims=True)

class TorchBackend(Backend):
    """PyTorch implementation of computation backend"""
    def __init__(self):
        self.name = 'torch'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize_weights(self, n_inputs, n_neurons):
        return 0.01 * torch.randn(n_inputs, n_neurons, device=self.device)

    def initialize_biases(self, n_neurons):
        return torch.zeros(1, n_neurons, device=self.device)

    def forward(self, inputs, weights, biases):
        return inputs @ weights + biases

    # Activation functions
    def linear(self, x):
        return x

    def relu(self, x):
        return F.relu(x)

    def sigmoid(self, x):
        return torch.sigmoid(x)

    def tanh(self, x):
        return torch.tanh(x)

    def softmax(self, x):
        return F.softmax(x, dim=1)

    # Loss function
    def categorical_cross_entropy(self, y_pred, y_true, epsilon=1e-15):
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        loss = -torch.sum(y_true * torch.log(y_pred), dim=1)
        return torch.mean(loss)

class Layer_Dense:
    """Dense (fully connected) neural network layer"""
    def __init__(self, n_inputs, n_neurons, activation=None, backend='numpy'):
        """
        Initialize weights, biases, and activation function for the layer.

        Args:
            n_inputs: Number of inputs to the layer
            n_neurons: Number of neurons in the layer
            activation: Activation function name (e.g., 'relu', 'sigmoid', 'tanh', 'softmax')
            backend: Computation backend ('numpy' or 'torch')
        """
        self.backend = Backend.get_backend(backend)
        self.weights = self.backend.initialize_weights(n_inputs, n_neurons)
        self.biases = self.backend.initialize_biases(n_neurons)

        # Set activation function
        if activation is None:
            self.activation = self.backend.linear
        else:
            self.activation = getattr(self.backend, activation.lower())

    def forward(self, inputs):
        """
        Perform the forward pass for the layer.

        Args:
            inputs: Input data

        Returns:
            Layer output after activation
        """
        self.output_raw = self.backend.forward(inputs, self.weights, self.biases)
        self.output = self.activation(self.output_raw)
        return self.output

# Example usage:
# numpy_layer = Layer_Dense(2, 64, activation='relu', backend='numpy')
# torch_layer = Layer_Dense(2, 64, activation='relu', backend='torch')
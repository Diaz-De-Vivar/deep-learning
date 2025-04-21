import numpy as np
import torch
import torch.nn.functional as F

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True # For reproducibility
torch.backends.cudnn.benchmark = False # Benchmarking disabled for reproducibility
torch.cuda.manual_seed(42) # Set seed for GPU if available
# torch.cuda.manual_seed_all(SEED) # lol someday
# torch.backends.cudnn.enabled = False # Disable cuDNN for reproducibility

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

class Optimizer_SGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
        # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

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
        y_pred_clamped = torch.clamp(y_pred, epsilon, 1 - epsilon)
        
        # Check if y_true is one-hot encoded or sparse
        if y_true.ndim == 1:  # Sparse labels
            loss = -torch.log(y_pred_clamped[torch.arange(len(y_pred_clamped)), y_true.long()])
        # elif y_true.ndim == 2:  # One-hot encoded labels
            # loss = -torch.sum(y_true * torch.log(y_pred_clamped), dim=1)
        else:  # One-hot encoded labels
            loss = -torch.sum(y_true * torch.log(y_pred_clamped), dim=1)
        
        return torch.mean(loss, dtype=torch.float32)

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
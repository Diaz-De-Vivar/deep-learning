import os
import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data, vertical_data
import numpy as np
import nnfs
import time
import torch
import torch.nn.functional as F

np.random.seed(42)
nnfs.init()

# Updated Layer_Dense class with activation function support
class Layer_DenseNumpy:
    def __init__(self, n_inputs, n_neurons, activation=None):
        """
        Initialize weights, biases, and activation function for the layer.
        :param n_inputs: Number of inputs to the layer
        :param n_neurons: Number of neurons in the layer
        :param activation: Activation function (e.g., relu, sigmoid, tanh, softmax)
        """
        # NumPy version
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

class Layer_DenseTorch:
    def __init__(self, n_inputs, n_neurons, activation=None):
        """
        Initialize weights, biases, and activation function for the layer.
        :param n_inputs: Number of inputs to the layer
        :param n_neurons: Number of neurons in the layer
        :param activation: Activation function (e.g., relu, sigmoid, tanh, softmax)
        """
        # PyTorch version
        # Check if CUDA is available and set device accordingly
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = 0.01 * torch.randn(n_inputs, n_neurons, device=device)
        self.biases = torch.zeros(1, n_neurons, device=device)
        self.activation = activation

    def forward_prop(self, inputs):
        """
        Perform the forward pass for the layer.
        :param inputs: Input data
        """
        self.output_raw = inputs @ self.weights + self.biases
        self.output = self.activation(self.output_raw) if self.activation else self.output_raw

# Define activation functions as a dictionary for better modularity
# CPU (numpy)
activation_functions = {
    "linear": lambda x: x,
    "relu": lambda x: np.maximum(0, x),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh": lambda x: np.tanh(x),
    "softmax": lambda x: np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)
}

# GPU (PyTorch)
activation_functions_torch = {
    "linear": lambda x: x,
    "relu": lambda x: F.relu(x),
    "sigmoid": lambda x: torch.sigmoid(x),
    "tanh": lambda x: torch.tanh(x),
    "softmax": lambda x: F.softmax(x, dim=1)
}

# Loss Functions
# CPU (numpy) GPU (PyTorch)

def categorical_cross_entropy_torch(y_true, y_pred):
    """
    Calculate categorical cross-entropy loss
    
    Args:
        y_true: Ground truth probabilities or one-hot encoded labels
        y_pred: Predicted probabilities from softmax
        
    Returns:
        Loss value
    """
    # Ensure numerical stability by adding a small epsilon to prevent log(0)
    epsilon = 1e-15
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    
    # Calculate cross-entropy loss
    loss = -torch.sum(y_true * torch.log(y_pred), dim=1)
    
    # Return mean loss across all samples
    return torch.mean(loss)

# Categorical cross-entropy loss function
def categorical_cross_entropy_numpy(y_pred, y_true, epsilon = 1e-15):
    """
    Categorical cross-entropy loss function.
    Args:
        y_pred (np.ndarray): Predicted probabilities (batch_size, num_classes).
        y_true (np.ndarray): True labels (batch_size, num_classes).
        epsilon (float): Small value to prevent log(0).
    Returns:
        np.ndarray: Loss value for each sample in the batch.
    """   
    # Clip predictions to prevent log(0)
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    # Calculate loss
    return -np.sum(y_true * np.log(y_pred_clipped), axis=1, keepdims=True)
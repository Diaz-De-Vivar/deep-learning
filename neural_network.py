import numpy as np
import matplotlib.pyplot as plt

class Layer_Dense:
    """
    Represents a fully connected (dense) neural network layer.
    
    Attributes:
        weights (np.ndarray): Weight matrix of shape (n_inputs, n_neurons).
        biases (np.ndarray): Bias vector of shape (1, n_neurons).
        activation (callable): Optional activation function to apply after the linear transformation.
    """
    def __init__(self, n_inputs, n_neurons, activation=None):
        """
        Initialize weights, biases, and activation function for the layer.

        Args:
            n_inputs (int): Number of input features.
            n_neurons (int): Number of neurons in the layer.
            activation (callable, optional): Activation function (e.g., relu, sigmoid, tanh, softmax).
        Raises:
            ValueError: If n_inputs or n_neurons is not positive.
        """
        if n_inputs <= 0 or n_neurons <= 0:
            raise ValueError("Number of inputs and neurons must be positive")
        # Initialize weights with small random values
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Initialize biases with zeros
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    def forward_prop(self, inputs):
        """
        Perform the forward pass for the layer.

        Args:
            inputs (np.ndarray): Input data of shape (batch_size, n_inputs).
        Sets:
            self.output_raw: Linear output before activation.
            self.output: Output after activation (if any).
        """
        self.output_raw = inputs @ self.weights + self.biases
        self.output = self.activation(self.output_raw) if self.activation else self.output_raw

# --- Below: Functions for multi-layer neural network training (book style) ---
def init_params(layer_dims):
    """
    Initialize parameters (weights and biases) for a multi-layer neural network.

    Args:
        layer_dims (list): List of layer sizes, e.g., [input_dim, hidden1, ..., output_dim].
    Returns:
        dict: Dictionary of parameters 'W1', 'b1', ..., 'WL', 'bL'.
    """
    np.random.seed(3)
    params = {}
    L = len(layer_dims)
    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return params

def sigmoid(Z):
    """
    Sigmoid activation function.
    Args:
        Z (np.ndarray): Input array.
    Returns:
        tuple: (A, cache) where A is the sigmoid output, cache is Z for backprop.
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def forward_prop(X, params):
    """
    Perform forward propagation through all layers.
    Args:
        X (np.ndarray): Input data.
        params (dict): Network parameters.
    Returns:
        tuple: (A, caches) where A is the output, caches for backprop.
    """
    A = X
    caches = []
    L = len(params) // 2
    for l in range(1, L + 1):
        A_prev = A
        Z = np.dot(params['W' + str(l)], A_prev) + params['b' + str(l)]
        A, activation_cache = sigmoid(Z)
        cache = ((A_prev, params['W' + str(l)], params['b' + str(l)]), activation_cache)
        caches.append(cache)
    return A, caches

def cost_function(A, Y):
    """
    Compute the cost (binary cross-entropy loss).
    Args:
        A (np.ndarray): Predictions.
        Y (np.ndarray): True labels.
    Returns:
        float: Cost value.
    """
    m = Y.shape[1]
    cost = (-1/m) * (np.dot(np.log(A), Y.T) + np.dot(np.log(1 - A), (1 - Y).T))
    return cost

def one_layer_backward(dA, cache):
    """
    Backward propagation for a single layer.
    Args:
        dA (np.ndarray): Gradient of the activation.
        cache (tuple): Values from forward pass.
    Returns:
        tuple: Gradients (dA_prev, dW, db).
    """
    linear_cache, activation_cache = cache
    Z = activation_cache
    dZ = dA * sigmoid(Z)[0] * (1 - sigmoid(Z)[0])
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def backprop(AL, Y, caches):
    """
    Perform backward propagation through all layers.
    Args:
        AL (np.ndarray): Final output.
        Y (np.ndarray): True labels.
        caches (list): Caches from forward pass.
    Returns:
        dict: Gradients for all layers.
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L - 1)], grads["db" + str(L - 1)] = one_layer_backward(dAL, current_cache)

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = one_layer_backward(grads["dA" + str(l + 1)], current_cache)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l)] = dW_temp
        grads["db" + str(l)] = db_temp

    return grads

def update_parameters(params, grads, learning_rate):
    """
    Update network parameters using gradient descent.
    Args:
        params (dict): Current parameters.
        grads (dict): Gradients.
        learning_rate (float): Learning rate.
    Returns:
        dict: Updated parameters.
    """
    L = len(params) // 2
    for l in range(L):
        params["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
    return params

def train(X, Y, layer_dims, epochs, lr):
    """
    Train a multi-layer neural network.
    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): True labels.
        layer_dims (list): List of layer sizes.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
    Returns:
        tuple: (params, cost_history)
    """
    params = init_params(layer_dims)
    cost_history = []

    for i in range(epochs):
        Y_hat, caches = forward_prop(X, params)
        cost = cost_function(Y_hat, Y)
        cost_history.append(cost)
        grads = backprop(Y_hat, Y, caches)
        params = update_parameters(params, grads, lr)

    return params, cost_history

if __name__ == "__main__":
    # Example usage: Train a simple neural network on random data
    np.random.seed(42)
    X = np.random.randn(2, 100)  # 2 features, 100 samples
    Y = np.random.randint(0, 2, (1, 100))  # Binary classification

    # Define network architecture
    layer_dims = [2, 3, 1]  # Input layer: 2, Hidden layer: 3, Output layer: 1

    # Train the network
    params, cost_history = train(X, Y, layer_dims, epochs=1000, lr=0.01)

    # Plot the cost history
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Training Cost Over Time')
    plt.grid(True)
    plt.show()

    # Make predictions
    Y_hat, _ = forward_prop(X, params)
    predictions = (Y_hat > 0.5).astype(int)
    accuracy = np.mean(predictions == Y)
    print(f"Training accuracy: {accuracy * 100:.2f}%")
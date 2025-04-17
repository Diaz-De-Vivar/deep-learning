import numpy as np
import pytest
from neural_network import Layer_Dense

def test_layer_initialization():
    """Test proper initialization of neural network layers"""
    n_inputs = 4
    n_neurons = 3
    layer = Layer_Dense(n_inputs, n_neurons)
    
    # Test weights shape
    assert layer.weights.shape == (n_inputs, n_neurons)
    
    # Test biases shape
    assert layer.biases.shape == (1, n_neurons)
    
    # Test weights initialization (should be small random values)
    assert np.all(np.abs(layer.weights) < 1)
    
    # Test biases initialization (should be zeros)
    assert np.all(layer.biases == 0)

def test_forward_propagation():
    """Test forward propagation through a layer"""
    # Create sample input
    X = np.array([[1, 2, 3, 2.5],
                  [2.0, 5.0, -1.0, 2.0]])
    
    # Initialize layer
    layer = Layer_Dense(4, 3)
    layer.weights = np.array([[0.2, 0.8, -0.5],
                            [0.5, -0.91, 0.26],
                            [-0.26, -0.27, 0.17],
                            [0.87, 0.5, -0.4]])
    layer.biases = np.array([[2, 3, 0.5]])
    
    # Perform forward propagation
    layer.forward_prop(X)
    
    # Test output shape
    assert layer.output.shape == (2, 3)
    
    # Test output values (pre-calculated)
    expected_output = X @ layer.weights + layer.biases
    np.testing.assert_array_almost_equal(layer.output, expected_output)

def test_activation_functions():
    """Test different activation functions"""
    x = np.array([-2, -1, 0, 1, 2])
    
    # ReLU activation
    relu = lambda x: np.maximum(0, x)
    np.testing.assert_array_equal(relu(x), np.array([0, 0, 0, 1, 2]))
    
    # Sigmoid activation
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    sigmoid_expected = 1 / (1 + np.exp(-x))
    np.testing.assert_array_almost_equal(sigmoid(x), sigmoid_expected)
    
    # Softmax activation
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    softmax_output = softmax(x)
    assert np.sum(softmax_output) == pytest.approx(1.0)
    assert np.all(softmax_output >= 0)

def test_layer_with_activation():
    """Test layer with activation function"""
    X = np.array([[1, 2, 3, 2.5]])
    layer = Layer_Dense(4, 3, activation=lambda x: np.maximum(0, x))  # ReLU
    
    layer.weights = np.array([[0.2, 0.8, -0.5],
                            [0.5, -0.91, 0.26],
                            [-0.26, -0.27, 0.17],
                            [0.87, 0.5, -0.4]])
    layer.biases = np.array([[2, 3, 0.5]])
    
    layer.forward_prop(X)
    
    # Test that output after activation is non-negative (ReLU property)
    assert np.all(layer.output >= 0)

def test_layer_input_validation():
    """Test input validation for layer initialization"""
    with pytest.raises(ValueError):
        Layer_Dense(0, 1)  # Invalid number of inputs
    
    with pytest.raises(ValueError):
        Layer_Dense(1, 0)  # Invalid number of neurons
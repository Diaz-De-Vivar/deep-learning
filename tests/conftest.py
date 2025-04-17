"""Pytest configuration and fixtures"""
import pytest
import numpy as np

@pytest.fixture
def sample_data():
    """Sample input data for testing"""
    X = np.array([[1, 2, 3, 2.5],
                  [2.0, 5.0, -1.0, 2.0],
                  [-1.5, 2.7, 3.3, -0.8]])
    return X

@pytest.fixture
def sample_weights():
    """Sample weights for testing"""
    return np.array([[0.2, 0.8, -0.5],
                    [0.5, -0.91, 0.26],
                    [-0.26, -0.27, 0.17],
                    [0.87, 0.5, -0.4]])

@pytest.fixture
def sample_biases():
    """Sample biases for testing"""
    return np.array([[2, 3, 0.5]])
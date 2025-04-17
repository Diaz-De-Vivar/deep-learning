import os
import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
import numpy as np
import nnfs
import time
import torch
import torch.nn.functional as F

np.random.seed(42)
nnfs.init()

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
def categorical_cross_entropy_torch_numpy(y_pred, y_true, epsilon = 1e-15):
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
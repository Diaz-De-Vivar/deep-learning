import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import nnfs
from nnfs.datasets import spiral_data

# --- PyTorch Model Definition ---

class NeuralNetwork(nn.Module):
    """PyTorch Neural Network Model."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() # Initialize the parent nn.Module class
        # Define the layers sequentially
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size), # Dense Layer 1 (Handles weights/bias)
            nn.ReLU(),                         # ReLU Activation 1
            nn.Linear(hidden_size, hidden_size),# Dense Layer 2
            nn.ReLU(),                         # ReLU Activation 2
            nn.Linear(hidden_size, output_size) # Output Layer (raw logits)
        )
        # Note: No Softmax here! nn.CrossEntropyLoss combines Softmax and NLLLoss

    def forward(self, x):
        """Defines the forward pass."""
        logits = self.layer_stack(x)
        return logits

# --- Helper Function for Accuracy ---
def calculate_accuracy(y_pred_logits, y_true):
    """Calculates accuracy given logits and true labels."""
    # Get predicted class indices by finding the max logit
    predicted_classes = torch.argmax(y_pred_logits, dim=1)
    # Compare predictions with true labels
    correct = (predicted_classes == y_true).sum().item()
    accuracy = correct / len(y_true)
    return accuracy

# --- Parameters ---
INPUT_SIZE = 2       # Number of features in input data (X has 2 columns)
HIDDEN_SIZE = 64     # Number of neurons in hidden layers
OUTPUT_SIZE = 3      # Number of classes
LEARNING_RATE = 0.02 # Corresponds to Adam LR in NumPy version
# Note: PyTorch Adam's weight_decay is L2 regularization strength * 2 compared to the manual NumPy implementation
# 5e-4 in NumPy -> 1e-3 in PyTorch's weight_decay? Or just use the same value?
# Let's use the exact same value 5e-4 for direct comparison fairness, acknowledging the potential factor of 2 difference
# depending on loss formulation. Often it's tuned empirically anyway.
WEIGHT_DECAY = 5e-4 # L2 Regularization strength
EPOCHS = 10001
DATA_SAMPLES = 1000 # Match NumPy setup
DATA_CLASSES = 3

# --- Data Preparation ---
nnfs.init() # For reproducible dataset generation
X_np, y_np = spiral_data(samples=DATA_SAMPLES, classes=DATA_CLASSES)

# --- PyTorch Training Setup ---

# 1. Device Selection (CPU or GPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("PyTorch using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("PyTorch using CPU")

# 2. Convert Data to PyTorch Tensors and Move to Device
# Inputs need to be FloatTensor, Labels for CrossEntropyLoss should be LongTensor
X_tensor = torch.from_numpy(X_np).float().to(device)
y_tensor = torch.from_numpy(y_np).long().to(device) # Use long for integer class labels

# 3. Instantiate Model and Move to Device
model = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)

# 4. Define Loss Function
# nn.CrossEntropyLoss expects raw logits (output of the last Linear layer)
# It internally applies log_softmax and calculates the Negative Log Likelihood Loss
criterion = nn.CrossEntropyLoss()

# 5. Define Optimizer
# Pass model parameters and L2 regularization strength (weight_decay)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# Note: PyTorch's Adam also handles betas, epsilon, decay internally like the NumPy version

# --- PyTorch Training Loop ---
print("\nStarting PyTorch Training...")
start_time_torch = time.time()

model.train() # Set the model to training mode (important for dropout, batchnorm etc., though not used here)

for epoch in range(EPOCHS):
    # --- Forward Pass ---
    # Model automatically calls the forward() method
    logits = model(X_tensor)

    # --- Calculate Loss ---
    # Loss function compares model's raw logits with true integer labels
    loss = criterion(logits, y_tensor)
    # Note: L2 Regularization is already included via optimizer's weight_decay

    # --- Calculate Accuracy ---
    accuracy = calculate_accuracy(logits, y_tensor)

    # --- Backward Pass and Optimization ---
    # 1. Zero gradients from previous iteration
    optimizer.zero_grad()
    # 2. Calculate gradients
    loss.backward()
    # 3. Update model parameters
    optimizer.step()

    # --- Print Progress ---
    if not epoch % 100:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Acc: {accuracy:.3f}, LR: {optimizer.param_groups[0]["lr"]:.5f}') # Access LR from optimizer

end_time_torch = time.time()
pytorch_training_time = end_time_torch - start_time_torch
print("\nPyTorch Training finished.")
print(f"Total PyTorch training time: {pytorch_training_time:.2f} seconds")

# --- PyTorch Validation (using training data) ---
model.eval() # Set the model to evaluation mode
with torch.no_grad(): # Disable gradient calculation for evaluation
    logits = model(X_tensor)
    final_loss = criterion(logits, y_tensor)
    final_accuracy = calculate_accuracy(logits, y_tensor)

print(f'PyTorch Validation - Acc: {final_accuracy:.3f}, Loss: {final_loss.item():.4f}')

# --- Store PyTorch results for comparison ---
pytorch_final_acc = final_accuracy
pytorch_final_loss = final_loss.item()
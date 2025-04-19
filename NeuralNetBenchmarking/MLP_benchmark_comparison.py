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


# 2. Reflection on Refactoring and GPU Usage

# *   **Code Reduction:** The PyTorch code is significantly shorter and cleaner. We don't need to manually implement:
#     *   Forward pass math for dense layers (handled by `nn.Linear`).
#     *   Forward pass for activations (handled by `nn.ReLU`).
#     *   Backward pass (gradient calculations) for any layer (handled by `torch.autograd`).
#     *   Optimizer logic (handled by `optim.Adam`).
#     *   Combined Softmax/Loss calculation (handled by `nn.CrossEntropyLoss`).
#     *   Manual L2 regularization gradient calculation (handled by `weight_decay` in `optim.Adam`).
# *   **Modularity:** Using `nn.Module` and `nn.Sequential` promotes modularity. It's easy to add, remove, or replace layers. The structure clearly defines the network architecture.
# *   **Readability:** PyTorch code often reads more like a description of the network architecture rather than the underlying mathematical operations, making it easier to understand the high-level structure.
# *   **GPU Acceleration:** Making the code run on a GPU was remarkably simple:
#     1.  Detect GPU availability (`torch.cuda.is_available()`).
#     2.  Define the target `device`.
#     3.  Move the `model` to the device using `.to(device)`.
#     4.  Move the input `X_tensor` and target `y_tensor` to the device using `.to(device)`.
#     PyTorch handles the rest, ensuring computations happen on the specified hardware. This abstraction is a major advantage.
# *   **Potential Pitfalls:**
#     *   **Data Types:** Ensuring tensors have the correct type (`float` for inputs/weights, `long` for `CrossEntropyLoss` targets) is crucial.
#     *   **Device Consistency:** All tensors involved in a computation *must* be on the same device. Forgetting to move data to the GPU alongside the model is a common error.
#     *   **Logits vs. Probabilities:** Understanding that `nn.CrossEntropyLoss` expects raw logits is important. Applying `nn.Softmax()` *before* this loss function is usually incorrect and can lead to training instability or wrong results.

# **3. Benchmarking**

# Now, let's add the NumPy code (slightly modified for timing) and run both.

# ```python
# ========================================================
# NumPy Implementation (from previous response, add timing)
# ========================================================
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import time # Make sure time is imported for NumPy part too

# Initialize nnfs for reproducible results (NumPy version)
# Using a different seed or just re-running might yield slightly different
# initial weights/data shuffles, but nnfs tries to keep data consistent.
nnfs.init()

# --- Layer Definitions (NumPy) ---
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        self.weight_momentums = None
        self.weight_cache = None
        self.bias_momentums = None
        self.bias_cache = None
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights); dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases); dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def regularization_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss
    def remember_trainable_layers(self, trainable_layers):
         self.trainable_layers = trainable_layers
    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        if not include_regularization:
             return data_loss
        return data_loss, self.regularization_loss()

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        samples = len(dvalues); labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true) # Only data loss here
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):
        if layer.weight_momentums is None:
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    def post_update_params(self):
        self.iterations += 1

# --- NumPy Network Assembly and Training ---
print("\nStarting NumPy Training...")
start_time_numpy = time.time()

# Use the same data generated earlier
X, y = X_np, y_np

# Create Network Layers (NumPy) - Use parameters consistent with PyTorch
dense1 = Layer_Dense(INPUT_SIZE, HIDDEN_SIZE, weight_regularizer_l2=WEIGHT_DECAY, bias_regularizer_l2=WEIGHT_DECAY)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(HIDDEN_SIZE, HIDDEN_SIZE) # Keep consistent, no reg on hidden
activation2 = Activation_ReLU()
dense3 = Layer_Dense(HIDDEN_SIZE, OUTPUT_SIZE)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=LEARNING_RATE, decay=0) # Match PyTorch decay (handled by Adam)
loss_calculator = Loss() # For regularization calculation
loss_calculator.remember_trainable_layers([dense1, dense2, dense3])

# --- NumPy Training Loop ---
for epoch in range(EPOCHS):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    data_loss = loss_activation.forward(dense3.output, y)
    regularization_loss = loss_calculator.regularization_loss()
    total_loss = data_loss + regularization_loss

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2: y_labels = np.argmax(y, axis=1)
    else: y_labels = y
    accuracy = np.mean(predictions == y_labels)

    if not epoch % 100:
        print(f'Epoch: {epoch}, Acc: {accuracy:.3f}, Loss: {total_loss:.4f} (Data: {data_loss:.4f}, Reg: {regularization_loss:.4f}), LR: {optimizer.current_learning_rate:.5f}')

    loss_activation.backward(loss_activation.output, y)
    dense3.backward(loss_activation.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

end_time_numpy = time.time()
numpy_training_time = end_time_numpy - start_time_numpy
print("\nNumPy Training finished.")
print(f"Total NumPy training time: {numpy_training_time:.2f} seconds")

# --- NumPy Validation ---
dense1.forward(X); activation1.forward(dense1.output)
dense2.forward(activation1.output); activation2.forward(dense2.output)
dense3.forward(activation2.output)
final_data_loss = loss_activation.forward(dense3.output, y)
final_reg_loss = loss_calculator.regularization_loss()
final_total_loss_np = final_data_loss + final_reg_loss
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2: y_labels = np.argmax(y, axis=1)
else: y_labels = y
final_accuracy_np = np.mean(predictions == y_labels)

print(f'NumPy Validation - Acc: {final_accuracy_np:.3f}, Loss: {final_total_loss_np:.4f} (Data: {final_data_loss:.4f}, Reg: {final_reg_loss:.4f})')

# --- Final Comparison ---
print("\n--- BENCHMARK RESULTS ---")
print(f"NumPy Final Accuracy: {final_accuracy_np:.4f}")
print(f"NumPy Final Loss:   {final_total_loss_np:.4f}")
print(f"NumPy Training Time:{numpy_training_time:.2f} seconds")
print("-" * 20)
print(f"PyTorch ({device}) Final Accuracy: {pytorch_final_acc:.4f}")
print(f"PyTorch ({device}) Final Loss:   {pytorch_final_loss:.4f}") # Note: PyTorch loss includes reg implicitly if using weight_decay
print(f"PyTorch ({device}) Training Time:{pytorch_training_time:.2f} seconds")
print("-" * 20)
if pytorch_training_time > 0: # Avoid division by zero
    speedup_factor = numpy_training_time / pytorch_training_time
    print(f"PyTorch ({device}) was ~{speedup_factor:.2f}x faster than NumPy.")
else:
    print("PyTorch training was extremely fast (or time was zero).")
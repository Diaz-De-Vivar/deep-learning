# >> --- BENCHMARK RESULTS ---
# >> NumPy Final Accuracy: 0.9277
# >> NumPy Final Loss:   0.1755
# >> NumPy Training Time (CPU):259.71 seconds
# >> --------------------
# >> PyTorch (cuda) Final Accuracy: 0.9217
# >> PyTorch (cuda) Final Loss:   0.2124
# >> PyTorch (cuda) Training Time (GPU):26.34 seconds
# >> --------------------
# >> PyTorch (cuda) was ~9.86x faster than NumPy.

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Initialize nnfs for reproducible results
nnfs.init()

# --- Layer Definitions ---

class Layer_Dense:
    """Fully connected layer."""
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        # Using Glorot/Xavier initialization implicitly via nnfs scale factor
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        # Placeholders for optimizer's adaptive parameters (like Adam)
        self.weight_momentums = None
        self.weight_cache = None
        self.bias_momentums = None
        self.bias_cache = None

    def forward(self, inputs):
        """Calculate forward pass output."""
        self.inputs = inputs # Store inputs for backward pass
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        """Calculate gradients for backward pass."""
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values passed to this layer
        self.dinputs = np.dot(dvalues, self.weights.T)


# --- Activation Function Definitions ---

class Activation_ReLU:
    """Rectified Linear Unit activation function."""
    def forward(self, inputs):
        self.inputs = inputs # Store inputs for backward pass
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        """Calculate gradients for backward pass."""
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    """Softmax activation function."""
    def forward(self, inputs):
        # Get unnormalized probabilities (subtract max for numerical stability)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        # Store output for backward pass (needed for combined loss backward)
        self.inputs = inputs # Store inputs for potential separate backward pass if needed

    def backward(self, dvalues):
        """Calculate gradients for backward pass."""
        # Create uninitialized array for gradients
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)


# --- Loss Function Definition ---

class Loss:
    """Base class for loss calculation."""
    def calculate(self, output, y, *, include_regularization=False):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated regularization loss if requested
        if not include_regularization:
             return data_loss

        return data_loss, self.regularization_loss()

    # Regularization loss calculation
    def regularization_loss(self):
        # Default value
        regularization_loss = 0
        # Calculate regularization loss for all trainable layers
        for layer in self.trainable_layers:
            # L1 regularization - weights
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                       np.sum(np.abs(layer.weights))
            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                       np.sum(layer.weights * layer.weights)
            # L1 regularization - biases
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                       np.sum(np.abs(layer.biases))
            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                       np.sum(layer.biases * layer.biases)

        return regularization_loss

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
         self.trainable_layers = trainable_layers


class Loss_CategoricalCrossentropy(Loss):
    """Categorical Cross-Entropy loss."""
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1: # Sparse labels (e.g., [0, 1, 1, 2])
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2: # One-hot encoded labels
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Calculate losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        """Calculate gradients for backward pass."""
        samples = len(dvalues)
        labels = len(dvalues[0]) # Number of labels in every sample

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# --- Combined Softmax + Loss Class ---

class Activation_Softmax_Loss_CategoricalCrossentropy():
    """
    Combines Softmax activation and CCE loss for faster backward step.
    """
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        """Calculate gradients (simplified combined)."""
        samples = len(dvalues)

        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we don't modify the original variable
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# --- Optimizer Definitions ---

class Optimizer_Adam:
    """Adam optimizer."""
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        """Called before parameter updates."""
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """Update layer's weights and biases."""
        # Initialize momentums and cache if needed
        if layer.weight_momentums is None:
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
            (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        # self.iterations is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                         bias_momentums_corrected / \
                         (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        """Called after parameter updates."""
        self.iterations += 1

# --- Network Assembly and Training ---

# Create dataset
X, y = spiral_data(samples=1000, classes=3) # Use more samples for better training

# --- Create Network Layers ---
# Layer 1: Dense layer with 2 input features (X data) and 64 output neurons
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()

# Layer 2: Dense layer with 64 input neurons (from previous layer) and 64 output neurons
dense2 = Layer_Dense(64, 64) # No regularization needed typically for hidden layers unless overfitting
activation2 = Activation_ReLU()

# Layer 3: Output Dense layer with 64 input neurons and 3 output neurons (for 3 classes)
dense3 = Layer_Dense(64, 3)

# --- Create Loss and Activation for Output Layer ---
# Using the combined class for efficiency and stability
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# --- Create Optimizer ---
optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-5) # Adjusted LR and decay

# --- Prepare Loss for Regularization ---
# Although CCE loss is handled by loss_activation, we need the base Loss object
# to calculate the regularization penalty across all layers.
loss_calculator = Loss()
loss_calculator.remember_trainable_layers([dense1, dense2, dense3]) # Tell loss about layers with params

# --- Training Loop ---
epochs = 10001 # Increase epochs for potentially better convergence

print("Starting Training...")
for epoch in range(epochs):

    # --- Forward Pass ---
    # Layer 1
    dense1.forward(X)
    activation1.forward(dense1.output)
    # Layer 2
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    # Layer 3 (Output Layer)
    dense3.forward(activation2.output)
    # Calculate data loss using combined activation/loss forward pass
    data_loss = loss_activation.forward(dense3.output, y)

    # --- Calculate Regularization Loss ---
    regularization_loss = loss_calculator.regularization_loss()

    # --- Calculate Total Loss ---
    total_loss = data_loss + regularization_loss

    # --- Calculate Accuracy ---
    # Get predictions from the output of the combined activation/loss object
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2: # If y is one-hot encoded
        y_labels = np.argmax(y, axis=1)
    else: # If y is sparse
        y_labels = y
    accuracy = np.mean(predictions == y_labels)

    # --- Print Progress ---
    if not epoch % 100: # Print every 100 epochs
        print(f'Epoch: {epoch}, ' +
              f'Acc: {accuracy:.3f}, ' +
              f'Loss: {total_loss:.3f} (' +
              f'Data: {data_loss:.3f}, ' +
              f'Reg: {regularization_loss:.3f}), ' +
              f'LR: {optimizer.current_learning_rate:.5f}')

    # --- Backward Pass ---
    # Start with the combined activation/loss backward pass
    loss_activation.backward(loss_activation.output, y)
    # Propagate gradients backward through the network
    dense3.backward(loss_activation.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # --- Update Parameters ---
    optimizer.pre_update_params() # Update learning rate if decay is used
    optimizer.update_params(dense1) # Update weights/biases for layer 1
    optimizer.update_params(dense2) # Update weights/biases for layer 2
    optimizer.update_params(dense3) # Update weights/biases for layer 3
    optimizer.post_update_params() # Increment optimizer iteration count

print("\nTraining finished.")

# --- Validate the model (optional, using the same training data here) ---
print("Validating on training data...")
# Perform a forward pass with the final trained parameters
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
# Calculate final loss and accuracy
final_loss = loss_activation.forward(dense3.output, y)
final_regularization_loss = loss_calculator.regularization_loss()
final_total_loss = final_loss + final_regularization_loss

predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y_labels = np.argmax(y, axis=1)
else:
    y_labels = y
final_accuracy = np.mean(predictions == y_labels)

print(f'Validation - Acc: {final_accuracy:.3f}, Loss: {final_total_loss:.3f} (Data: {final_loss:.3f}, Reg: {final_regularization_loss:.3f})')
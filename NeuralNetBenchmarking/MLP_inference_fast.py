import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import nnfs
from nnfs.datasets import spiral_data
import os
from sklearn.model_selection import train_test_split # For splitting data

# --- Constants ---
INPUT_SIZE = 2
HIDDEN_SIZE = 64
OUTPUT_SIZE = 3
LEARNING_RATE = 0.02
WEIGHT_DECAY = 5e-4 # L2 Regularization
EPOCHS = 10001
DATA_SAMPLES = 5000 # Increase samples for a more meaningful validation split
DATA_CLASSES = 3
VALIDATION_SPLIT = 0.2 # Use 20% of data for validation
RANDOM_STATE = 42 # For reproducible data split

# --- File Paths ---
STATE_DICT_SAVE_PATH = "spiral_model_pytorch_state.pth" # For retraining/inspection
TORCHSCRIPT_SAVE_PATH = "spiral_model_pytorch_script.pt" # For optimized inference


# --- PyTorch Model Definition (Needed for training and TorchScript creation) ---
class NeuralNetwork(nn.Module):
    """PyTorch Neural Network Model."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        logits = self.layer_stack(x)
        return logits

# --- Helper Function for Accuracy ---
def calculate_accuracy(y_pred_logits, y_true):
    predicted_classes = torch.argmax(y_pred_logits, dim=1)
    correct = (predicted_classes == y_true).sum().item()
    accuracy = correct / len(y_true)
    return accuracy

# --- Training Function ---
def train_model(device, retrain=False):
    """Trains the model, evaluates on validation set, and saves state dict."""

    # --- Check if retraining is needed ---
    if not retrain and os.path.exists(STATE_DICT_SAVE_PATH):
        print(f"Model state dict '{STATE_DICT_SAVE_PATH}' already exists. Skipping training.")
        print("Set retrain=True to force retraining.")
        # Load the existing model state to return it for potential immediate use/evaluation
        model = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        try:
            model.load_state_dict(torch.load(STATE_DICT_SAVE_PATH, map_location=device))
            model.to(device)
            print("Loaded existing model state dict.")
            # We still need validation data if we skipped training but want evaluation
            # Generate full dataset
            nnfs.init()
            X_np_full, y_np_full = spiral_data(samples=DATA_SAMPLES, classes=DATA_CLASSES)
             # Split data
            X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
                X_np_full, y_np_full, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=y_np_full
            )
            # Convert validation data to tensors
            X_val = torch.from_numpy(X_val_np).float().to(device)
            y_val = torch.from_numpy(y_val_np).long().to(device)
            return model, X_val, y_val # Return loaded model and val data
        except Exception as e:
            print(f"Warning: Could not load existing state dict ({e}). Proceeding with retraining.")
            # Fall through to training if loading fails

    print("--- Starting Model Training ---")
    start_time_train = time.time()

    # --- Data Preparation ---
    nnfs.init() # Ensure consistent data if retraining
    X_np_full, y_np_full = spiral_data(samples=DATA_SAMPLES, classes=DATA_CLASSES)

    # Split into training and validation sets
    # stratify=y ensures similar class distribution in train/val sets
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        X_np_full, y_np_full, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=y_np_full
    )
    print(f"Training samples: {len(X_train_np)}, Validation samples: {len(X_val_np)}")

    # Convert to PyTorch Tensors and Move to Device
    X_train = torch.from_numpy(X_train_np).float().to(device)
    y_train = torch.from_numpy(y_train_np).long().to(device)
    X_val = torch.from_numpy(X_val_np).float().to(device)
    y_val = torch.from_numpy(y_val_np).long().to(device)

    # --- Model, Loss, Optimizer ---
    model = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Training Loop ---
    model.train() # Set model to training mode
    for epoch in range(EPOCHS):
        # Training Step
        logits = model(X_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Optional: Periodic Validation ---
        if not epoch % 500:
            model.eval() # Switch to evaluation mode for validation
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = criterion(val_logits, y_val)
                val_accuracy = calculate_accuracy(val_logits, y_val)
            model.train() # Switch back to training mode
            train_accuracy = calculate_accuracy(logits, y_train) # Accuracy on current train batch
            print(f'Epoch: {epoch}, Train Loss: {loss.item():.4f}, Train Acc: {train_accuracy:.3f} | Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.3f}')

    end_time_train = time.time()
    print(f"\nTraining finished in {end_time_train - start_time_train:.2f} seconds.")

    # --- Save the Model State Dictionary ---
    print(f"Saving model state dictionary to: {STATE_DICT_SAVE_PATH}")
    torch.save(model.state_dict(), STATE_DICT_SAVE_PATH)
    print("State dict saved successfully.")

    return model, X_val, y_val # Return trained model and validation data

# --- Evaluation Function ---
def evaluate_model(model, X_val, y_val, criterion, device):
    """Evaluates the model on the validation set."""
    print("\n--- Evaluating Model on Validation Set ---")
    model.eval() # Set model to evaluation mode
    val_loss_list = []
    val_acc_list = []

    with torch.no_grad(): # Disable gradient calculations
        # In a real scenario with large validation set, you might batch this
        val_logits = model(X_val)
        val_loss = criterion(val_logits, y_val)
        val_accuracy = calculate_accuracy(val_logits, y_val)
        val_loss_list.append(val_loss.item())
        val_acc_list.append(val_accuracy)

    avg_val_loss = sum(val_loss_list) / len(val_loss_list)
    avg_val_acc = sum(val_acc_list) / len(val_acc_list)

    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {avg_val_acc:.4f}")
    return avg_val_loss, avg_val_acc

# --- TorchScript Saving Function ---
def save_torchscript_model(model, device):
    """Converts the model to TorchScript and saves it."""
    print(f"\n--- Saving TorchScript Model to: {TORCHSCRIPT_SAVE_PATH} ---")
    model.eval() # Ensure model is in eval mode before tracing

    # Create dummy input with the correct shape and type matching model input
    # The batch size (first dimension) can often be 1 for tracing
    dummy_input = torch.randn(1, INPUT_SIZE, device=device).float()

    try:
        # Trace the model: Run the dummy input through the model and record operations
        traced_script_module = torch.jit.trace(model, dummy_input)
        # Save the traced model
        traced_script_module.save(TORCHSCRIPT_SAVE_PATH)
        print("TorchScript model saved successfully.")
        return traced_script_module
    except Exception as e:
        print(f"Error saving TorchScript model: {e}")
        return None

# --- TorchScript Loading Function ---
def load_torchscript_model(device):
    """Loads the TorchScript model from file."""
    print(f"\n--- Loading TorchScript Model from: {TORCHSCRIPT_SAVE_PATH} ---")
    if not os.path.exists(TORCHSCRIPT_SAVE_PATH):
        print(f"Error: TorchScript model file not found at {TORCHSCRIPT_SAVE_PATH}")
        print("Ensure you have trained and saved the TorchScript model first.")
        return None
    try:
        # Load the TorchScript model, mapping it to the correct device
        scripted_model = torch.jit.load(TORCHSCRIPT_SAVE_PATH, map_location=device)
        scripted_model.eval() # Ensure it's in eval mode after loading
        print("TorchScript model loaded successfully.")
        return scripted_model
    except Exception as e:
        print(f"Error loading TorchScript model: {e}")
        return None

# --- Optimized Inference Function (using TorchScript) ---
def perform_inference_optimized(scripted_model, input_data_np, device):
    """
    Performs inference using the loaded TorchScript model.

    Args:
        scripted_model: The loaded TorchScript model object.
        input_data_np (np.ndarray): NumPy array of input samples.
        device (torch.device): The device to run inference on (CPU or GPU).

    Returns:
        tuple: predicted_classes (np.ndarray), probabilities (np.ndarray)
    """
    if scripted_model is None:
        print("Error: TorchScript model not loaded. Cannot perform inference.")
        return None, None

    print("\n--- Performing Optimized Inference (TorchScript) ---")
    # No need for model.eval() or torch.no_grad() wrapper,
    # TorchScript module handles this execution context.

    # Prepare input data
    if not isinstance(input_data_np, np.ndarray):
         input_data_np = np.array(input_data_np, dtype=np.float32)
    input_tensor = torch.from_numpy(input_data_np).float().to(device)
    if input_tensor.ndim == 1:
         input_tensor = input_tensor.unsqueeze(0)

    # Forward pass using the TorchScript model
    logits = scripted_model(input_tensor) # Directly call the loaded object

    # Get probabilities and classes
    probabilities_tensor = torch.softmax(logits, dim=1)
    predicted_classes_tensor = torch.argmax(probabilities_tensor, dim=1)

    # Convert results back to NumPy
    predicted_classes = predicted_classes_tensor.cpu().detach().numpy()
    probabilities = probabilities_tensor.cpu().detach().numpy()

    print("Optimized inference complete.")
    return predicted_classes, probabilities


# --- Main Execution ---
if __name__ == "__main__":

    # 1. Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 2. Train the Model (or load if exists) & Get Validation Data
    # Set retrain=True if you want to force retraining
    trained_model, X_val_tensor, y_val_tensor = train_model(device, retrain=False)

    # 3. Evaluate the Trained Model on Validation Set
    if trained_model:
        criterion = nn.CrossEntropyLoss() # Need loss function for evaluation too
        evaluate_model(trained_model, X_val_tensor, y_val_tensor, criterion, device)
    else:
        print("\nSkipping evaluation as model was not trained or loaded.")


    # 4. Save the Model as TorchScript (for optimized inference)
    if trained_model:
        save_torchscript_model(trained_model, device)
    else:
        print("\nSkipping TorchScript saving as model was not available.")


    # 5. Load the TorchScript Model for Inference
    inference_model_scripted = load_torchscript_model(device)


    # 6. Perform Optimized Inference
    if inference_model_scripted:
        # Prepare Sample Data for Inference
        new_data = np.array([
            [0, 0],
            [0.5, 0.5],
            [-0.5, 0.5],
            [0.8, -0.8],
            [-1.0, -0.2],
            [0.1, -0.9] # Add another point
        ], dtype=np.float32)

        print("\nSample input data for inference:")
        print(new_data)

        # Run Optimized Inference
        start_infer = time.time()
        predicted_classes, probabilities = perform_inference_optimized(
            inference_model_scripted, new_data, device
        )
        end_infer = time.time()


        # Display Inference Results
        if predicted_classes is not None:
            print(f"\nOptimized Inference Time: {(end_infer - start_infer)*1000:.4f} ms")
            print("\nInference Results (Optimized):")
            for i in range(len(new_data)):
                print(f"  Input: {new_data[i]}")
                print(f"  Predicted Class: {predicted_classes[i]}")
                print(f"  Probabilities: {probabilities[i].round(3)}")
                print("---")
    else:
        print("\nCould not proceed with optimized inference as TorchScript model failed to load.")
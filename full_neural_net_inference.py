import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import nnfs
from nnfs.datasets import spiral_data
import os # To check for existing model file

# --- Constants ---
INPUT_SIZE = 2
HIDDEN_SIZE = 64
OUTPUT_SIZE = 3
LEARNING_RATE = 0.02
WEIGHT_DECAY = 5e-4
EPOCHS = 10001 # Keep epochs consistent for comparison if rerunning training
DATA_SAMPLES = 1000
DATA_CLASSES = 3
MODEL_SAVE_PATH = "spiral_model_pytorch.pth" # File to save/load model state


# --- PyTorch Model Definition (Needs to be available for both training and loading) ---
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
    """Calculates accuracy given logits and true labels."""
    predicted_classes = torch.argmax(y_pred_logits, dim=1)
    correct = (predicted_classes == y_true).sum().item()
    accuracy = correct / len(y_true)
    return accuracy

# --- Training Function ---
def train_model(device, retrain=False):
    """Trains the model and saves its state dictionary."""
    if not retrain and os.path.exists(MODEL_SAVE_PATH):
        print(f"Model file '{MODEL_SAVE_PATH}' already exists. Skipping training.")
        print("Set retrain=True to force retraining.")
        # Optionally load and return the existing model here if needed immediately
        # model = load_model(device)
        # return model
        return None # Indicate training was skipped

    print("--- Starting Model Training ---")
    start_time_train = time.time()

    # Data Preparation
    nnfs.init() # Ensure consistent data if retraining
    X_np, y_np = spiral_data(samples=DATA_SAMPLES, classes=DATA_CLASSES)
    X_tensor = torch.from_numpy(X_np).float().to(device)
    y_tensor = torch.from_numpy(y_np).long().to(device)

    # Model, Loss, Optimizer
    model = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)
        accuracy = calculate_accuracy(logits, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not epoch % 500: # Print less frequently during longer training
             print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Acc: {accuracy:.3f}')

    end_time_train = time.time()
    print(f"\nTraining finished in {end_time_train - start_time_train:.2f} seconds.")

    # --- Save the Model State ---
    print(f"Saving model state dictionary to: {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully.")

    return model # Return the trained model instance

# --- Loading Function ---
def load_model(device):
    """Loads the model state dictionary from the file."""
    print(f"\n--- Loading Model for Inference from: {MODEL_SAVE_PATH} ---")
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}")
        print("Train the model first using train_model()")
        return None

    # Instantiate model with the SAME architecture used during training
    model = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    # Load the state dictionary
    # map_location=device ensures tensors are loaded onto the correct device directly
    try:
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model state: {e}")
        print("Ensure the model architecture in the script matches the saved state.")
        return None


    # Move model to the target device (redundant if map_location worked, but good practice)
    model.to(device)

    print("Model loaded successfully.")
    return model

# --- Inference Function ---
def perform_inference(model, input_data_np, device):
    """
    Performs inference on new data using the loaded model.

    Args:
        model (nn.Module): The loaded PyTorch model.
        input_data_np (np.ndarray): NumPy array of input samples.
        device (torch.device): The device to run inference on (CPU or GPU).

    Returns:
        tuple: A tuple containing:
            - predicted_classes (np.ndarray): The predicted class index for each sample.
            - probabilities (np.ndarray): The probability distribution over classes for each sample.
    """
    if model is None:
        print("Error: Model not loaded. Cannot perform inference.")
        return None, None

    print("\n--- Performing Inference ---")
    # 1. Set model to evaluation mode
    model.eval()

    # 2. Prepare input data
    # Ensure input is a NumPy array first if it's not already
    if not isinstance(input_data_np, np.ndarray):
         input_data_np = np.array(input_data_np, dtype=np.float32)
    # Convert to PyTorch tensor and move to device
    input_tensor = torch.from_numpy(input_data_np).float().to(device)

    # Ensure tensor has the correct shape (e.g., [n_samples, n_features])
    if input_tensor.ndim == 1:
         input_tensor = input_tensor.unsqueeze(0) # Add batch dimension if single sample

    # 3. Disable gradient calculation
    with torch.no_grad():
        # 4. Forward pass
        logits = model(input_tensor)

        # 5. Get probabilities (apply Softmax)
        probabilities_tensor = torch.softmax(logits, dim=1)

        # 6. Get predicted classes (index with highest probability)
        predicted_classes_tensor = torch.argmax(probabilities_tensor, dim=1)

    # 7. Convert results back to NumPy arrays (move to CPU first)
    predicted_classes = predicted_classes_tensor.cpu().numpy()
    probabilities = probabilities_tensor.cpu().numpy()

    print("Inference complete.")
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

    # 2. Train the Model (or skip if already exists)
    # Set retrain=True if you want to force retraining even if a file exists
    trained_model = train_model(device, retrain=False)

    # 3. Load the Model for Inference
    # We load it even if we just trained it to demonstrate the loading process
    inference_model = load_model(device)

    # Check if model loading was successful
    if inference_model:
        # 4. Prepare Sample Data for Inference
        # Create a few new data points (e.g., near the origin, and potentially in class boundaries)
        # Shape should be (n_samples, n_features) = (n_samples, 2)
        new_data = np.array([
            [0, 0],
            [0.5, 0.5],
            [-0.5, 0.5],
            [0.8, -0.8],
            [-1.0, -0.2]
        ], dtype=np.float32)

        print("\nSample input data for inference:")
        print(new_data)

        # 5. Run Inference
        start_infer = time.time()
        predicted_classes, probabilities = perform_inference(inference_model, new_data, device)
        end_infer = time.time()

        # 6. Display Inference Results
        if predicted_classes is not None:
            print(f"\Regular Inference Time: {(end_infer - start_infer)*1000:.4f} ms")
            print("\nInference Results (regular):")
            for i in range(len(new_data)):
                print(f"  Input: {new_data[i]}")
                print(f"  Predicted Class: {predicted_classes[i]}")
                print(f"  Probabilities: {probabilities[i].round(3)}") # Round for display
                print("---")
    else:
        print("\nCould not proceed with inference as model failed to load.")
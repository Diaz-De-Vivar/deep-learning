import numpy as np
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import nnfs
from nnfs.datasets import spiral_data
import os
from sklearn.model_selection import train_test_split

# Import Blitz Bayes components
try:
    import blitz.bayes_layers as bnn
    import blitz.losses as blitz_losses
    import blitz.modules as bmodules # Contains Bayesian modules wrapping others
    BLITZ_AVAILABLE = True
except ImportError:
    print("Blitz Bayesian PyTorch library not found.")
    print("pip install blitz-bayesian-pytorch")
    BLITZ_AVAILABLE = False
    # Define dummy classes/functions if blitz is not available to avoid crashing script
    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc=nn.Linear(1,1)
        def forward(self, x): return self.fc(torch.zeros(x.shape[0],1, device=x.device))
    bnn = type('obj', (object,), {'BayesLinear': DummyModule})
    blitz_losses = type('obj', (object,), {'elbo_loss': lambda *args, **kwargs: torch.tensor(0.0)})


# --- Constants ---
# Data & Training Params
DATA_SAMPLES = 2000
DATA_CLASSES = 3
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
EPOCHS = 4001 # BNNs might need different #epochs
BATCH_SIZE = 128
LEARNING_RATE = 0.005 # Adjust LR if needed

# BNN Specific Params
# Factor to balance KL divergence term in ELBO loss
# See blitz docs; common values are 1/num_batches or fixed small values
KL_WEIGHTING = 1 / (DATA_SAMPLES * (1 - VALIDATION_SPLIT) / BATCH_SIZE)
INFERENCE_SAMPLES = 100 # Number of samples for prediction uncertainty estimation

# --- File Paths ---
BNN_MODEL_SAVE_PATH = "spiral_bnn_model.pth"

# --- Bayesian Neural Network Model ---
class BayesianMLP(bmodules.BayesianModule): # Inherit from BayesianModule for ELBO loss helper
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Use Bayesian Linear layers instead of nn.Linear
        self.blinear1 = bnn.BayesLinear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.blinear2 = bnn.BayesLinear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.blinear3 = bnn.BayesLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu1(self.blinear1(x))
        x = self.relu2(self.blinear2(x))
        x = self.blinear3(x)
        return x

# --- Helper Function for Accuracy (Mean Prediction) ---
def calculate_accuracy_bnn_mean(model, X_data, y_true, device):
    """Calculates accuracy based on the mean prediction of the BNN."""
    model.eval() # Use mean weights for a single pass evaluation
    with torch.no_grad():
        logits = model(X_data.to(device))
        predicted_classes = torch.argmax(logits, dim=1)
        correct = (predicted_classes == y_true.to(device)).sum().item()
        accuracy = correct / len(y_true)
    return accuracy


# --- Training and Evaluation Function ---
def run_bnn_experiment(device, retrain=False):
    """Trains/Evaluates the Bayesian MLP model."""
    if not BLITZ_AVAILABLE:
        print("Cannot run BNN experiment without blitz library.")
        return None, None, None

    # --- Check if retraining is needed ---
    if not retrain and os.path.exists(BNN_MODEL_SAVE_PATH):
        print(f"Model file '{BNN_MODEL_SAVE_PATH}' exists. Skipping training.")
        # Optionally load and return model here
        model = BayesianMLP(input_dim=2, hidden_dim=64, output_dim=DATA_CLASSES)
        model.load_state_dict(torch.load(BNN_MODEL_SAVE_PATH, map_location=device))
        model.to(device)
        print("Loaded existing BNN model.")
         # Still need validation data if we skipped training but want evaluation
        nnfs.init(); X_np_full, y_np_full = spiral_data(DATA_SAMPLES, DATA_CLASSES)
        _, X_val_np, _, y_val_np = train_test_split(X_np_full, y_np_full, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=y_np_full)
        X_val = torch.from_numpy(X_val_np).float().to(device)
        y_val = torch.from_numpy(y_val_np).long().to(device)
        return model, X_val, y_val # Return loaded model and validation data

    print("--- Starting Bayesian Neural Network Experiment ---")

    # --- Data Preparation ---
    nnfs.init()
    X_np_full, y_np_full = spiral_data(samples=DATA_SAMPLES, classes=DATA_CLASSES)
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        X_np_full, y_np_full, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=y_np_full
    )
    print(f"Training samples: {len(X_train_np)}, Validation samples: {len(X_val_np)}")

    # Convert to Tensors
    X_train = torch.from_numpy(X_train_np).float()
    y_train = torch.from_numpy(y_train_np).long()
    X_val = torch.from_numpy(X_val_np).float().to(device)
    y_val = torch.from_numpy(y_val_np).long().to(device)

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Model, Loss, Optimizer ---
    model = BayesianMLP(input_dim=2, hidden_dim=64, output_dim=DATA_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss() # Standard likelihood loss
    # ELBO Loss combines likelihood (criterion) and KL divergence
    # It samples weights multiple times per call (sample_nbr) to estimate the expected log likelihood
    elbo = blitz_losses.elbo_loss(model=model,
                                  criterion=criterion,
                                  sample_nbr=3, # Number of samples per forward pass during training
                                  complexity_cost_weight=KL_WEIGHTING) # Weight for KL term

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("Starting Training...")
    start_time_train = time.time()
    for epoch in range(EPOCHS):
        model.train() # Keep model in training mode for weight sampling
        epoch_loss = 0.0
        batch_count = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            # Calculate ELBO loss (includes likelihood + KL divergence)
            loss = elbo(batch_X, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_epoch_loss = epoch_loss / batch_count

        # --- Periodic Validation ---
        if not epoch % 200:
            # Calculate accuracy using mean weights (single pass)
            val_accuracy = calculate_accuracy_bnn_mean(model, X_val, y_val, device)
            # Calculate validation loss (ELBO) - needs sampling
            model.train() # Temporarily switch to train for sampling in ELBO calculation
            with torch.no_grad():
                 val_loss = elbo(X_val, y_val).item()
            model.eval() # Switch back to eval
            print(f'Epoch: {epoch}, Train ELBO Loss: {avg_epoch_loss:.4f} | Val ELBO Loss: {val_loss:.4f}, Val Acc (Mean): {val_accuracy:.3f}')


    end_time_train = time.time()
    print(f"\nTraining finished in {end_time_train - start_time_train:.2f} seconds.")

    # --- Final Evaluation (Mean Prediction) ---
    print("\n--- Evaluating Final Model on Validation Set (Mean Prediction) ---")
    final_val_acc = calculate_accuracy_bnn_mean(model, X_val, y_val, device)
    model.train()
    with torch.no_grad(): final_val_loss = elbo(X_val, y_val).item()
    model.eval()
    print(f"Final Validation ELBO Loss: {final_val_loss:.4f}")
    print(f"Final Validation Accuracy (Mean): {final_val_acc:.4f}")

    # --- Save the Final Model ---
    print(f"Saving trained model to: {BNN_MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), BNN_MODEL_SAVE_PATH)
    print("Model saved successfully.")

    return model, X_val, y_val

# --- Inference Function with Uncertainty Quantification ---
def predict_with_uncertainty(model, X_input_np, n_samples, device):
    """Performs inference using multiple weight samples to estimate uncertainty."""
    if not BLITZ_AVAILABLE: return None, None, None
    print(f"\n--- Performing BNN Inference (Samples: {n_samples}) ---")

    model.eval() # Keep in eval mode (disables dropout etc.), Bayes layers should still sample if called repeatedly or configured.
                 # If sampling doesn't happen automatically in eval, might need model.train() here. Let's assume eval works for sampling.

    input_tensor = torch.from_numpy(X_input_np).float().to(device)
    if input_tensor.ndim == 1:
        input_tensor = input_tensor.unsqueeze(0) # Add batch dim if single sample

    sampled_probabilities = []
    start_infer = time.time()
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            sampled_probabilities.append(probabilities.cpu().numpy())
    end_infer = time.time()
    print(f"Inference Time ({n_samples} samples): {(end_infer - start_infer)*1000:.2f} ms")


    # Stack results: list of [batch_size, num_classes] -> [n_samples, batch_size, num_classes]
    sampled_probabilities = np.stack(sampled_probabilities, axis=0)

    # Calculate mean probability across samples
    mean_probs = np.mean(sampled_probabilities, axis=0) # Shape: [batch_size, num_classes]

    # Calculate standard deviation across samples (measure of uncertainty)
    std_dev_probs = np.std(sampled_probabilities, axis=0) # Shape: [batch_size, num_classes]

    # Get mean predicted class
    mean_preds = np.argmax(mean_probs, axis=1) # Shape: [batch_size]

    return mean_preds, mean_probs, std_dev_probs


# --- Main Execution ---
if __name__ == "__main__":
    if not BLITZ_AVAILABLE:
        exit()

    # Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Run the experiment
    trained_model, X_val_tensor, y_val_tensor = run_bnn_experiment(device, retrain=True) # Force training

    if trained_model:
        print("\nBNN Experiment Complete.")
        # Final accuracy already printed in run_bnn_experiment

        # --- Demonstrate Inference with Uncertainty ---
        # Select a few validation points for detailed analysis
        num_inference_points = 5
        if len(X_val_tensor) >= num_inference_points:
            inference_indices = np.random.choice(len(X_val_tensor), num_inference_points, replace=False)
            X_infer_np = X_val_tensor[inference_indices].cpu().numpy()
            y_infer_np = y_val_tensor[inference_indices].cpu().numpy() # True labels for comparison

            mean_preds, mean_probs, std_dev_probs = predict_with_uncertainty(
                trained_model, X_infer_np, n_samples=INFERENCE_SAMPLES, device=device
            )

            print("\n--- Detailed Inference Results (Sample Points) ---")
            for i in range(num_inference_points):
                print(f"Input Point {i+1}: {X_infer_np[i].round(3)}")
                print(f"  True Class:      {y_infer_np[i]}")
                print(f"  Mean Pred Class: {mean_preds[i]}")
                print(f"  Mean Probs:      {[f'{p:.3f}' for p in mean_probs[i]]}")
                print(f"  Std Dev Probs:   {[f'{s:.3f}' for s in std_dev_probs[i]]}  <-- Uncertainty")
                print("-" * 20)

                # Interpretation example: If std dev is high for all classes,
                # the model is very uncertain overall for this input.
                # If std dev is high for the predicted class probability,
                # the model isn't very confident in that specific prediction value.

        else:
            print("\nNot enough validation samples to run detailed inference demo.")

    else:
        print("\nTraining skipped or failed.")
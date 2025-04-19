import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import nnfs
from nnfs.datasets import spiral_data
import os
from sklearn.model_selection import train_test_split

# --- Constants ---
# Data & Training Params
DATA_SAMPLES = 2000
DATA_CLASSES = 3
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
EPOCHS = 3001 # CNNs might converge faster/slower than MLP
BATCH_SIZE = 128
LEARNING_RATE = 0.005 # Adjust LR if needed
WEIGHT_DECAY = 1e-4   # Optional L2 Regularization

# --- File Paths ---
CNN_MODEL_SAVE_PATH = "spiral_cnn_model.pth"

# --- CNN Classifier Model ---
class CNNClassifier1D(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        # Input shape: [batch_size, input_channels (2), length (1)]

        self.conv_block1 = nn.Sequential(
            # Kernel size 1 acts like a linear transformation across channels
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=1, padding=0),
            nn.ReLU(),
            # Optional: Batch Normalization can sometimes help
            # nn.BatchNorm1d(16)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(32)
        )
        self.conv_block3 = nn.Sequential(
             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, padding=0),
             nn.ReLU(),
        )

        # Flatten the output of conv blocks
        self.flatten = nn.Flatten() # Flattens dimensions starting from dim 1

        # Classifier Head (Fully Connected Layers)
        # Input size is num_channels * length = 64 * 1 = 64
        self.classifier_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2), # Add dropout for regularization
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape [batch_size, 2]
        """
        # Reshape [batch_size, 2] -> [batch_size, 2, 1] (Batch, Channels, Length)
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        # Apply Convolutional Blocks
        x = self.conv_block1(x) # Output: [batch_size, 16, 1]
        x = self.conv_block2(x) # Output: [batch_size, 32, 1]
        x = self.conv_block3(x) # Output: [batch_size, 64, 1]


        # Flatten for the classifier
        x = self.flatten(x)     # Output: [batch_size, 64]

        # Apply Classifier Head
        logits = self.classifier_head(x) # Output: [batch_size, num_classes]
        return logits


# --- Helper Function for Accuracy (same as before) ---
def calculate_accuracy(y_pred_logits, y_true):
    predicted_classes = torch.argmax(y_pred_logits, dim=1)
    correct = (predicted_classes == y_true).sum().item()
    accuracy = correct / len(y_true)
    return accuracy

# --- Training and Evaluation Function ---
def run_cnn_experiment(device, retrain=False):
    """Trains/Evaluates the 1D CNN model."""

    # --- Check if retraining is needed ---
    if not retrain and os.path.exists(CNN_MODEL_SAVE_PATH):
        print(f"Model file '{CNN_MODEL_SAVE_PATH}' exists. Skipping training.")
        return None, None, None # Indicate skip

    print("--- Starting 1D CNN Experiment ---")

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
    model = CNNClassifier1D(input_channels=2, num_classes=DATA_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Training Loop ---
    print("Starting Training...")
    start_time_train = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        batch_count = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += calculate_accuracy(logits, batch_y)
            batch_count += 1

        avg_epoch_loss = epoch_loss / batch_count
        avg_epoch_acc = epoch_acc / batch_count

        # --- Periodic Validation ---
        if not epoch % 200:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = criterion(val_logits, y_val)
                val_accuracy = calculate_accuracy(val_logits, y_val)
            print(f'Epoch: {epoch}, Train Loss: {avg_epoch_loss:.4f}, Train Acc: {avg_epoch_acc:.3f} | Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.3f}')

    end_time_train = time.time()
    print(f"\nTraining finished in {end_time_train - start_time_train:.2f} seconds.")

    # --- Final Evaluation ---
    print("\n--- Evaluating Final Model on Validation Set ---")
    model.eval()
    final_val_loss = 0.0
    final_val_acc = 0.0
    with torch.no_grad():
        val_logits = model(X_val)
        final_val_loss = criterion(val_logits, y_val).item()
        final_val_acc = calculate_accuracy(val_logits, y_val)

    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")

    # --- Save the Final Model ---
    print(f"Saving trained model to: {CNN_MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), CNN_MODEL_SAVE_PATH)
    print("Model saved successfully.")

    return model, final_val_loss, final_val_acc

# --- Main Execution ---
if __name__ == "__main__":
    # Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Run the experiment
    trained_model, val_loss, val_acc = run_cnn_experiment(device, retrain=True) # Force training

    if trained_model:
        print("\nCNN Experiment Complete.")
        print(f"Achieved Validation Accuracy: {val_acc:.4f}")
    else:
        print("\nTraining skipped as model file already exists.")
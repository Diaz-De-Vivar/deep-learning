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

# --- Constants ---
# Data & Training Params
DATA_SAMPLES = 2000
DATA_CLASSES = 3
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
EPOCHS = 5001 # Transformers might need more/less tuning
BATCH_SIZE = 128 # Use batching for potentially better training stability/speed
LEARNING_RATE = 0.001 # Transformers often benefit from smaller LR & schedulers
# Transformer Hyperparameters
D_MODEL = 32       # Embedding dimension (must be divisible by NHEAD) - Keep small for this simple task
NHEAD = 4          # Number of attention heads
DIM_FEEDFORWARD = 64 # Dimension of the hidden layer in the FFN
NUM_LAYERS = 2      # Number of Transformer encoder layers
DROPOUT = 0.1      # Dropout rate

# --- File Paths ---
TRANSFORMER_MODEL_SAVE_PATH = "spiral_transformer_model.pth"


# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create position indices (0, 1, 2, ...)
        position = torch.arange(max_len).unsqueeze(1) # [max_len, 1]

        # Calculate the division term for sine/cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # [d_model/2]

        # Allocate positional encoding matrix
        pe = torch.zeros(max_len, 1, d_model) # Shape for broadcasting: [max_len, 1, d_model]

        # Calculate sine for even indices, cosine for odd indices
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Register 'pe' as a buffer, not a parameter to be trained
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] # batch_first=True assumed
        """
        # Add positional encoding to the input tensor x
        # x.size(1) is the sequence length
        # self.pe[:x.size(1)] selects encodings for the actual sequence length
        # .transpose(0, 1) changes pe shape to [1, seq_len, d_model] to match batch dim
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


# --- Transformer Classifier Model ---
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, dim_feedforward: int,
                 num_layers: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = input_dim # Here, input_dim is 2 (x, y) which we treat as seq_len

        # 1. Input Embedding: Project 1 feature (x or y) to d_model
        # We apply this linearly to each coordinate independently first.
        self.input_proj = nn.Linear(1, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.seq_len) # Max length is just 2

        # 3. Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Expect input as (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers
        )

        # 4. Classification Head
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        # Initialize weights for better stability
        initrange = 0.1
        self.input_proj.weight.data.uniform_(-initrange, initrange)
        self.input_proj.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Input tensor, shape [batch_size, seq_len (2), features_per_element (1)]
                 For our case, we reshape [batch_size, 2] -> [batch_size, 2, 1]
        """
        # Reshape [batch_size, 2] -> [batch_size, 2, 1]
        if src.ndim == 2:
             src = src.unsqueeze(-1)

        # Input projection: [batch_size, 2, 1] -> [batch_size, 2, d_model]
        src = self.input_proj(src) * math.sqrt(self.d_model) # Scale embedding

        # Add positional encoding: [batch_size, 2, d_model] -> [batch_size, 2, d_model]
        src = self.pos_encoder(src)

        # Pass through transformer encoder: [batch_size, 2, d_model] -> [batch_size, 2, d_model]
        output = self.transformer_encoder(src)

        # Classification: Use the output of the first element ([CLS] token style)
        # [batch_size, 2, d_model] -> [batch_size, d_model]
        output = output[:, 0, :] # Take the output corresponding to the 'x' position

        # Final linear layer: [batch_size, d_model] -> [batch_size, num_classes]
        logits = self.classifier(output)
        return logits


# --- Helper Function for Accuracy (same as before) ---
def calculate_accuracy(y_pred_logits, y_true):
    predicted_classes = torch.argmax(y_pred_logits, dim=1)
    correct = (predicted_classes == y_true).sum().item()
    accuracy = correct / len(y_true)
    return accuracy

# --- Training and Evaluation Function ---
def run_experiment(device, retrain=False):
    """Trains/Evaluates the Transformer model."""

    # --- Check if retraining is needed ---
    if not retrain and os.path.exists(TRANSFORMER_MODEL_SAVE_PATH):
        print(f"Model file '{TRANSFORMER_MODEL_SAVE_PATH}' exists. Skipping training.")
        # Optionally load and evaluate here if needed
        return None, None, None # Indicate skip

    print("--- Starting Transformer Experiment ---")

    # --- Data Preparation ---
    nnfs.init()
    X_np_full, y_np_full = spiral_data(samples=DATA_SAMPLES, classes=DATA_CLASSES)
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        X_np_full, y_np_full, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=y_np_full
    )
    print(f"Training samples: {len(X_train_np)}, Validation samples: {len(X_val_np)}")

    # Convert to Tensors
    X_train = torch.from_numpy(X_train_np).float() # Keep on CPU for DataLoader
    y_train = torch.from_numpy(y_train_np).long()
    X_val = torch.from_numpy(X_val_np).float().to(device) # Move val data directly
    y_val = torch.from_numpy(y_val_np).long().to(device)

    # Create DataLoaders for batching
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # We can evaluate validation in one go if it fits memory
    # val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- Model, Loss, Optimizer ---
    model = TransformerClassifier(
        input_dim=2, # We treat the 2 features (x,y) as sequence length
        d_model=D_MODEL,
        nhead=NHEAD,
        dim_feedforward=DIM_FEEDFORWARD,
        num_layers=NUM_LAYERS,
        num_classes=DATA_CLASSES,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Optional: Learning rate scheduler (common for transformers)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

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

            # Forward pass
            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient clipping (common for transformers)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += calculate_accuracy(logits, batch_y)
            batch_count += 1

        # Optional: Scheduler step
        # scheduler.step()

        avg_epoch_loss = epoch_loss / batch_count
        avg_epoch_acc = epoch_acc / batch_count

        # --- Periodic Validation ---
        if not epoch % 200: # Validate less often maybe
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val) # Evaluate full validation set
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
    print(f"Saving trained model to: {TRANSFORMER_MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), TRANSFORMER_MODEL_SAVE_PATH)
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
    trained_model, val_loss, val_acc = run_experiment(device, retrain=True) # Set retrain=True to force training

    if trained_model:
        print("\nExperiment Complete.")
        print(f"Achieved Validation Accuracy: {val_acc:.4f}")
    else:
        print("\nTraining skipped as model file already exists.")
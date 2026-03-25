#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
################################################################################
# IMPORT ALL REQUIRED MODULES
################################################################################
################################################################################
import os
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
 
from torch.utils.data import DataLoader, TensorDataset


################################################################################
################################################################################
# PYTORCH TO LOAD AND PREP THE DATA
################################################################################
################################################################################
NUM_SAMPLES = 400000    # Number of EOS samples
# Directory to save/load dataset and models 
# If not defined, model will save in current directory.
save_dir = f"{NUM_SAMPLES}files"

# Load dataset
data = np.load(os.path.join(save_dir, f"EOS_dataset_{NUM_SAMPLES}samples.npy"))

# ==========================================================
# Set device and batch size
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
BATCH_SIZE = 256

# ==========================================================
# 1. Split into training and validation sets
# ==========================================================
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

# ==========================================================
# 2. Convert to Tensor
# ==========================================================
# Inputs: Cols 0 to 4 (5 features: log_rho_c, log_p, Gamma1, Gamma2, Gamma3)
# Output: Col 5 - 7 (Mass, Radius, Tidal Deformability)
X_eos_train = torch.tensor(train_data[:, :5], dtype=torch.float32)  # Training Input
y_eos_train = torch.tensor(train_data[:, 5:8], dtype=torch.float32) # Training Output

X_eos_val = torch.tensor(val_data[:, :5], dtype=torch.float32)      # Validation Input
y_eos_val = torch.tensor(val_data[:, 5:8], dtype=torch.float32)     # Validation Output

# ==========================================================
# 3. Normalize the all the inputs using Z-score (Pressure is already logged)
# ==========================================================
X_eos_mean = X_eos_train.mean(dim=0, keepdim=True)
X_eos_std = X_eos_train.std(dim=0, keepdim=True)
# Save the normalization statistics to be used later for maodel evaluation
torch.save(X_eos_mean, os.path.join(save_dir, "X_eos_mean.pt"))
torch.save(X_eos_std, os.path.join(save_dir, "X_eos_std.pt"))
print("Normalization statistics saved.")
X_train_norm = (X_eos_train - X_eos_mean) / X_eos_std
X_val_norm = (X_eos_val - X_eos_mean) / X_eos_std

# ==========================================================
# 4. Separate the outputs (Mass, Radius and TD) from the training and validation datasets
# ==========================================================
y_mass_train, y_radius_train, y_td_train = y_eos_train[:, 0:1], y_eos_train[:, 1:2], y_eos_train[:, 2:3]
y_mass_val, y_radius_val, y_td_val = y_eos_val[:, 0:1], y_eos_val[:, 1:2], y_eos_val[:, 2:3]

# ==========================================================
# 5. Normalize the outputs
# ==========================================================
# 5.1  Constant Scaling on Mass (M)
MASS_SCALE = 3.5
y_mass_train_norm = y_mass_train / MASS_SCALE
y_mass_val_norm = y_mass_val / MASS_SCALE

# 5.2. Constant Scaling on Radius (R)
RADIUS_SCALE = 25.0
y_radius_train_norm = y_radius_train / RADIUS_SCALE
y_radius_val_norm = y_radius_val / RADIUS_SCALE

# 5.3 Log Scaling on Tidal Deformability (TD)
y_td_train_norm = torch.log10(y_td_train)
y_td_val_norm = torch.log10(y_td_val)

# ==========================================================
# 6. Recombine Outputs
# ==========================================================
y_train_norm = torch.cat((y_mass_train_norm, y_radius_train_norm, y_td_train_norm), dim=1)
y_val_norm = torch.cat((y_mass_val_norm, y_radius_val_norm, y_td_val_norm), dim=1)


################################################################################
################################################################################
# DEFINE THE MODEL
################################################################################
################################################################################
# ==========================================================
# Single Residual Network Block
# ==========================================================
class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        out = self.act(self.fc(x))
        return x + out   # Residual connection


# ==========================================================
# Residual Network
# ==========================================================
class PhysicsEmulator(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=512):
        super().__init__()

        # 1. Initial encoding of all inputs (EOS params + logp)
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # 2. Residual blocks
        self.block1 = ResNetBlock(hidden_dim)
        self.block2 = ResNetBlock(hidden_dim)
        self.block3 = ResNetBlock(hidden_dim)
        self.block4 = ResNetBlock(hidden_dim)

        # 3. Output layers
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 512 -> 256
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)            # 256 -> 3 outputs
        )

    def forward(self, x):

        # 1. Encode full input
        x_hidden = self.input_layer(x)      # 5 -> 512

        # 2. Residual flow
        x_hidden = self.block1(x_hidden)
        x_hidden = self.block2(x_hidden)
        x_hidden = self.block3(x_hidden)
        x_hidden = self.block4(x_hidden)

        # 3. Prediction
        return self.final_layer(x_hidden)
    

################################################################################
################################################################################
# TRAIN THE MODEL
################################################################################
################################################################################
# ==========================================================
# Plotting Function
# ==========================================================
def plot_and_save_losses(train_losses, val_losses, filename="loss_curve.png"):
    """Plots training and validation loss and saves the figure."""
    epochs = range(len(train_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss (Normalized)')
    plt.yscale('log') # Using log scale for clearer visualization of small losses
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    try:
        plt.savefig(os.path.join(save_dir, filename))
        print(f"Loss plot saved to {filename}", flush=True)
    except Exception as e:
        print(f"ERROR saving plot: {e}", flush=True)
    plt.close()

# ==========================================================
# Set training parameters
# ==========================================================
model = PhysicsEmulator().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,T_max=1000, eta_min=1e-7
            )   # Modulates adaptive learning rate
criterion = nn.HuberLoss()
epochs = 500    # With early stopping
patience = 50  # Number of epochs to wait for improvement before stopping
patience_counter = 0
best_val_loss = float('inf')

# Lists to store losses or plotting
train_losses = []
val_losses = []

# ==============================================================================
# Load training and validation tensors
# ==============================================================================
# Ensure Mass and Radius are Torch Tensors
if isinstance(y_train_norm, np.ndarray):
    y_train_norm = torch.from_numpy(y_train_norm).to(torch.float32)
if isinstance(y_val_norm, np.ndarray):
    y_val_norm = torch.from_numpy(y_val_norm).to(torch.float32)

train_loader = DataLoader(TensorDataset(X_train_norm, y_train_norm), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_norm, y_val_norm), batch_size=BATCH_SIZE, shuffle=False)

# ==============================================================================
# Training
# ==============================================================================
for epoch in range(epochs):
    # 1. Set model in training mode
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()       # Clear previous gradient
        pred = model(X_b)           # Make prediction
        loss = criterion(pred, y_b) # Calculate training loss
        loss.backward()             # Backpropagate loss
        optimizer.step()            # Use optimizer
        train_loss += loss.item()   # Update training loss
    
    train_loss /= len(train_loader)
    
    # 2. Set model in evaluation mode
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            pred = model(X_b)           # Make prediction
            loss = criterion(pred, y_b) # Calculate validation loss
            val_loss += loss.item()     # Update validation loss
    
    val_loss /= len(val_loader)
    scheduler.step()                    # Update Scheduler
    
    # 3. Append Losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # 4. Early Stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model weights
        best_model_state = model.state_dict()

    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 5. Calculate and print error in output parameters every 100 epochs
    if epoch % 100 == 0:
        # Calculate the Approximate Physical Error in km
        radius_error_km = np.sqrt(2 * val_loss) * RADIUS_SCALE 
        mass_error = np.sqrt(2 * val_loss) * MASS_SCALE
        td_error_log = np.sqrt(2 * val_loss)  # Since td is log-scaled, this is in log units

        print(f"Epoch {epoch} | Train Loss: {train_loss:.6e} | Val Loss: {val_loss:.6e} | Approx Radius Error: {radius_error_km:.4f} km | Approx Mass Error: {mass_error:.4f} | Approx TD error: {td_error_log:.4f}", flush=True)

        # 6. Plot and save errors
        # Plot every 100 epochs
        if epoch % 100 == 0 and epoch > 0:
            plot_and_save_losses(train_losses, val_losses, filename=f"loss_curve_epoch{epoch}.png")

# Restore best model
model.load_state_dict(best_model_state)
print(f"Training finished. Best validation loss: {best_val_loss:.10f}")

# 7. Final Plot after training finishes
plot_and_save_losses(train_losses, val_losses, filename="loss_curve_final.png")

# 8. Saving the best model to be loaded later
torch.save(model.state_dict(), os.path.join(save_dir, "Best_EOS_Model.pth"))

print("Training complete. Best validation loss:", best_val_loss)

################################################################################
################################################################################
# END OF TRAINING
################################################################################
################################################################################
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

# --- Updated Paths ---
PREPROCESSED_DIR = os.path.join(DATASET_DIR, 'preprocessed')
MODEL_PATH = os.path.join(DATASET_DIR, 'ids_model.pth')
LE_PATH = os.path.join(PREPROCESSED_DIR, 'label_encoder.pkl')
HISTORY_PATH = os.path.join(DATASET_DIR, 'train_history.json') 

X_TRAIN_PATH = os.path.join(PREPROCESSED_DIR, 'X_train.npy')
Y_TRAIN_PATH = os.path.join(PREPROCESSED_DIR, 'y_train.npy')
X_VAL_PATH = os.path.join(PREPROCESSED_DIR, 'X_val.npy')
Y_VAL_PATH = os.path.join(PREPROCESSED_DIR, 'y_val.npy')
# --------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 30
PATIENCE = 5
# --------------------------

# --- Load preprocessed data directly ---
print("Loading preprocessed data...")
X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)
X_val = np.load(X_VAL_PATH)
y_val = np.load(Y_VAL_PATH)
le = joblib.load(LE_PATH)
print("Data loaded.")

# Compute class weights
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# --------------------------
# Model definition
# --------------------------
class IDSModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Initialize model
input_dim = X_train.shape[1]
num_classes = len(le.classes_)
model = IDSModel(input_dim, num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# --------------------------
# Training loop
# --------------------------
best_val_f1 = 0
patience_counter = 0

# --- Lists to store all metrics ---
history = {
    'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
    'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
}

print("Starting model training...")
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
    
    # --- Compute all training metrics for this epoch ---
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    train_loss_epoch = total_loss / len(train_loader)
    train_acc_epoch = accuracy_score(all_labels_np, all_preds_np)
    train_precision_epoch = precision_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
    train_recall_epoch = recall_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
    train_f1_epoch = f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
    
    # --- Compute all validation metrics ---
    model.eval()
    with torch.no_grad():
        logits_val = model(X_val_tensor)
        val_loss_epoch = criterion(logits_val, y_val_tensor).item() 
        preds_val = torch.argmax(logits_val, dim=1).cpu().numpy()
        y_val_np = y_val_tensor.cpu().numpy()
        val_acc_epoch = accuracy_score(y_val_np, preds_val)
        val_precision_epoch = precision_score(y_val_np, preds_val, average='macro', zero_division=0)
        val_recall_epoch = recall_score(y_val_np, preds_val, average='macro', zero_division=0)
        val_f1_epoch = f1_score(y_val_np, preds_val, average='macro', zero_division=0)
    
    # --- Store all metrics ---
    history['train_loss'].append(train_loss_epoch)
    history['train_acc'].append(train_acc_epoch)
    history['train_precision'].append(train_precision_epoch)
    history['train_recall'].append(train_recall_epoch)
    history['train_f1'].append(train_f1_epoch)
    
    history['val_loss'].append(val_loss_epoch)
    history['val_acc'].append(val_acc_epoch)
    history['val_precision'].append(val_precision_epoch)
    history['val_recall'].append(val_recall_epoch)
    history['val_f1'].append(val_f1_epoch)
    
    # --- Updated Print Statement ---
    print(f"--- Epoch {epoch}/{EPOCHS} ---")
    print(f"Train | Loss: {train_loss_epoch:.4f} | Acc: {train_acc_epoch*100:.2f}% | "
          f"Prec: {train_precision_epoch:.4f} | Rec: {train_recall_epoch:.4f} | F1: {train_f1_epoch:.4f}")
    print(f"Valid | Loss: {val_loss_epoch:.4f} | Acc: {val_acc_epoch*100:.2f}% | "
          f"Prec: {val_precision_epoch:.4f} | Rec: {val_recall_epoch:.4f} | F1: {val_f1_epoch:.4f}")
    
    scheduler.step(val_f1_epoch)
    
    # Early stopping
    if val_f1_epoch > best_val_f1:
        best_val_f1 = val_f1_epoch
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Saved best model (Val F1: {best_val_f1:.4f}).")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print(f"\nTraining complete. Best model saved to {MODEL_PATH}")

# --- Save history to JSON ---
with open(HISTORY_PATH, 'w') as f:
    json.dump(history, f, indent=4)
print(f"Training history saved to {HISTORY_PATH}")

# --------------------------
# Final summarized metrics on validation set (using best model)
# --------------------------
print("\nLoading best model for final validation summary...")
model.load_state_dict(torch.load(MODEL_PATH)) # Load best model
model.eval()
with torch.no_grad():
    logits_val = model(X_val_tensor)
    preds_val = torch.argmax(logits_val, dim=1).cpu().numpy()
    y_val_np = y_val_tensor.cpu().numpy()
    final_acc = accuracy_score(y_val_np, preds_val)
    final_precision = precision_score(y_val_np, preds_val, average='macro', zero_division=0)
    final_recall = recall_score(y_val_np, preds_val, average='macro', zero_division=0)
    final_f1 = f1_score(y_val_np, preds_val, average='macro', zero_division=0)

print("\n🎯 Final Validation Metrics Summary (Best Model):")
print(f"Accuracy : {final_acc*100:.2f}%")
print(f"Precision: {final_precision:.4f}")
print(f"Recall   : {final_recall:.4f}")
print(f"F1 Score : {final_f1:.4f}")
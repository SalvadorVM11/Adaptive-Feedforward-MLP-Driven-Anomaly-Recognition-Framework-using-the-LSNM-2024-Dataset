# ==========================================
# evaluate.py — IDS Model Evaluation
# ==========================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
# Note: We no longer import from preprocess.py

# --------------------------
# PATH CONFIGURATION
# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

# --- Updated Paths ---
PREPROCESSED_DIR = os.path.join(DATASET_DIR, 'preprocessed')
MODEL_PATH = os.path.join(DATASET_DIR, 'ids_model.pth')
REPORT_PATH = os.path.join(DATASET_DIR, 'evaluation_report.txt')
CM_PATH = os.path.join(DATASET_DIR, "confusion_matrix.png")
LE_PATH = os.path.join(PREPROCESSED_DIR, 'label_encoder.pkl')
X_TEST_PATH = os.path.join(PREPROCESSED_DIR, 'X_test.npy')
Y_TEST_PATH = os.path.join(PREPROCESSED_DIR, 'y_test.npy')
# --------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------
# LOAD DATA
# --------------------------
print("📂 Loading preprocessed test data and label encoder...")
X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)
le = joblib.load(LE_PATH)
print("Test data loaded.")

# Convert to tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

# --------------------------
# MODEL DEFINITION
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

# --------------------------
# LOAD TRAINED MODEL
# --------------------------
input_dim = X_test.shape[1]
num_classes = len(le.classes_)
model = IDSModel(input_dim, num_classes).to(DEVICE)

print(f"🧠 Loading trained IDS model from {MODEL_PATH}...")
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please run train.py to train and save the model first.")
    exit()
    
model.eval()

# --------------------------
# INFERENCE
# --------------------------
print("🔍 Running inference on test data...")
with torch.no_grad():
    logits = model(X_test_tensor)
    probs = F.softmax(logits, dim=1)
    y_pred = torch.argmax(probs, dim=1).cpu().numpy()

# --------------------------
# METRIC COMPUTATION
# --------------------------
y_test_np = y_test

print("\n📊 Classification Report:\n")
cls_report = classification_report(y_test_np, y_pred, target_names=le.classes_, zero_division=0)
print(cls_report)

acc = accuracy_score(y_test_np, y_pred)
prec = precision_score(y_test_np, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test_np, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test_np, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_test_np, y_pred)

print(f"✅ Accuracy: {acc * 100:.2f}%")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall: {rec:.4f}")
print(f"✅ F1-score: {f1:.4f}")

# Save metrics to text file
with open(REPORT_PATH, "w") as f:
    f.write("=== IDS Model Evaluation Report ===\n\n")
    f.write(f"Model Path: {MODEL_PATH}\n\n")
    f.write(cls_report + "\n")
    f.write("--- Weighted Averages ---\n")
    f.write(f"Accuracy: {acc * 100:.2f}%\n")
    f.write(f"Precision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f1:.4f}\n")
    f.write("\n--- Confusion Matrix ---\n")
    f.write(np.array2string(cm, separator=', '))

print(f"\n📁 Evaluation report saved at: {REPORT_PATH}")

# --------------------------
# CONFUSION MATRIX VISUALIZATION
# --------------------------
print("📈 Generating confusion matrix visualization...")

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix (Test Set)", fontsize=16)
plt.colorbar()

tick_marks = np.arange(len(le.classes_))
plt.xticks(tick_marks, le.classes_, rotation=45, ha='right', fontsize=10)
plt.yticks(tick_marks, le.classes_, fontsize=10)

# Annotate cells
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, format(cm[i, j], 'd'),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=9
        )

plt.tight_layout()
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

plt.savefig(CM_PATH, dpi=300, bbox_inches='tight')
# plt.show() # Uncomment this if you want the plot to pop up

print(f"🖼️ Confusion matrix saved at: {CM_PATH}")

# --------------------------
# END OF SCRIPT
# --------------------------
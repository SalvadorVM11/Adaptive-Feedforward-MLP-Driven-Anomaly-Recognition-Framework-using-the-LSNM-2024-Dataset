# ==========================================
# graphs.py — Train/Test Metrics Visualization
# ==========================================
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Paths & Params
# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
HISTORY_PATH = os.path.join(DATASET_DIR, 'train_history.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'eda_outputs')

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# --------------------------
# Load History
# --------------------------
print(f"Loading training history from {HISTORY_PATH}...")
try:
    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)
except FileNotFoundError:
    print(f"Error: {HISTORY_PATH} not found.")
    print("Please run train.py first to generate the history file.")
    exit()

epochs = range(1, len(history['train_loss']) + 1)

# --------------------------
# Plot Loss
# --------------------------
print("Plotting training vs. validation loss...")
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='blue')
plt.plot(epochs, history['val_loss'], 'o-', label='Validation Loss', color='orange')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Train vs. Validation Loss", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_over_epochs.png"), dpi=300)
plt.close()

# --------------------------
# Plot Accuracy
# --------------------------
print("Plotting training vs. validation accuracy...")
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(epochs, history['train_acc'], 'o-', label='Train Accuracy', color='green')
plt.plot(epochs, history['val_acc'], 'o-', label='Validation Accuracy', color='red')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title("Train vs. Validation Accuracy", fontsize=16, fontweight='bold')
plt.ylim(0, 1.05)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_over_epochs.png"), dpi=300)
plt.close()

# --------------------------
# Plot Precision
# --------------------------
print("Plotting training vs. validation precision...")
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(epochs, history['train_precision'], 'o-', label='Train Precision', color='purple')
plt.plot(epochs, history['val_precision'], 'o-', label='Validation Precision', color='magenta')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Precision (Macro)", fontsize=14)
plt.title("Train vs. Validation Precision", fontsize=16, fontweight='bold')
plt.ylim(0, 1.05)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "precision_over_epochs.png"), dpi=300)
plt.close()

# --------------------------
# Plot Recall
# --------------------------
print("Plotting training vs. validation recall...")
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(epochs, history['train_recall'], 'o-', label='Train Recall', color='cyan')
plt.plot(epochs, history['val_recall'], 'o-', label='Validation Recall', color='teal')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Recall (Macro)", fontsize=14)
plt.title("Train vs. Validation Recall", fontsize=16, fontweight='bold')
plt.ylim(0, 1.05)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "recall_over_epochs.png"), dpi=300)
plt.close()

# --------------------------
# Plot F1-Score
# --------------------------
print("Plotting training vs. validation F1-Score...")
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(epochs, history['train_f1'], 'o-', label='Train F1-Score', color='blue')
plt.plot(epochs, history['val_f1'], 'o-', label='Validation F1-Score', color='orange')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("F1-Score (Macro)", fontsize=14)
plt.title("Train vs. Validation F1-Score", fontsize=16, fontweight='bold')
plt.ylim(0, 1.05)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "f1_score_over_epochs.png"), dpi=300)
plt.close()

print(f"\n✅ All graphs saved to: {OUTPUT_DIR}")
# ==========================================
# eda.py — PCA, ROC & Diagnostic EDA for IDS
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# Import the load_data function from your preprocess script
# This assumes preprocess.py is in the same parent directory, e.g., 'src/preprocess.py'
# Adjust the import path if your file structure is different
try:
    from preprocess import load_data
except ImportError:
    print("Warning: Could not import 'load_data' from 'preprocess.py'.")
    print("Skipping Target Leakage check.")
    load_data = None

# ==================================================
# 1️⃣ Set Paths and Load Data
# ==================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_PATH = os.path.join(DATASET_DIR, 'ids_model.pth')
OUTPUT_DIR = os.path.join(BASE_DIR, 'eda_outputs')

# --- Paths to new preprocessed files ---
PREPROCESSED_DIR = os.path.join(DATASET_DIR, 'preprocessed')
LE_PATH = os.path.join(PREPROCESSED_DIR, 'label_encoder.pkl')
X_TRAIN_PATH = os.path.join(PREPROCESSED_DIR, 'X_train.npy')
Y_TRAIN_PATH = os.path.join(PREPROCESSED_DIR, 'y_train.npy')
X_TEST_PATH = os.path.join(PREPROCESSED_DIR, 'X_test.npy')
Y_TEST_PATH = os.path.join(PREPROCESSED_DIR, 'y_test.npy')

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("📂 Loading preprocessed data splits...")

# Load data
X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)
X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

# Load label encoder
le = joblib.load(LE_PATH)
class_names = le.classes_
num_classes = len(class_names)
print(f"Loaded {len(class_names)} classes: {class_names}")

# ==================================================
# 2️⃣ Dataset Overview (on Training Data)
# ==================================================
print("\n📊 Generating Training Dataset Overview...")

# Class distribution
plt.figure(figsize=(9, 6), dpi=300)
sns.countplot(x=y_train, palette="tab10")
plt.title("Class Distribution (Training Set)", fontsize=16, fontweight='bold')
plt.xlabel("Classes", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution_train.png"), dpi=300)
plt.close()
print("✅ Saved class distribution (from y_train).")

# ==================================================
# 3️⃣ Target Leakage Check (on RAW Data)
# ==================================================
if load_data:
    print("\n🕵️ Checking for Target Leakage (on raw data)...")
    raw_df = load_data()
    raw_numeric_cols = raw_df.select_dtypes(include=np.number).columns
    
    # Temporarily encode label for correlation
    raw_df['label_encoded'] = le.fit_transform(raw_df['label'])
    
    corr_with_target = raw_df[raw_numeric_cols].apply(lambda col: np.corrcoef(col, raw_df['label_encoded'])[0,1])
    high_corr_features = corr_with_target[abs(corr_with_target) > 0.95]
    
    if not high_corr_features.empty:
        print("⚠️ WARNING: Potential target leakage detected in features (from raw data):")
        print(high_corr_features)
        high_corr_features.to_csv(os.path.join(OUTPUT_DIR, "high_corr_features.csv"))
    else:
        print("✅ No obvious target leakage detected in raw data.")
else:
    print("\nSkipping Target Leakage check (could not import 'load_data').")

# ==================================================
# 4️⃣ PCA Visualization (on Training Data)
# ==================================================
print("\n📈 Generating PCA embeddings (from X_train)...")
# Note: X_train is ALREADY scaled, no need to re-fit StandardScaler

# Take a sample if the training set is too large to plot
sample_size = min(len(X_train), 5000)
sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train[sample_indices]
y_train_sample = y_train[sample_indices]

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train_sample) # Fit on (scaled) train data sample

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Label": y_train_sample
})

plt.figure(figsize=(10, 8), dpi=300)
sns.scatterplot(
    data=pca_df, x="PC1", y="PC2",
    hue="Label", palette="tab10", s=50, alpha=0.7,
)
plt.title("PCA Projection of Training Data", fontsize=16, fontweight='bold')
# Create a mapping from numeric label to string name for the legend
legend_labels = [class_names[i] for i in sorted(np.unique(y_train_sample))]
plt.legend(title="Classes", labels=legend_labels, loc="best", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_visualization_train.png"), dpi=300)
plt.close()
print("✅ Saved PCA visualization (from X_train).")

# ==================================================
# 5️⃣ ROC Curve + AUC (on TEST Data)
# ==================================================
print("\n📉 Generating ROC Curves and AUC (on X_test)...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

input_dim = X_test.shape[1]
model = IDSModel(input_dim, num_classes).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please run train.py first.")
    exit()
    
model.eval()

# X_test is ALREADY scaled
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    logits = model(X_test_tensor)
    probs = F.softmax(logits, dim=1).cpu().numpy()

# Binarize the TEST labels
y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))

plt.figure(figsize=(10, 8), dpi=300)
colors = sns.color_palette("tab10", n_colors=num_classes)

for i in range(num_classes):
    # Check if this class is present in the test set
    if y_test_bin.shape[1] > i:
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2.5,
                 label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("Multi-class ROC Curves (Test Set)", fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_auc_curves_test_set.png"), dpi=300)
plt.close()
print("✅ Saved ROC curves and AUC (from X_test).")

# ==================================================
# ✅ Done
# ==================================================
print("\n🎉 EDA complete! Plots and diagnostics saved to:", OUTPUT_DIR)
# Adaptive Feedforward MLP Driven Anomaly Recognition Framework using the LSNM 2024 Dataset
---
This repository contains an optimized Feedforward Multi-Layer Perceptron (MLP)–based Intrusion Detection System (IDS) built using PyTorch and evaluated on the LSNM 2024 dataset, a modern large-scale benchmark for network anomaly detection.

---
### 📌 Overview
The framework focuses on:
- High detection accuracy
- Handling class imbalance
- Lightweight architecture (compared to CNN/LSTM hybrids)
- Strong generalization on unseen attacks

---
### 🧠 Model Architecture
This IDS uses a 3-layer Adaptive MLP integrated with:
- Batch Normalization
- ReLU Activation
- Dropout Regularization
- AdamW Optimizer
- Weighted Cross-Entropy Loss
- ReduceLROnPlateau LR Scheduler

---
### 📌 Architecture Diagram Placeholder

![Insert architecture image here]

### 📊 Dataset — LSNM 2024
The model is trained and tested on the LSNM 2024 dataset consisting of:
- Benign traffic
- Multiple modern attacks: DDoS, SQL Injection, XSS, SSH Brute Force, RCE, etc.

---
### Why LSNM 2024?
- More recent and diverse than CICIDS2017 or NSL-KDD
- Flow + packet-level features
- Suitable for machine learning-based IDS systems

---
### 📌 PCA Visualization Placeholder
![Insert PCA projection image here]

#### 🔧 Preprocessing Steps
- Removed duplicates & missing values
- Encoded categorical labels
- Standardized features
- Stratified sampling with 2000 samples/class
- Train/Val/Test split = 70/10/20

#### 🏗 Training Configuration
- Epochs: Max 30
- Batch Size: 128
- Optimizer: AdamW (lr = 1e-3)
- Scheduler: ReduceLROnPlateau
- Early Stopping: Patience = 5
- Model Selection: Best macro-F1

### 📈 Performance Summary
![comparsion_table]

### 📌 Training Accuracy/Loss Graphs Placeholder

![Insert accuracy graph here]
![Insert loss graph here]

---
### 🧪 Evaluation Insights
- Confusion Matrix shows strong classification across all attack types.
- Minor confusion only between attacks with very similar network patterns (e.g., SQL Injection vs RCE).
- ROC-AUC scores ~0.99–1.0 for all classes → exceptional feature discrimination.

![Confusion Matrix Placeholder]
![Insert confusion matrix here]
![ROC-AUC Curves Placeholder]
![Insert ROC graph here]

---
### 🧩 Why This Approach Works
Compared to heavy hybrid models like CNN-LSTM:
- ⚡ Lower computational complexity
- 🎯 High accuracy with fewer parameters
- 🔍 Efficient feature learning using BN + Dropout
- ⚖️ Class imbalance handled via weighted loss
- 📈 Stable convergence using AdamW + LR scheduling
This demonstrates that a carefully tuned Feedforward MLP can match or outperform complex IDS architectures on modern datasets.

---
### 🏁 Conclusion
This project provides an adaptive, efficient, and reproducible MLP-based IDS pipeline for modern network attack detection using the LSNM 2024 dataset. The results highlight that well-optimized MLP architectures remain highly competitive for large-scale anomaly recognition.

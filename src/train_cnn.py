"""
1D-CNN PyTorch Implementation for Multi-Label Odor Classification.
Compares with XGBoost and LightGBM as required by the thesis.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from pathlib import Path
import json
import time

from .config import path

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

class Odor1DCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # 1D-CNN expects input shape: (batch_size, channels, length)
        # We treat our 2053 features as a 1D sequence with 1 channel
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flattened dimension
        # 2053 -> pool(2) -> 1026 -> pool(2) -> 513
        # 513 * 64 channels = 32832
        
        # Using adaptive pooling to be safe and avoid dimension calc issues
        self.adaptive_pool = nn.AdaptiveAvgPool1d(100)
        self.fc1 = nn.Linear(64 * 100, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x shape: (batch, 1, 2053)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Output is logits. Binary Cross Entropy with Logits will handle sigmoid.
        return x

def run_cnn():
    print("=" * 60)
    print("TRAINING 1D-CNN (PyTorch)")
    print("=" * 60)
    
    proc = path("data_processed")
    X_train = sp.load_npz(proc / "X_train.npz").toarray()
    X_test = sp.load_npz(proc / "X_test.npz").toarray()
    Y_train = sp.load_npz(proc / "Y_train.npz").toarray()
    Y_test = sp.load_npz(proc / "Y_test.npz").toarray()
    
    labels = json.loads((proc / "label_names.json").read_text(encoding="utf-8"))
    
    input_dim = X_train.shape[1]
    num_classes = Y_train.shape[1]
    
    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"Input features: {input_dim}, Labels: {num_classes}")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1) # Add channel dim
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
    Y_train_t = torch.FloatTensor(Y_train)
    
    # DataLoader
    dataset = TensorDataset(X_train_t, Y_train_t)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Model
    model = Odor1DCNN(input_dim, num_classes)
    
    # Weighted loss for imbalance
    pos_weight = (len(Y_train) - Y_train.sum(axis=0)) / np.maximum(1, Y_train.sum(axis=0))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(pos_weight))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    print("\nTraining CNN...")
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
            
    print(f"Training selesai dalam {time.time()-t0:.1f} detik.")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).numpy()
    
    # Find best thresholds
    Y_pred = np.zeros_like(probs)
    for j in range(num_classes):
        # We use standard 0.5 threshold for CNN simplicity
        # or optimal threshold from train
        Y_pred[:, j] = (probs[:, j] >= 0.5).astype(int)
        
    roc_auc = roc_auc_score(Y_test, probs, average="macro")
    f1 = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
    precision = precision_score(Y_test, Y_pred, average="macro", zero_division=0)
    recall = recall_score(Y_test, Y_pred, average="macro", zero_division=0)
    
    print("\n" + "=" * 60)
    print("EVALUASI 1D-CNN PADA 25 LABEL AROMA")
    print("=" * 60)
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    
    results = {
        "model": "1D-CNN",
        "ROC-AUC": float(roc_auc),
        "F1-Macro": float(f1),
        "Precision": float(precision),
        "Recall": float(recall)
    }
    
    with open(REPORTS / "results_cnn.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_cnn()

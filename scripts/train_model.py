import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.cnn_model import CNN1D

import argparse
import numpy as np
import pandas as pd
import ast

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# Load Participant Data
# ----------------------------
def load_participant(file_path):
    df = pd.read_csv(file_path)

    signals = []
    labels = []

    for _, row in df.iterrows():
        signal = np.array(ast.literal_eval(row['signal']))
        signals.append(signal)
        labels.append(row['label'])

    return np.array(signals), np.array(labels)


# ----------------------------
# Main Training Script (LOPO)
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", required=True)
    args = parser.parse_args()

    files = sorted(os.listdir(args.data_dir))

    participants = {}
    for f in files:
        path = os.path.join(args.data_dir, f)
        name = f.split("_")[0]
        X, y = load_participant(path)
        participants[name] = (X, y)

    label_encoder = LabelEncoder()

    # Fit encoder on all labels
    all_labels = []
    for _, (_, y) in participants.items():
        all_labels.extend(y)

    label_encoder.fit(all_labels)

    print("Classes:", label_encoder.classes_)

    results = []

    # ----------------------------
    # LOPO Loop
    # ----------------------------
    for test_participant in participants.keys():

        print(f"\nTesting on {test_participant}")

        X_train, y_train = [], []
        X_test, y_test = [], []

        for name, (X, y) in participants.items():
            if name == test_participant:
                X_test.extend(X)
                y_test.extend(y)
            else:
                X_train.extend(X)
                y_train.extend(y)

        X_train = np.array(X_train)
        X_test = np.array(X_test)

        # ----------------------------
        # Per-window normalization
        # ----------------------------
        X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / \
                  (np.std(X_train, axis=1, keepdims=True) + 1e-8)

        X_test = (X_test - np.mean(X_test, axis=1, keepdims=True)) / \
                 (np.std(X_test, axis=1, keepdims=True) + 1e-8)

        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)

        # ----------------------------
        # Class Weights (balanced)
        # ----------------------------
        num_classes = len(label_encoder.classes_)
        class_weights = np.ones(num_classes)

        unique_classes = np.unique(y_train)

        computed_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train
        )

        for cls, weight in zip(unique_classes, computed_weights):
            class_weights[cls] = weight

        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        # ----------------------------
        # Convert to Torch
        # ----------------------------
        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=32,
            shuffle=True
        )

        # ----------------------------
        # Model
        # ----------------------------
        model = CNN1D(num_classes=num_classes)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # ----------------------------
        # Training
        # ----------------------------
        model.train()
        for epoch in range(10):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

        # ----------------------------
        # Evaluation
        # ----------------------------
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, preds = torch.max(outputs, 1)

        y_pred = preds.numpy()
        y_true = y_test.numpy()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(num_classes)
        )

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("Confusion Matrix:\n", cm)

        results.append({
            "participant": test_participant,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        })

    # ----------------------------
    # Print Overall Average
    # ----------------------------
    avg_acc = np.mean([r["accuracy"] for r in results])
    avg_prec = np.mean([r["precision"] for r in results])
    avg_rec = np.mean([r["recall"] for r in results])

    print("\nOverall LOPO Results")
    print("Average Accuracy:", avg_acc)
    print("Average Precision:", avg_prec)
    print("Average Recall:", avg_rec)


if __name__ == "__main__":
    main()